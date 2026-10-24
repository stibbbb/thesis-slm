"""
two_way_demo.py -- test BOTH directions, with conversation memory.

  1 = the DEAF person signs      -> a full sentence for the hearing person
  2 = the HEARING person speaks  -> key signs for the Deaf person's screen
  r = reset the conversation memory
  q = quit

CONVERSATION MEMORY
Real conversations drop the subject once it is established:

    "Where is mother?"  ...  "Is she home?"

The second sentence never repeats "mother", but a person understands it.
This demo does the same. It remembers the most recently mentioned PERSON
-- from EITHER side of the conversation -- and when the Deaf person signs
something with no person in it, that remembered person is filled back in.

    sign "mother drive"   -> "mother is driving"      (mother remembered)
    sign "where"          -> "where is mother"        (recovered)

Recovery is deliberately skipped when the signer uses "my", because that
is an explicit self-reference: "my fine" means "I am fine", not
"mother is fine".

The remembered person is shown after every turn, so you can see exactly
what the system is holding on to.
"""

import torch

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB, CONFIRMED_GESTURES, CATEGORIES
from word_picker import WordPicker, picker_predict
from decoder_tinyllama import TinyLlamaForGeneration
from encoder_distilbert import DistilBertForRestructure
from compare import load_checkpoint, decoder_predict
from make_labels import normalize_word
from onnx_config import GLOSS_MAP

# A sign is "covered" by more than its own spelling. The self-reference
# sign "my" surfaces as "i am"; "you" surfaces as "your" or "are". The
# check rejected "my sick" -> "i am sick" because the literal word "my"
# was absent, and showed the raw signs instead of a correct sentence.
COVER_FORMS = {
    "my": {"my", "i", "me", "am"},
    "you": {"you", "your", "are"},
}


def _covered(sign, out_words):
    forms = COVER_FORMS.get(sign, {sign}) | {
        k for k, v in GLOSS_MAP.items() if v == sign}
    if forms & out_words:
        return True
    # an inflected form of the sign, e.g. drive -> driving
    return any(o.startswith(sign[:4]) for o in out_words)


def to_declarative(sentence):
    """
    Turn a yes/no question into a statement.

    "mother home" alone is ambiguous -- it could be "is mother home?"
    or "mother is home". As an ANSWER to "where is mother?" it is the
    statement. This rewrites the question form when the previous turn
    was a question.
    """
    words = sentence.split()
    if len(words) > 2 and words[0] == "is":
        rest = words[1:]
        # "is her mother at school" -> "her mother is at school"
        for i, w in enumerate(rest):
            if w in ("at", "home", "school"):
                return " ".join(rest[:i] + ["is"] + rest[i:])
        return " ".join(rest[:1] + ["is"] + rest[1:])
    return sentence


PICKER_PATH = "trained_picker.pt"
DECODER_PATH = "trained_tinyllama.pt"
ENCODER_PATH = "trained_distilbert.pt"

SIGNS = set(CONFIRMED_GESTURES)
# Only these can be recovered as the subject of a later bare gesture.
PEOPLE = {w for w, c in CATEGORIES.items() if c == "Person / Family"}
# An explicit self-reference blocks recovery.
SELF = {"my"}
# A question with no subject is INCOMPLETE -- "where?" on its own is not
# a sentence, so the remembered person belongs in it.
QUESTIONS = {w for w, c in CATEGORIES.items() if c == "Basic Question"}
# A state with no subject is ambiguous, and in a conversation about
# someone it usually refers to them: "How is mother?" ... "Fine."
STATES = {w for w, c in CATEGORIES.items() if c == "Response / State"}
# An activity with no subject is already COMPLETE -- a bare "eat" means
# "I am eating". Filling in a person there changes the meaning to
# something the signer did not say, so recovery must not fire.
ACTIVITIES = {w for w, c in CATEGORIES.items() if c == "Daily Activity"}


class Memory:
    """Remembers the most recently mentioned person, and the turns so far."""

    def __init__(self):
        self.person = None
        self.history = []
        self.addressed_to_you = False
        # True when the hearing person's last turn was a question.
        # A Deaf reply to a question is an ANSWER, so "mother home"
        # means "mother is home", not "is mother home?".
        self.last_was_question = False

    def note(self, side, text, result):
        self.history.append((side, text, result))

        # WHO was the last turn about? Not just who was mentioned.
        # "How are you?" is addressed to the Deaf person, so a bare
        # "fine" in reply means THEY are fine -- even if father was the
        # subject a moment earlier. Tracking only the last person
        # mentioned got this wrong and answered "father is fine".
        words = [normalize_word(w.lower()) for w in (text + " " + result).split()]
        if side == "hearing":
            self.last_was_question = any(w in QUESTIONS for w in words)
            if "you" in words:
                self.addressed_to_you = True
            elif any(w in PEOPLE for w in words):
                self.addressed_to_you = False
        else:
            # a Deaf turn answers the question, so it is used up
            self.last_was_question = False
            if any(w in PEOPLE for w in words):
                self.addressed_to_you = False

        for w in words:
            if w in PEOPLE:
                self.person = w

    def reset(self):
        self.person = None
        self.history = []
        self.addressed_to_you = False


def box(title, text):
    w = 60
    print(f"    +{'-' * w}+")
    print(f"    | {title:<{w - 2}} |")
    print(f"    | {text:<{w - 2}} |")
    print(f"    +{'-' * w}+")


def show_decisions(sentence, labels, is_question):
    words = sentence.split()
    n = min(len(words), len(labels))
    print("    word-by-word decision:")
    for i in range(n):
        w, lab = words[i], labels[i]
        norm = normalize_word(w.lower())
        mark = "sign" if norm in SIGNS else "    "
        if lab == 0:
            print(f"      {w:<12} {mark}  DROP")
        else:
            extra = f"   (shown as '{norm}')" if norm != w.lower() else ""
            print(f"      {w:<12} {mark}  KEEP -> position {lab}{extra}")
    if is_question:
        print("    wellbeing question? YES -- 'how' added at the front")


def sign_direction(encoder, decoder, tokenizer, text, device, mem):
    from slm_inference import restructure

    words = text.lower().split()
    kept, dropped = [], []
    for w in words:
        n = normalize_word(w)
        if n in SIGNS:
            if n != w:
                print(f"    [{w!r} is the sign {n!r}]")
            kept.append(n)
        else:
            dropped.append(w)
    if dropped:
        print(f"    [ignored -- not signs the camera can produce] {dropped}")
    if not kept:
        return "", ""

    # --- context recovery ---
    # Only fill in a remembered person when the signs actually NEED a
    # subject. A question does ("where?"). A state usually does in a
    # conversation about someone ("fine" answering "how is mother?").
    # An activity does NOT -- a bare "eat" already means "I am eating",
    # and inserting a person would put words in the signer's hands.
    has_person = any(w in PEOPLE for w in kept)
    has_self = any(w in SELF for w in kept)
    has_question = any(w in QUESTIONS for w in kept)
    has_state = any(w in STATES for w in kept)
    has_activity = any(w in ACTIVITIES for w in kept)

    needs_subject = has_question or (has_state and not has_activity)

    if (mem.person and needs_subject and not has_person and not has_self
            and not mem.addressed_to_you):
        print(f"    [context] no person signed -- filling in {mem.person!r} "
              f"from earlier in the conversation")
        kept = [mem.person] + kept
    elif mem.person and not has_person and not has_self and has_activity:
        print(f"    [context] activity with no person -- taking this as the "
              f"signer's own action, not {mem.person!r}")

    cleaned = " ".join(kept)
    ordered = restructure(encoder, tokenizer, cleaned, device)
    sentence = decoder_predict(decoder, tokenizer, ordered, device)

    # Safety net: every sign made must appear in the sentence. The
    # decoder GENERATES, so it can answer about something nobody
    # signed. A plain list of signs is worse English but it is honest.
    if mem.last_was_question:
        sentence = to_declarative(sentence)

    signs = ordered.split()
    out = set(sentence.lower().split())
    missing = [w for w in signs if not _covered(w, out)]
    if missing:
        print(f"    [warning] the sentence left out {missing} -- this sign "
              f"combination was never taught, so the signs are shown as-is")
        sentence = " ".join(signs)
    return ordered, sentence


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    mem = Memory()

    picker = load_checkpoint(PICKER_PATH, WordPicker, tokenizer, device)
    decoder = load_checkpoint(DECODER_PATH, TinyLlamaForGeneration,
                              tokenizer, device)
    encoder = load_checkpoint(ENCODER_PATH, DistilBertForRestructure,
                              tokenizer, device)

    print("=" * 70)
    print("  TWO-WAY CONVERSATION DEMO  (with memory)")
    print(f"  {len(CONFIRMED_GESTURES)} signs   |   "
          f"{len(ALL_VOCAB)} vocabulary words   |   device: {device}")
    print("=" * 70)
    print("  1  = Deaf person signs      2  = hearing person speaks")
    print("  r  = reset memory           q  = quit")
    print("=" * 70)
    print(f"\n  Signs: {', '.join(sorted(CONFIRMED_GESTURES))}")
    print(f"\n  Recoverable subjects: {', '.join(sorted(PEOPLE))}\n")

    while True:
        try:
            choice = input("[1=sign, 2=speak, r=reset, q=quit]> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice in ("q", "quit", "exit"):
            break

        if choice == "r":
            mem.reset()
            print("  memory cleared\n")
            continue

        if choice == "1":
            text = input("  signs> ").strip()
            if not text:
                continue
            print()
            ordered, sentence = sign_direction(encoder, decoder, tokenizer,
                                               text, device, mem)
            if not sentence:
                print("    nothing recognised\n")
                continue
            print(f"    signs in order : {ordered!r}")
            print()
            box("HEARING PERSON'S SCREEN:", sentence)
            mem.note("deaf", ordered, sentence)

        elif choice == "2":
            text = input("  says> ").strip()
            if not text:
                continue
            print()
            glosses, labels, is_q = picker_predict(picker, tokenizer,
                                                   text.lower(), device)
            show_decisions(text, labels, is_q)
            print()
            box("DEAF PERSON'S SCREEN:", glosses if glosses else "(nothing)")
            mem.note("hearing", text.lower(), glosses)

        else:
            print("  type 1, 2, r, or q\n")
            continue

        held = mem.person if mem.person else "(nobody yet)"
        print(f"    memory: currently talking about {held}   "
              f"| {len(mem.history)} turns so far\n")

    print("Done.")


if __name__ == "__main__":
    main()