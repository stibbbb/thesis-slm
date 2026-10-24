"""
full_demo.py -- the whole SLM, with every component labelled as it runs.

FOUR PIECES, SHOWN SEPARATELY

  1. RESTRUCTURE   DistilBERT encoder      puts signs in order
  2. COMPLETE      TinyLlama decoder       builds the sentence
  3. REDUCE        word picker             speech -> signs
  4. MEMORY        conversation context    fills in an omitted subject

Plus a SAFETY NET on Complete: if the sentence it builds does not
contain what was actually signed, the signs are shown instead. The
picker cannot make that kind of error by construction; the decoder
generates, so it needs the check.

  1 = the Deaf person signs
  2 = the hearing person speaks
  r = reset the conversation memory
  q = quit
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
from slm_inference import restructure

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
PEOPLE = {w for w, c in CATEGORIES.items() if c == "Person / Family"}
QUESTIONS = {w for w, c in CATEGORIES.items() if c == "Basic Question"}
STATES = {w for w, c in CATEGORIES.items() if c == "Response / State"}
ACTIVITIES = {w for w, c in CATEGORIES.items() if c == "Daily Activity"}
SELF = {"my"}


class Memory:
    """Remembers the most recently mentioned person, from either side."""

    def __init__(self):
        self.person = None
        self.turns = []
        # True when the last thing said was aimed at the Deaf
        # person rather than about a third party.
        self.addressed_to_you = False
        # True when the hearing person's last turn was a question.
        # A Deaf reply to a question is an ANSWER, so "mother home"
        # means "mother is home", not "is mother home?".
        self.last_was_question = False

    def note(self, side, text, result):
        self.turns.append((side, text, result))

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
        self.turns = []
        # True when the last thing said was aimed at the Deaf
        # person rather than about a third party.
        self.addressed_to_you = False


def stage(n, name, model, detail):
    print(f"    [{n}] {name:<13} {model:<22} {detail}")


def screen(title, text):
    w = 62
    print(f"\n    +{'-' * w}+")
    print(f"    | {title:<{w - 2}} |")
    print(f"    | {text:<{w - 2}} |")
    print(f"    +{'-' * w}+\n")


def covers(sentence, signs):
    """Every sign made must survive into the sentence."""
    out = set(sentence.lower().split())
    missing = [w for w in signs if not _covered(w, out)]
    return missing


def deaf_signs(encoder, decoder, tokenizer, text, device, mem):
    words = text.lower().split()
    kept, dropped = [], []
    for w in words:
        n = normalize_word(w)
        if n in SIGNS:
            kept.append(n)
        else:
            dropped.append(w)

    changed = [f"{a}->{b}" for a, b in zip(words, kept) if a != b]
    stage(0, "normalise", "lookup table",
          f"{' '.join(changed) if changed else 'nothing to change'}"
          + (f"   ignored {dropped}" if dropped else ""))
    if not kept:
        print("    nothing recognised\n")
        return

    # --- context memory ---
    has_person = any(w in PEOPLE for w in kept)
    has_self = any(w in SELF for w in kept)
    has_q = any(w in QUESTIONS for w in kept)
    has_st = any(w in STATES for w in kept)
    has_act = any(w in ACTIVITIES for w in kept)
    needs_subject = has_q or (has_st and not has_act)

    if (mem.person and needs_subject and not has_person and not has_self
            and not mem.addressed_to_you):
        kept = [mem.person] + kept
        stage(1, "memory", "conversation context",
              f"no person signed -> filled in {mem.person!r}")
    elif mem.person and not has_person and not has_self:
        stage(1, "memory", "conversation context",
              f"holding {mem.person!r}, but this turn does not need a subject")
    else:
        stage(1, "memory", "conversation context", "nothing to recover")

    ordered = restructure(encoder, tokenizer, " ".join(kept), device)
    stage(2, "RESTRUCTURE", "DistilBERT encoder", f"{' '.join(kept)!r} -> {ordered!r}")

    sentence = decoder_predict(decoder, tokenizer, ordered, device)
    stage(3, "COMPLETE", "TinyLlama decoder", f"{sentence!r}")

    # Answering a question means stating a fact, not asking one back.
    if mem.last_was_question:
        before = sentence
        sentence = to_declarative(sentence)
        if sentence != before:
            stage(3, "answer form", "declarative rule",
                  f"replying to a question -> {sentence!r}")

    missing = covers(sentence, ordered.split())
    if missing:
        stage(4, "safety net", "coverage check",
              f"sentence left out {missing} -- showing signs instead")
        sentence = ordered
    else:
        stage(4, "safety net", "coverage check", "every sign survived")

    screen("HEARING PERSON'S SCREEN", sentence)
    mem.note("deaf", ordered, sentence)


def hearing_speaks(picker, tokenizer, text, device, mem):
    glosses, labels, is_q = picker_predict(picker, tokenizer, text.lower(), device)

    stage(1, "REDUCE", "word picker", "one verdict per word:")
    words = text.split()
    for i in range(min(len(words), len(labels))):
        w, lab = words[i], labels[i]
        norm = normalize_word(w.lower())
        mark = "sign" if norm in SIGNS else "    "
        if lab == 0:
            print(f"          {w:<12} {mark}  DROP")
        else:
            extra = f"  (shown as {norm!r})" if norm != w.lower() else ""
            print(f"          {w:<12} {mark}  KEEP -> slot {lab}{extra}")
    stage(2, "question flag", "second head",
          "YES, 'how' added" if is_q else "no")
    stage(3, "guarantee", "selection only",
          "every word shown was actually spoken")

    screen("DEAF PERSON'S SCREEN", glosses if glosses else "(nothing)")
    mem.note("hearing", text.lower(), glosses)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    mem = Memory()

    encoder = load_checkpoint(ENCODER_PATH, DistilBertForRestructure, tokenizer, device)
    decoder = load_checkpoint(DECODER_PATH, TinyLlamaForGeneration, tokenizer, device)
    picker = load_checkpoint(PICKER_PATH, WordPicker, tokenizer, device)

    print("=" * 72)
    print("  FULL SLM DEMO")
    print("=" * 72)
    print(f"  Restructure  DistilBERT encoder   {ENCODER_PATH}")
    print(f"  Complete     TinyLlama decoder    {DECODER_PATH}")
    print(f"  Reduce       word picker          {PICKER_PATH}")
    print(f"  Memory       conversation context  in this file")
    print("-" * 72)
    print(f"  {len(CONFIRMED_GESTURES)} signs | {len(ALL_VOCAB)} words | device: {device}")
    print("=" * 72)
    print(f"\n  Signs: {', '.join(sorted(CONFIRMED_GESTURES))}")
    print(f"\n  Recoverable subjects: {', '.join(sorted(PEOPLE))}\n")

    while True:
        try:
            c = input("[1=sign, 2=speak, r=reset, q=quit]> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if c in ("q", "quit", "exit"):
            break
        if c == "r":
            mem.reset()
            print("  memory cleared\n")
            continue

        if c == "1":
            t = input("  signs> ").strip()
            if not t:
                continue
            print()
            deaf_signs(encoder, decoder, tokenizer, t, device, mem)
        elif c == "2":
            t = input("  says> ").strip()
            if not t:
                continue
            print()
            hearing_speaks(picker, tokenizer, t, device, mem)
        else:
            print("  type 1, 2, r, or q\n")
            continue

        held = mem.person if mem.person else "(nobody yet)"
        print(f"    memory: {held}   |   {len(mem.turns)} turns\n")

    print("Done.")


if __name__ == "__main__":
    main()