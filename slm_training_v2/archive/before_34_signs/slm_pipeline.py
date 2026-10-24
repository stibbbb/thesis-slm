"""
slm_pipeline.py -- the SLM, organised exactly like the adviser's pipeline.

HEARING -> DEAF (solid path in the diagram)
    Voice Recording -> Speech to text -> Normalization -> Reduction
    -> Tokenize -> Embedding -> Contextualize -> Space Restriction
    -> Active Context -> Transcribed Words -> Display

DEAF -> HEARING (dashed path)
    Transcribed Words -> Tokenize -> Embedding -> Active Context
    -> Sentence Completion -> Display

Every box is one method of SLMPipeline, named after the box. The two
paths are non_deaf_to_deaf() and deaf_to_non_deaf(). Each returns a Turn:
the text for the reader's screen plus what every box produced, in order,
so a demo can show the whole path.

Trained models (weights unchanged, nothing retrained):
    trained_picker.pt     word picker: Embedding, Contextualize and
                          Space Restriction for hearing -> Deaf
    trained_tinyllama.pt  TinyLlama:   Embedding and Sentence Completion
                          for Deaf -> hearing
Restructure (the DistilBERT reordering model) is not part of this
pipeline, per the adviser. TinyLlama reads the signs without their
order, so the sentences it writes are the same with or without it.

This file is the pipeline only. To try it live, run: python full_demo.py
"""

import re

import torch

import speech_input
from decoder_tinyllama import TinyLlamaForGeneration
from make_labels import normalize_word
from onnx_config import GLOSS_MAP, normalize_input
from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB, CATEGORIES, CONFIRMED_GESTURES
from word_picker import NEGATIONS, WordPicker, restrict_to_signs

PICKER_PATH = "trained_picker.pt"
DECODER_PATH = "trained_tinyllama.pt"

LOOKUP_TABLE = set(ALL_VOCAB)       # the 88 words the SLM knows
SIGNS = set(CONFIRMED_GESTURES)     # the 40 signs a Deaf reader can be shown
MAX_WORDS = 16                      # the models' position limit

PEOPLE = {w for w, c in CATEGORIES.items() if c == "Person / Family"}
QUESTIONS = {w for w, c in CATEGORIES.items() if c == "Basic Question"}
STATES = {w for w, c in CATEGORIES.items() if c == "Response / State"}
ACTIVITIES = {w for w, c in CATEGORIES.items() if c == "Daily Activity"}
PLACES = {w for w, c in CATEGORIES.items() if c == "Location / Place"}

# Spoken short forms, written out the way the training sentences are.
CONTRACTIONS = {
    "don't": "do not", "dont": "do not", "doesn't": "does not",
    "doesnt": "does not", "didn't": "did not", "didnt": "did not",
    "can't": "can not", "cant": "can not", "cannot": "can not",
    "won't": "will not", "wont": "will not", "isn't": "is not",
    "isnt": "is not", "aren't": "are not", "arent": "are not",
    "i'm": "i am", "im": "i am", "you're": "you are", "he's": "he is",
    "she's": "she is", "it's": "it is", "what's": "what is",
    "where's": "where is", "how's": "how is", "who's": "who is",
    "i'll": "i will", "you'll": "you will", "never": "not",
}

# Which remembered person "he" / "she" can point to. Teacher and friend
# can be either; family is plural, so neither pronoun refers to it.
PRONOUN_FITS = {
    "he": {"father", "teacher", "friend"},
    "she": {"mother", "teacher", "friend"},
}

# A sign counts as present in a sentence under these spellings too:
# the self-sign "my" appears as "i am", and "you" as "your" or "are".
COVER_FORMS = {"my": {"my", "i", "me", "am"}, "you": {"you", "your", "are"}}


class Turn:
    """What one pass through a path produced."""

    def __init__(self, direction):
        self.direction = direction
        self.steps = []          # (box name, what that box produced)
        self.output = ""         # the text on the reader's screen

    def add(self, box, detail):
        self.steps.append((box, detail))


class ActiveContext:
    """The Active Context box: the conversation memory both paths share.

    person               the last person mentioned (mother, father, ...)
    last_was_question    the hearing person's last turn was a question
    last_question_words  which question words it used (where, how, ...)
    addressed_to_you     that turn was asked to the Deaf person ("how are you")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.person = None
        self.last_was_question = False
        self.last_question_words = set()
        self.addressed_to_you = False
        self.turns = 0

    # ---- hearing -> Deaf ----

    def resolve_pronoun(self, spoken_words, glosses):
        """Replace he/she with the remembered person: after talking about
        father, "where is he" shows "where father" instead of "where"."""
        if self.person is None or any(g in PEOPLE for g in glosses):
            return glosses, None
        said = set(spoken_words)
        if not any(p in said and self.person in fits
                   for p, fits in PRONOUN_FITS.items()):
            return glosses, None
        # The person goes after a leading question word or "no", matching
        # the training outputs: "where father", "how mother", "why father sad".
        i = 0
        while i < len(glosses) and (glosses[i] in QUESTIONS or glosses[i] == "no"):
            i += 1
        return glosses[:i] + [self.person] + glosses[i:], self.person

    def note_hearing(self, spoken_words, glosses):
        """Remember what the hearing person just said."""
        words = [normalize_word(w) for w in spoken_words] + list(glosses)
        self.last_question_words = {w for w in words if w in QUESTIONS}
        self.last_was_question = bool(self.last_question_words)
        # Only the word "you" itself means the question was asked to the
        # Deaf person. "your father" is about father, so it must not count
        # (reading "your" as "you" made "how is your father" + "fine"
        # come out as "i am fine").
        if "you" in spoken_words:
            self.addressed_to_you = True
        elif any(w in PEOPLE for w in words):
            self.addressed_to_you = False
        self._remember_person(words)
        self.turns += 1

    # ---- Deaf -> hearing ----

    def fill_subject(self, signs):
        """If the signs need a subject and none was signed, add the
        remembered person ("where" after talking about mother means
        "where is mother"). Returns (signs, person added or None, reason)."""
        if any(s in PEOPLE for s in signs):
            return signs, None, "a person was signed"
        if "my" in signs:
            return signs, None, "'my': the signer is talking about themselves"
        if self.person is None:
            return signs, None, "no person remembered yet"
        needs_subject = (
            any(s in QUESTIONS for s in signs)
            or (any(s in STATES for s in signs)
                and not any(s in ACTIVITIES for s in signs))
            # a bare place answering "where is ...?" is about that person
            or ("where" in self.last_question_words
                and all(s in PLACES for s in signs))
        )
        if not needs_subject:
            return signs, None, "this turn does not need a subject"
        if self.addressed_to_you:
            return signs, None, "the last question was asked to the Deaf person"
        return [self.person] + signs, self.person, f"added {self.person!r} from earlier"

    def note_deaf(self, signs, sentence):
        """Remember what the Deaf person just signed."""
        words = [normalize_word(w) for w in list(signs) + sentence.split()]
        self.last_was_question = False
        self.last_question_words = set()
        if any(w in PEOPLE for w in words):
            self.addressed_to_you = False
        self._remember_person(words)
        self.turns += 1

    def _remember_person(self, words):
        for w in words:
            if w in PEOPLE:
                self.person = w


def to_declarative(sentence):
    """Answer form: "is mother home" as a reply to a question is the
    statement "mother is home"."""
    words = sentence.split()
    if len(words) > 2 and words[0] == "is":
        rest = words[1:]
        for i, w in enumerate(rest):
            if w in ("at", "home", "school"):
                return " ".join(rest[:i] + ["is"] + rest[i:])
        return " ".join(rest[:1] + ["is"] + rest[1:])
    return sentence


def covered(sign, out_words):
    """Safety-net check: does the sentence contain this sign in any form?"""
    forms = COVER_FORMS.get(sign, {sign}) | {
        k for k, v in GLOSS_MAP.items() if v == sign}
    if forms & out_words:
        return True
    # an inflected form of the sign, e.g. drive -> driving
    return any(o.startswith(sign[:4]) for o in out_words)


class SLMPipeline:
    """Both directions of the SLM, one method per pipeline box."""

    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = SimpleTokenizer(ALL_VOCAB)
        self.picker = self._load(PICKER_PATH, WordPicker)
        self.decoder = self._load(DECODER_PATH, TinyLlamaForGeneration)
        self.context = ActiveContext()

    def _load(self, path, ModelClass):
        ckpt = torch.load(path, map_location=self.device)
        cfg = {k: v for k, v in ckpt.items()
               if k not in ("model_state_dict", "vocab_size", "variant")}
        model = ModelClass(vocab_size=ckpt["vocab_size"],
                           pad_id=self.tokenizer.pad_id, **cfg).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    # ================= HEARING -> DEAF boxes =================

    def voice_recording(self, seconds=speech_input.RECORD_SECONDS):
        """Voice Recording: capture the hearing person's voice."""
        return speech_input.record(seconds)

    def speech_to_text(self, audio):
        """Speech to text: the pre-trained Vosk model turns audio into words."""
        return speech_input.transcribe(audio)

    def normalization(self, text):
        """Normalization: clean the transcript into plain lowercase words.
        - punctuation is removed ("mother?" -> "mother")
        - short forms are written out ("don't" -> "do not", "i'm" -> "i am")
        - a synonym the word picker never saw in training becomes the word
          it knows ("mommy"/"mom" -> "mother", "clothes" -> "laundry")
        Words the picker was trained on stay exactly as spoken ("okay",
        "driving"), because that is the form it learned them in.
        Returns (words, list of changes made)."""
        text = normalize_input(text).lower().replace("’", "'")
        text = re.sub(r"[^\w\s']", " ", text)
        words, changes = [], []
        for raw in text.split():
            w = raw.strip("'")
            if not w:
                continue
            if w in CONTRACTIONS:
                parts = CONTRACTIONS[w].split()
            elif w.endswith("'s") and known_form(w[:-2]) in LOOKUP_TABLE:
                parts = [known_form(w[:-2])]          # "dad's" -> "father"
            elif w in GLOSS_MAP and w not in picker_training_words():
                parts = [GLOSS_MAP[w]]
            else:
                parts = [w.replace("'", "")]
            if parts != [w]:
                changes.append(f"{w} -> {' '.join(parts)}")
            words.extend(parts)
        return words, changes

    def reduction(self, words):
        """Reduction: keep only the words in the lookup table (the 88 words
        the SLM knows); everything else is removed before tokenizing.
        Returns (kept, removed)."""
        kept = [w for w in words if w in LOOKUP_TABLE]
        removed = [w for w in words if w not in LOOKUP_TABLE]
        return kept, removed

    def contextualize(self, vectors):
        """Contextualize: the picker's transformer blocks let every word
        read the whole sentence before anything is kept or dropped."""
        mask = torch.zeros(vectors.shape[:2], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            return self.picker.contextualize(vectors, mask)

    def space_restriction(self, hidden, words):
        """Space Restriction: narrow the sentence down to the 40-sign space.
        The picker scores each word (drop, or keep in slot 1-16) and
        whether a leading "how" is needed; then only signs can survive,
        negation becomes "no", and the Hungarian algorithm gives every
        kept word its own slot. Returns (glosses, labels, is_question)."""
        mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            tag_logits, q_logits = self.picker.heads(hidden, mask)
        return restrict_to_signs(tag_logits, q_logits, words)

    def transcribed_words(self, glosses):
        """Transcribed Words (hearing -> Deaf): the signs for the Deaf reader."""
        return " ".join(glosses)

    # ================= DEAF -> HEARING boxes =================

    def transcribed_signs(self, recognized):
        """Transcribed Words (Deaf -> hearing): the words the sign
        recognizer produced, a string or a list. Recognizer labels are
        mapped to sign names ("mama" -> "mother"); anything that is not
        one of the 40 signs is set aside. Returns (signs, ignored)."""
        text = normalize_input(recognized)
        signs, ignored = [], []
        for w in self.tokenizer._clean_and_split(text):
            s = normalize_word(w)
            if s in SIGNS:
                signs.append(s)
            else:
                ignored.append(w)
        return signs, ignored

    def sentence_completion(self, vectors, signs):
        """Sentence Completion: TinyLlama writes the English sentence one
        word at a time from the embedded signs. Two rules follow:
        - answer form: a reply to a question is written as a statement
        - safety net: if the sentence left out a sign, show the signs
          instead of a fluent sentence about something else.
        Returns (sentence, notes)."""
        sentence = self.decoder.generate_from_memory(vectors, self.tokenizer)
        notes = []
        if self.context.last_was_question:
            stated = to_declarative(sentence)
            if stated != sentence:
                notes.append(f"answer form: {sentence!r} -> {stated!r}")
                sentence = stated
        missing = [s for s in signs if not covered(s, set(sentence.split()))]
        if missing:
            notes.append(f"safety net: {sentence!r} left out {missing}, showing the signs")
            sentence = " ".join(signs)
        return sentence, notes

    # ================= boxes shared by both paths =================

    def tokenize(self, words):
        """Tokenize: each word -> its ID in the lookup table."""
        return [self.tokenizer.token_to_id[w] for w in words]

    def embedding(self, ids, direction):
        """Embedding: each ID -> 64 numbers. Each direction has its own
        trained table: the picker's for speech, TinyLlama's for signs."""
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            if direction == "speech":
                return self.picker.embed(x)
            return self.decoder.embed_source(x)

    def display(self, turn, text):
        """Display: the text that appears on the reader's screen."""
        turn.output = text
        return text

    # ================= the two paths =================

    def non_deaf_to_deaf(self, text=None, seconds=speech_input.RECORD_SECONDS):
        """Hearing -> Deaf. With `text`, the microphone is skipped (typed
        input); without it, the voice is recorded and transcribed."""
        turn = Turn("hearing -> Deaf")

        if text is None:
            audio = self.voice_recording(seconds)
            turn.add("Voice Recording", f"{seconds:g} seconds of audio")
            text = self.speech_to_text(audio)
            turn.add("Speech to text", repr(text))
        else:
            turn.add("Voice Recording", "skipped: typed input")
            turn.add("Speech to text", f"typed: {text!r}")

        words, changes = self.normalization(text)
        turn.add("Normalization", " ".join(words)
                 + (f"   [{'; '.join(changes)}]" if changes else ""))

        kept, removed = self.reduction(words)
        if len(kept) > MAX_WORDS:
            removed += kept[MAX_WORDS:]
            kept = kept[:MAX_WORDS]
        turn.add("Reduction", (" ".join(kept) or "(nothing left)")
                 + (f"   [removed: {', '.join(removed)}]" if removed else "   [nothing removed]"))

        if not kept:
            self.context.note_hearing(words, [])
            turn.add("Transcribed Words", "(nothing)")
            self.display(turn, "")
            turn.add("Display", "Deaf person's screen")
            return turn

        ids = self.tokenize(kept)
        turn.add("Tokenize", str(ids))

        vectors = self.embedding(ids, "speech")
        turn.add("Embedding", f"{_count(len(ids), 'word')} x {vectors.shape[-1]} numbers")

        hidden = self.contextualize(vectors)
        turn.add("Contextualize", f"{len(self.picker.layers)} transformer blocks: "
                                  "each word now reflects the whole sentence")

        glosses, labels, is_question = self.space_restriction(hidden, kept)
        verdicts = []
        for w, lab in zip(kept, labels):
            if w in NEGATIONS:
                verdicts.append(f"{w}=negation")
            elif normalize_word(w) not in SIGNS:
                verdicts.append(f"{w}=drop (no sign)")
            else:
                verdicts.append(f"{w}=slot {lab}" if lab else f"{w}=drop")
        turn.add("Space Restriction", ", ".join(verdicts)
                 + f"; add 'how': {'yes' if is_question else 'no'}"
                 + f" -> {' '.join(glosses) or '(nothing)'}")

        glosses, resolved = self.context.resolve_pronoun(words, glosses)
        self.context.note_hearing(words, glosses)
        detail = f"remembers {self.context.person or 'nobody yet'}"
        if resolved:
            detail = f"he/she -> {resolved}; " + detail
        turn.add("Active Context", detail)

        out = self.transcribed_words(glosses)
        turn.add("Transcribed Words", out or "(nothing)")
        self.display(turn, out)
        turn.add("Display", "Deaf person's screen")
        return turn

    def deaf_to_non_deaf(self, recognized):
        """Deaf -> hearing. `recognized` is the sign recognizer's output,
        e.g. "drive mother where" or ["drive", "mother", "where"]."""
        turn = Turn("Deaf -> hearing")

        signs, ignored = self.transcribed_signs(recognized)
        signs = signs[:MAX_WORDS]
        turn.add("Transcribed Words", (" ".join(signs) or "(no signs)")
                 + (f"   [not signs, set aside: {', '.join(ignored)}]" if ignored else ""))
        if not signs:
            self.display(turn, "")
            turn.add("Display", "nothing recognised")
            return turn

        ids = self.tokenize(signs)
        turn.add("Tokenize", str(ids))

        vectors = self.embedding(ids, "signs")
        turn.add("Embedding", f"{_count(len(ids), 'sign')} x {vectors.shape[-1]} numbers")

        signs, added, reason = self.context.fill_subject(signs)
        if added:
            extra = self.embedding(self.tokenize([added]), "signs")
            vectors = torch.cat([extra, vectors], dim=1)
        turn.add("Active Context", reason)

        sentence, notes = self.sentence_completion(vectors, signs)
        turn.add("Sentence Completion", sentence
                 + (f"   [{'; '.join(notes)}]" if notes else ""))

        self.context.note_deaf(signs, sentence)
        self.display(turn, sentence)
        turn.add("Display", "hearing person's screen")
        return turn


def _count(n, noun):
    return f"{n} {noun}" + ("" if n == 1 else "s")


_training_words = None


def picker_training_words():
    """Every word in the word picker's training sentences (labels_train.jsonl).
    Normalization leaves a synonym as spoken only if the picker saw it there:
    "mom", "dad" and "house" are in the lookup table but never appear in the
    training sentences, so the picker never learned them and they are mapped
    to "mother", "father" and "home" instead."""
    global _training_words
    if _training_words is None:
        import json
        with open("labels_train.jsonl", encoding="utf-8") as f:
            _training_words = {w for line in f for w in json.loads(line)["input"].split()}
    return _training_words


def known_form(word):
    """The form of `word` the picker learned: a synonym it never saw in
    training becomes the word it knows ("dad" -> "father")."""
    if word in GLOSS_MAP and word not in picker_training_words():
        return GLOSS_MAP[word]
    return word


def format_turn(turn):
    """The path as printable lines, one per pipeline box."""
    width = max(len(box) for box, _ in turn.steps)
    lines = [f"  {turn.direction}"]
    for n, (box, detail) in enumerate(turn.steps, start=1):
        lines.append(f"    {n:>2}. {box:<{width}}  {detail}")
    return lines
