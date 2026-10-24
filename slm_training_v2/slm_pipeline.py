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
from word_picker import NEGATION_MARKER, NEGATIONS, WordPicker, restrict_to_signs

PICKER_PATH = "trained_picker.pt"
DECODER_PATH = "trained_tinyllama.pt"

LOOKUP_TABLE = set(ALL_VOCAB)       # every word the SLM knows: the signs + helper words
SIGNS = set(CONFIRMED_GESTURES)     # the 34 signs a Deaf reader can be shown
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

# "Good morning" is not in our vocabulary: the phrase is dropped, so its
# "good" is not shown as the sign GOOD (which says how someone is).
GREETING_PHRASES = re.compile(r"\bgood (morning|afternoon|evening|night)\b")

# Punctuation is shown only to the hearing reader (Deaf -> hearing): the
# Deaf reader's screen stays plain signs. A sentence that starts with one
# of these is a question and ends with "?"; any other ends with ".".
QUESTION_STARTERS = {"are", "is", "am", "do", "does", "did", "can", "could", "will", "would",
                     "was", "were", "what", "where", "why", "how", "who", "when"}

# Which remembered person "he" / "she" can point to. Teacher and friend
# can be either; family is plural, so neither pronoun refers to it.
PRONOUN_FITS = {
    "he": {"father", "teacher", "friend"},
    "she": {"mother", "teacher", "friend"},
}

# A sign counts as present in a sentence under these spellings too:
# the self-sign "my" appears as "i am", and "you" as "your" or "are".
COVER_FORMS = {"my": {"my", "i", "me", "am"}, "you": {"you", "your", "are"}}

# The seven conversation contexts from the deaf students' interviews
# ("[SLM] Data Interpretation", Possible Contexts), each with its trigger
# signs, in the order listed there (that order breaks the last ties). NOT, GO,
# SIGN, THANK YOU and SLEEP are not recorded as signs yet; they are listed
# so they work as soon as the recognizer can produce them.
TOPICS = {
    "help and repair": {"help", "please", "understand", "again", "not"},
    "school and learning": {"study", "learn", "teacher", "class", "school"},
    "whereabouts and plans": {"where", "school", "home", "now", "today", "tomorrow", "go", "going"},
    "greeting": {"hello", "nice", "friend", "deaf", "sign"},
    "courtesy": {"sorry", "please", "thank"},
    "home and family": {"mother", "father", "family", "home", "eat", "time", "sleep"},
    "feelings": {"fine", "good"},
}
REPAIR_TOPIC = "help and repair"

# Context leak and switch (adviser's consultation). Every context keeps a
# weight, so words from another context leak in without switching it; the
# whole context changes only when another context covers at least half of
# the turn's context words (e.g. 9 of 16). Each turn moves the weights
# halfway to that turn's shares, so an unmentioned context fades
# (1/2, 1/4, 1/8 ...) instead of dropping straight to 0.
TOPIC_SWITCH_SHARE = 0.5
TOPIC_FADE = 0.5

# Repair rule: when NOT + UNDERSTAND is signed, ask the hearing person to
# rephrase or write it (students say they do not understand in 4 of 5 groups).
REPAIR_SENTENCE = "i do not understand please say it again or write it"

# A sign the recognizer is less sure of than this never reaches the SLM:
# students mix in gesture and home sign, so it must not guess.
SIGN_CONFIDENCE_THRESHOLD = 0.6

# A signed NOT goes after the first of these: "i am fine" -> "i am not fine".
HELPER_VERBS = {"am", "is", "are", "was", "were", "will", "can", "do", "does"}


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
    topic                the active conversation context (one of TOPICS), or None
    topic_weights        how present each of the seven contexts is (0 to 1)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.person = None
        self.last_was_question = False
        self.last_question_words = set()
        self.addressed_to_you = False
        self.topic = None
        self.topic_weights = dict.fromkeys(TOPICS, 0.0)
        self.turns = 0

    # ---- both directions ----

    def update_topic(self, words, weights=None, repair=False):
        """Track the conversation topic, one of the seven contexts. One topic
        is active at a time; the conversation starts neutral.
        - Each context's share of the turn = its trigger words / all the
          turn's context words, each counted by how sure the recognizer was.
          A word that fits the active context is read in it ("home" while
          talking about whereabouts); only the other words leak.
        - Every context weight moves halfway to its share: words from other
          contexts leak in, and an unmentioned context fades, never jumping to 0.
        - NOT + UNDERSTAND always switches to help and repair.
        - Otherwise the topic changes only when another context covers at
          least half of the context words and more than the active one.
        - Ties go to the context with more weight from earlier turns.
        Returns a short note for the demo."""
        old, conf = self.topic, weights or {}
        context_words = {w for w in words if any(w in trig for trig in TOPICS.values())}
        total = sum(conf.get(w, 1.0) for w in context_words)
        if total == 0:
            return f"topic stays {old or 'neutral'} (no context words)"

        def contexts_of(w):
            if old is not None and w in TOPICS[old]:
                return {old}
            return {t for t, trig in TOPICS.items() if w in trig}

        share = {t: sum(conf.get(w, 1.0) for w in context_words if t in contexts_of(w)) / total
                 for t in TOPICS}
        for t in TOPICS:
            self.topic_weights[t] += TOPIC_FADE * (share[t] - self.topic_weights[t])

        def rank(t):   # a tie goes to the weight from earlier turns, then to TOPICS order
            return share[t], self.topic_weights[t]

        if repair:
            self.topic = REPAIR_TOPIC
        elif old is None:
            self.topic = max(TOPICS, key=rank)
        else:
            best = max((t for t in TOPICS if t != old), key=rank)
            if share[best] >= TOPIC_SWITCH_SHARE and share[best] > share[old]:
                self.topic = best
        if self.topic != old:
            return f"topic {old or 'neutral'} -> {self.topic} ({share[self.topic]:.0%} of the context words)"
        leaks = [f"{t} {share[t]:.0%}" for t in TOPICS if t != old and share[t] > 0]
        return f"topic stays {old}" + (f" ({', '.join(leaks)} leaks in)" if leaks else "")

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
        # The person goes after a leading question word or "not", matching
        # the training outputs: "where father", "how mother", "why father sad".
        i = 0
        while i < len(glosses) and (glosses[i] in QUESTIONS or glosses[i] == NEGATION_MARKER):
            i += 1
        return glosses[:i] + [self.person] + glosses[i:], self.person

    def note_hearing(self, spoken_words, glosses, asked=False):
        """Remember what the hearing person just said. `asked` = it was a
        yes/no question ("is your mother here?"), so the Deaf reply is
        written as a statement too ("mother is home", not "is mother home")."""
        words = [normalize_word(w) for w in spoken_words] + list(glosses)
        self.last_question_words = {w for w in words if w in QUESTIONS}
        self.last_was_question = bool(self.last_question_words) or asked
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
        """If the signs need a person and none was signed, add the
        remembered person ("where" after talking about mother means
        "where is mother"). Only signs that are about a person get one;
        nothing is filled in otherwise ("where water" stays about the
        water), as the students' interviews ask.
        Returns (signs, person added or None, reason)."""
        if any(s in PEOPLE for s in signs):
            return signs, None, "a person was signed"
        if "my" in signs:
            return signs, None, "'my': the signer is talking about themselves"
        if "you" in signs:
            return signs, None, "'you': the signs already say who"
        if self.person is None:
            return signs, None, "no person remembered yet"
        others = [s for s in signs if s not in QUESTIONS]
        about_a_person = (
            # a question with nothing else to ask about: "where", "how", "why fine"
            (any(s in QUESTIONS for s in signs) and all(s in STATES for s in others))
            # how someone is: "fine" after "how is your mother?"
            or (any(s in STATES for s in signs)
                and not any(s in ACTIVITIES for s in signs))
            # a bare place answering "where is ...?" is about that person
            or ("where" in self.last_question_words
                and all(s in PLACES for s in signs))
        )
        if not about_a_person:
            if any(s in QUESTIONS for s in signs):
                return signs, None, f"the question is about {', '.join(others)}, not a person"
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


def asked_question(text, words):
    """The hearing person asked something: typed with "?", or spoken (no
    punctuation) starting with a question word after any greeting."""
    if "?" in text:
        return True
    rest = [w for w in words if w not in ("hello", "hi", "hey")]
    return bool(rest) and rest[0] in QUESTION_STARTERS


# Words a negation passes over to reach the word it negates ("do not BE sad").
NEGATION_SKIP = {"am", "is", "are", "was", "were", "be", "do", "does", "did", "will", "can",
                 "a", "an", "the", "to", "it", "very", "really", "so", "still"}


def negated_word(words, i):
    """The word the negation at words[i] negates ("do not worry" -> "worry"),
    or None when nothing follows."""
    for w in words[i + 1:]:
        if w not in NEGATION_SKIP and w not in NEGATIONS:
            return w
    return None


# "can you", "are you", "do i" ... inside a sentence also make it a question
# ("sorry can you say it again?").
SUBJECTS = {"you", "i", "he", "she", "it", "we", "they"}


def punctuate(sentence):
    """The hearing reader's sentence with its end mark: "where is the
    water?", "sorry where father?", "mother is home." (Deaf -> hearing
    only). A signed question word (what, where, why, how) makes it a question."""
    words = sentence.split()
    if not words:
        return sentence
    asked = (words[0] in QUESTION_STARTERS or any(w in QUESTIONS for w in words)
             or any(a in QUESTION_STARTERS and b in SUBJECTS for a, b in zip(words, words[1:])))
    return sentence + ("?" if asked else ".")


def to_declarative(sentence):
    """Answer form: "is mother home" as a reply to a question is the
    statement "mother is home"."""
    words = sentence.split()
    if len(words) > 2 and words[0] == "is":
        rest = words[1:]
        for i, w in enumerate(rest):
            if w in ("at", "in", "home", "here"):
                return " ".join(rest[:i] + ["is"] + rest[i:])
        return " ".join(rest[:1] + ["is"] + rest[1:])
    return sentence


def negate(sentence):
    """Put a signed NOT into the sentence, so a negation is never lost:
    "i am fine" -> "i am not fine", "is mother home" -> "mother is not home",
    "i want water" -> "i do not want water", "help me" -> "do not help me"."""
    words = to_declarative(sentence).split()
    if not words:
        return "no"
    if words[0] in HELPER_VERBS:                      # a question: "are you studying"
        return " ".join(words[:2] + ["not"] + words[2:])
    for i, w in enumerate(words):
        if w in HELPER_VERBS:
            return " ".join(words[:i + 1] + ["not"] + words[i + 1:])
    if words[0] in ("i", "you", "we"):
        return " ".join(words[:1] + ["do", "not"] + words[1:])
    return "do not " + " ".join(words)


def covered(sign, out_words):
    """Safety-net check: does the sentence contain this sign in any form?"""
    forms = COVER_FORMS.get(sign, {sign}) | {
        k for k, v in GLOSS_MAP.items() if v == sign}
    if forms & out_words:
        return True
    # an inflected form of the sign, e.g. study -> studying
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

    def voice_recording(self, max_seconds=speech_input.MAX_RECORD_SECONDS):
        """Voice Recording: capture the hearing person's voice, stopping as
        soon as they stop talking. Returns (audio, why it stopped)."""
        return speech_input.record_until_silence(max_seconds)

    def speech_to_text(self, audio):
        """Speech to text: the pre-trained Vosk model turns audio into words."""
        return speech_input.transcribe(audio)

    def normalization(self, text):
        """Normalization: clean the transcript into plain lowercase words.
        - punctuation is removed ("mother?" -> "mother")
        - short forms are written out ("don't" -> "do not", "i'm" -> "i am")
        - a synonym the word picker never saw in training becomes the word
          it knows ("mommy"/"mom" -> "mother", "classroom" -> "class")
        - a greeting phrase outside our vocabulary is dropped ("good morning")
        Words the picker was trained on stay exactly as spoken ("okay",
        "driving"), because that is the form it learned them in.
        Returns (words, list of changes made)."""
        text = normalize_input(text).lower().replace("’", "'")
        words, changes = [], []
        for m in GREETING_PHRASES.finditer(text):
            changes.append(f"{m.group(0)} -> dropped")
        text = GREETING_PHRASES.sub(" ", text)
        text = re.sub(r"[^\w\s']", " ", text)
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
        """Reduction: keep only the words in the lookup table (the words
        the SLM knows); everything else is removed before tokenizing.
        A negation goes with the word it negates: "do not worry" loses
        "worry", so its "not" is removed too, or "please do not worry, it
        is fine" would reach the Deaf reader as "please not fine".
        Returns (kept, removed)."""
        kept, removed = [], []
        for i, w in enumerate(words):
            target = negated_word(words, i) if w in NEGATIONS else None
            if target is not None and target not in LOOKUP_TABLE:
                removed.append(f"{w} (with {target})")      # a bare "no" stays
            elif w in LOOKUP_TABLE:
                kept.append(w)
            else:
                removed.append(w)
        return kept, removed

    def contextualize(self, vectors):
        """Contextualize: the picker's transformer blocks let every word
        read the whole sentence before anything is kept or dropped."""
        mask = torch.zeros(vectors.shape[:2], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            return self.picker.contextualize(vectors, mask)

    def space_restriction(self, hidden, words):
        """Space Restriction: narrow the sentence down to the 34-sign space.
        The picker scores each word (drop or keep) and whether a leading
        "how" is needed; then only signs can survive, every sign except
        "my"/"you" is kept, the signs keep their spoken order, and a spoken
        negation is shown as "not". Returns (glosses, labels, is_question)."""
        mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            tag_logits, q_logits = self.picker.heads(hidden, mask)
        return restrict_to_signs(tag_logits, q_logits, words)

    def transcribed_words(self, glosses):
        """Transcribed Words (hearing -> Deaf): the signs for the Deaf reader,
        plain words without punctuation."""
        return " ".join(glosses)

    # ================= DEAF -> HEARING boxes =================

    def transcribed_signs(self, recognized):
        """Transcribed Words (Deaf -> hearing): what the sign recognizer
        produced. It can be text ("where mother"), a list of words, or a
        list of (sign, confidence) pairs; typed text can give a confidence
        as "mother:0.9".
        - A sign below SIGN_CONFIDENCE_THRESHOLD is dropped as unsure, so
          the SLM never guesses from gesture or home sign.
        - A negation (NOT) is kept as a flag and never thrown away.
        - Labels become sign names ("mama" -> "mother"); anything that is
          not one of the signs is set aside.
        Returns (signs, notes): notes holds negated, unsure, ignored and
        each sign's confidence."""
        items = []                                    # (word, confidence or None)
        if isinstance(recognized, (list, tuple)):
            for r in recognized:
                if isinstance(r, dict):
                    items.append((str(r.get("sign") or r.get("word") or ""), r.get("confidence")))
                elif isinstance(r, (list, tuple)) and len(r) == 2:
                    items.append((str(r[0]), r[1]))
                else:
                    items += [(w, None) for w in str(r).split()]
        else:
            for raw in normalize_input(recognized).split():
                word, _, conf = raw.partition(":")
                try:
                    c = float(conf) if conf else None
                except ValueError:
                    word, c = raw, None
                if word.lower().strip(".,!?") in NEGATIONS:   # "don't" must stay one word
                    items.append(("not", c))
                    continue
                items += [(w, c) for w in self.tokenizer._clean_and_split(word)]

        signs, ignored, unsure, confidence, negated = [], [], [], {}, False
        for w, c in items:
            w = w.lower().strip()
            if not w:
                continue
            if c is not None and float(c) < SIGN_CONFIDENCE_THRESHOLD:
                unsure.append(f"{w} ({float(c):.2f})")
            elif w in NEGATIONS:
                negated = True
            elif normalize_word(w) in SIGNS:
                signs.append(normalize_word(w))
                confidence[signs[-1]] = 1.0 if c is None else float(c)
            else:
                ignored.append(w)
        return signs, {"negated": negated, "unsure": unsure, "ignored": ignored,
                       "confidence": confidence}

    def sentence_completion(self, vectors, signs, negated=False):
        """Sentence Completion: TinyLlama writes the English sentence one
        word at a time from the embedded signs. Three rules follow:
        - answer form: a reply to a question is written as a statement
        - safety net: if the sentence left out a sign, show the signs
          instead of a fluent sentence about something else
        - negation: a signed NOT is put into the sentence, never dropped.
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
            sentence = " ".join((["not"] if negated else []) + signs)
        elif negated:
            negative = negate(sentence)
            notes.append(f"negation: {sentence!r} -> {negative!r}")
            sentence = negative
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

    def non_deaf_to_deaf(self, text=None, max_seconds=speech_input.MAX_RECORD_SECONDS):
        """Hearing -> Deaf. With `text`, the microphone is skipped (typed
        input); without it, the voice is recorded until the speaker stops
        talking, then transcribed."""
        turn = Turn("hearing -> Deaf")

        if text is None:
            audio, why = self.voice_recording(max_seconds)
            seconds = len(audio) / (2 * speech_input.SAMPLE_RATE)
            turn.add("Voice Recording", f"{seconds:.1f} seconds of audio ({why})")
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
            self.context.note_hearing(words, [], asked_question(text, words))
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
        self.context.note_hearing(words, glosses, asked_question(text, words))
        heard = {normalize_word(w) for w in words}
        repair = any(w in NEGATIONS for w in words) and "understand" in heard
        topic_note = self.context.update_topic(heard | set(glosses), repair=repair)
        detail = f"remembers {self.context.person or 'nobody yet'}; {topic_note}"
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
        e.g. "where mother" or ["where", "mother"]."""
        turn = Turn("Deaf -> hearing")

        signs, found = self.transcribed_signs(recognized)
        signs = signs[:MAX_WORDS]
        negated = found["negated"]
        extras = []
        if found["unsure"]:
            extras.append("unsure, dropped: " + ", ".join(found["unsure"]))
        if found["ignored"]:
            extras.append("not signs, set aside: " + ", ".join(found["ignored"]))
        turn.add("Transcribed Words", (" ".join((["NOT"] if negated else []) + signs) or "(no signs)")
                 + (f"   [{'; '.join(extras)}]" if extras else ""))

        if not signs:
            # only a negation was signed: it still means "no"
            self.display(turn, "no." if negated else "")
            turn.add("Display", "hearing person's screen" if negated else "nothing recognised")
            return turn

        ids = self.tokenize(signs)
        turn.add("Tokenize", str(ids))

        vectors = self.embedding(ids, "signs")
        turn.add("Embedding", f"{_count(len(ids), 'sign')} x {vectors.shape[-1]} numbers")

        # Active Context. Repair rule: NOT + UNDERSTAND asks the hearing
        # person to rephrase, whatever the active topic.
        repair = negated and "understand" in signs
        topic_note = self.context.update_topic(signs + (["not"] if negated else []),
                                               found["confidence"], repair)
        if repair:
            turn.add("Active Context", f"NOT + UNDERSTAND = repair; {topic_note}")
        else:
            signs, added, reason = self.context.fill_subject(signs)
            if added:
                extra = self.embedding(self.tokenize([added]), "signs")
                vectors = torch.cat([extra, vectors], dim=1)
            turn.add("Active Context", f"{reason}; {topic_note}")

        if repair:
            sentence, notes = REPAIR_SENTENCE, ["repair rule"]
        else:
            sentence, notes = self.sentence_completion(vectors, signs, negated)
        turn.add("Sentence Completion", sentence
                 + (f"   [{'; '.join(notes)}]" if notes else ""))

        self.context.note_deaf(signs, sentence)
        self.display(turn, punctuate(sentence))
        turn.add("Display", "hearing person's screen (with punctuation)")
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
