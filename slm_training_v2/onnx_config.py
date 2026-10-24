"""
Shared constants and helpers for ONNX export and inference.
Deliberately has ZERO PyTorch dependency -- this is the only config
file onnx_inference.py needs, so Raspberry Pi deployment requires
only onnxruntime + scipy + numpy, never PyTorch.

Also the single source of truth for input validation and the gloss
alias mapping, shared by BOTH the PyTorch (slm_inference.py) and ONNX
(onnx_inference.py) inference paths, so the two can never drift out
of sync.
"""

from vocab_pool import ALL_VOCAB

ENCODER_ONNX_PATH = "distilbert_encoder.onnx"
DECODER_ONNX_PATH = "tinyllama_decoder.onnx"

ENCODER_MAX_LEN = 16
DECODER_SRC_LEN = 16
DECODER_TGT_LEN = 32

# THE SINGLE SOURCE OF TRUTH for word normalisation.
#
# Every path -- training labels, PyTorch inference, ONNX inference --
# imports this one table. It previously existed twice, in two different
# versions, which meant the model was trained under one convention and
# run under another: "hi" worked in one path and was discarded in the
# other.
#
# Two kinds of entry:
#   1. True synonyms      "mom" -> "mother"
#   2. Inflected forms    "driving" -> "drive"
# Gloss output uses the BASE form, because that is the name of the sign
# the Deaf reader knows.
#
# A word that is itself a sign must NEVER appear as a key here. "good"
# used to map to "fine", which was correct when "good" was not a sign;
# now that it is one, collapsing them would destroy a real distinction.
GLOSS_MAP = {
    # --- person synonyms
    "mama": "mother", "papa": "father",
    "mom": "mother", "mommy": "mother", "ma": "mother",
    "dad": "father", "daddy": "father", "pa": "father",

    # --- place / object synonyms
    "house": "home", "residence": "home",
    "classroom": "class", "classes": "class",

    # --- state synonyms  (NOTE: "good" is excluded -- it is its own sign)
    "okay": "fine", "alright": "fine", "great": "fine",

    # --- inflected verb forms fold onto the sign name
    "studying": "study", "studies": "study", "studied": "study",
    "eating": "eat", "eats": "eat", "ate": "eat",
    "learning": "learn", "learns": "learn", "learned": "learn",
    "does": "doing", "do": "doing",
    "go": "going",

    # --- possessive / pronoun forms
    "your": "you",
    "me": "my", "mine": "my",

    # --- greetings and time
    "hi": "hello", "hey": "hello",
    "tmrw": "tomorrow", "tonight": "today",
}

KNOWN_WORDS = set(ALL_VOCAB)


def pad_ids(ids, length, pad_id):
    ids = list(ids)[:length]
    return ids + [pad_id] * (length - len(ids))


def normalize_input(text):
    if text is None:
        raise TypeError(
            "SLM received None as input. The caller must send a string "
            "(or list of words), even an empty one ('' or [])."
        )
    if isinstance(text, (list, tuple)):
        text = " ".join(str(w) for w in text)
    if not isinstance(text, str):
        raise TypeError(
            f"SLM received input of type {type(text).__name__}, expected "
            f"a string or a list of word strings."
        )
    return text.strip()


def map_glosses(text):
    """
    Resolves aliases (GLOSS_MAP) to their canonical trained form, then
    de-duplicates words that collapsed onto the same canonical form
    via aliasing (e.g. "wash" and "clothes" both -> "laundry").
    Preserves first-occurrence order. The model was never trained on
    duplicate tokens, so this prevents alias expansion from creating
    a new, unfamiliar input pattern of its own.
    """
    words = text.split() if isinstance(text, str) else list(text)
    mapped = [GLOSS_MAP.get(w.lower(), w.lower()) for w in words]
    seen = set()
    deduped = []
    for w in mapped:
        if w not in seen:
            seen.add(w)
            deduped.append(w)
    return " ".join(deduped)


def sanitize_glosses(words, warn=True):
    kept, dropped = [], []
    for w in words:
        if w in KNOWN_WORDS:
            kept.append(w)
        else:
            dropped.append(w)
    if warn and dropped:
        print(f"[SLM WARNING] Unrecognized word(s) dropped: {dropped} "
              f"-- not in the trained vocabulary.")
    return kept, dropped


def enforce_max_len(ids, max_len, label="sequence"):
    if len(ids) > max_len:
        print(f"[SLM WARNING] {label} length {len(ids)} exceeds max_len="
              f"{max_len}; truncating. With the current sign "
              f"vocabulary this should not happen in normal use -- if "
              f"it does, check the recognition module's output before "
              f"trusting the result.")
        ids = ids[:max_len]
    return ids


# --- Reduce confidence signal (see slm_progress_notes.md \u00a712 / \u00a714,
# --- known issue 14: Reduce has no OOV robustness).
OOV_CONFIDENCE_THRESHOLD = 0.5


def oov_density(text):
    words = [w.strip(".,?!").lower() for w in text.split()]
    if not words:
        return 0.0, []
    oov_words = [w for w in words if w not in KNOWN_WORDS]
    return len(oov_words) / len(words), oov_words


def reduce_confidence(text, threshold=OOV_CONFIDENCE_THRESHOLD):
    density, oov_words = oov_density(text)
    return {
        "density": density,
        "low_confidence": density > threshold,
        "oov_words": oov_words,
    }