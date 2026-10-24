"""
slm_api.py -- The ONE file the recognition module needs to import.
"""

import threading

import onnxruntime as ort

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB, CONFIRMED_GESTURES
from onnx_config import (
    ENCODER_ONNX_PATH, DECODER_ONNX_PATH, normalize_input, reduce_confidence,
)
from onnx_inference import (
    sign_to_sentence_onnx, sentence_to_glosses_onnx,
    map_glosses, restructure_onnx, generate_onnx,
)

_lock = threading.Lock()
_state = {"tokenizer": None, "enc_session": None, "dec_session": None}


def _ensure_loaded():
    if _state["tokenizer"] is not None:
        return
    with _lock:
        if _state["tokenizer"] is not None:
            return
        _state["tokenizer"] = SimpleTokenizer(ALL_VOCAB)
        _state["enc_session"] = ort.InferenceSession(ENCODER_ONNX_PATH)
        _state["dec_session"] = ort.InferenceSession(DECODER_ONNX_PATH)


def translate_signs(glosses):
    _ensure_loaded()
    return sign_to_sentence_onnx(
        _state["enc_session"], _state["dec_session"], _state["tokenizer"], glosses
    )


def translate_signs_verbose(glosses):
    _ensure_loaded()
    mapped = map_glosses(glosses)
    restructured = restructure_onnx(_state["enc_session"], _state["tokenizer"], mapped)
    sentence = "" if not restructured else generate_onnx(
        _state["dec_session"], _state["tokenizer"], restructured
    )
    return {
        "input": glosses,
        "mapped": mapped,
        "restructured": restructured,
        "sentence": sentence,
    }


def translate_speech(sentence):
    _ensure_loaded()
    return sentence_to_glosses_onnx(_state["dec_session"], _state["tokenizer"], sentence)


def translate_speech_with_confidence(sentence):
    """
    Same job as translate_speech(), but also reports whether the input
    contains enough unfamiliar vocabulary that the result should be
    treated with caution -- see onnx_config.reduce_confidence() and
    slm_progress_notes.md \u00a712/\u00a714 for why this exists and what it
    does and does not solve. translate_speech() itself is unchanged;
    this does not alter Reduce's actual output.

    Returns a dict:
        "glosses"        -- same output translate_speech() would return
        "oov_density"     -- fraction of input words outside the
                              trained vocabulary
        "low_confidence"  -- True if this result should be treated
                              with caution
        "oov_words"       -- the specific unrecognized words, if any
    """
    _ensure_loaded()
    text = normalize_input(sentence)
    confidence = reduce_confidence(text)
    glosses = translate_speech(sentence)
    return {
        "glosses": glosses,
        "oov_density": confidence["density"],
        "low_confidence": confidence["low_confidence"],
        "oov_words": confidence["oov_words"],
    }


def known_gestures():
    return list(CONFIRMED_GESTURES)