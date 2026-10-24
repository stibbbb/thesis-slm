"""
ONNX Runtime version of the SLM pipeline -- mirrors slm_inference.py
exactly, but runs on onnxruntime.InferenceSession instead of PyTorch.
This is the version that ships to the Raspberry Pi 5.

Run export_onnx.py first to produce the .onnx files this script loads.

Fixed-shape note: both ONNX graphs were exported with fixed input
shapes (batch=1, fixed padded length). All inputs here are padded to
exactly those lengths -- see export_onnx.py for why.

Input hardening (edge cases handled explicitly):
  - Accepts a string OR a list of words (normalize_input)
  - Rejects None / wrong types with a clear error, not a cryptic crash
  - Unknown gloss words (Restructure only) are dropped with a warning
    instead of silently corrupted through the untrained <unk> embedding
  - Oversized input is truncated with a warning instead of crashing
  - Empty input returns "" immediately

Demo + edge-case suite: python onnx_inference.py
"""

import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from onnx_config import (
    ENCODER_ONNX_PATH, DECODER_ONNX_PATH,
    ENCODER_MAX_LEN, DECODER_SRC_LEN, DECODER_TGT_LEN,
    GLOSS_MAP, pad_ids, normalize_input, sanitize_glosses, enforce_max_len,
)


def map_glosses(text):
    """Translate recognition-module glosses into SLM vocabulary.
    Accepts a string or a list of words."""
    text = normalize_input(text)
    return " ".join(GLOSS_MAP.get(w, w) for w in text.lower().split())


def restructure_onnx(session, tokenizer, gloss_text):
    """Encoder step: Hungarian assignment on ONNX logits. Every KNOWN
    input word gets a unique output position. Unknown words are
    dropped BEFORE reaching the model (warning printed)."""
    raw_tokens = tokenizer._clean_and_split(gloss_text)
    input_tokens, _dropped = sanitize_glosses(raw_tokens)
    input_tokens = enforce_max_len(input_tokens, ENCODER_MAX_LEN, "Restructure input")
    n = len(input_tokens)
    if n == 0:
        return ""
    if n == 1:
        return input_tokens[0]

    ids = [tokenizer.token_to_id[w] for w in input_tokens]
    padded = pad_ids(ids, ENCODER_MAX_LEN, tokenizer.pad_id)
    logits = session.run(
        None, {"input_ids": np.array([padded], dtype=np.int64)}
    )[0]
    score = logits[0, :n, :n]
    row_ind, col_ind = linear_sum_assignment(-score)
    ordered = [None] * n
    for word_idx, pos in zip(row_ind, col_ind):
        ordered[pos] = input_tokens[word_idx]
    return " ".join(ordered)


def generate_onnx(session, tokenizer, text, max_new_tokens=16, strip_unknown=False):
    """Decoder step: greedy generation using a FIXED-length buffer.

    strip_unknown=False by default: Reduce's input is a natural,
    unconstrained sentence from a hearing person and is EXPECTED to
    contain words outside the 23-word vocabulary -- silently deleting
    them would alter what the person said without their knowledge.
    True out-of-vocabulary robustness for natural speech is a Phase B
    item, not something safely papered over at the interface layer."""
    text = normalize_input(text)
    if not text:
        return ""
    words = tokenizer._clean_and_split(text)
    if strip_unknown:
        words, _dropped = sanitize_glosses(words)
    words = enforce_max_len(words, DECODER_SRC_LEN, "generate() input")
    if not words:
        return ""

    src_ids = [tokenizer.token_to_id.get(w, tokenizer.unk_id) for w in words]
    src_padded = pad_ids(src_ids, DECODER_SRC_LEN, tokenizer.pad_id)
    src_arr = np.array([src_padded], dtype=np.int64)

    buffer = [tokenizer.bos_id] + [tokenizer.pad_id] * (DECODER_TGT_LEN - 1)
    current_length = 1

    generated_ids = []
    for _ in range(max_new_tokens):
        if current_length >= DECODER_TGT_LEN:
            break
        tgt_arr = np.array([buffer], dtype=np.int64)
        logits = session.run(
            None, {"input_ids": src_arr, "target_ids": tgt_arr}
        )[0]
        next_token = int(logits[0, current_length - 1, :].argmax())
        if next_token == tokenizer.eos_id:
            break
        buffer[current_length] = next_token
        generated_ids.append(next_token)
        current_length += 1

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def sign_to_sentence_onnx(enc_session, dec_session, tokenizer, gloss_text):
    mapped = map_glosses(gloss_text)
    ordered = restructure_onnx(enc_session, tokenizer, mapped)
    if not ordered:
        return ""
    return generate_onnx(dec_session, tokenizer, ordered)


def sentence_to_glosses_onnx(dec_session, tokenizer, sentence):
    return generate_onnx(dec_session, tokenizer, sentence, strip_unknown=False)


def main():
    print("CP1: loading ONNX Runtime sessions", flush=True)
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    enc_session = ort.InferenceSession(ENCODER_ONNX_PATH)
    dec_session = ort.InferenceSession(DECODER_ONNX_PATH)
    print(f"CP2: encoder={ENCODER_ONNX_PATH}  decoder={DECODER_ONNX_PATH}", flush=True)

    print("\n--- Direction 1: signs -> sentence (ONNX) ---")
    for glosses in ["drive mother where", "home drive father", "papa laundry", "fine how"]:
        sentence = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, glosses)
        print(f"  {glosses!r:24s} -> {sentence!r}")

    print("\n--- Direction 2: sentence -> glosses (ONNX) ---")
    for sentence in ["Where is mother? Is she home?", "How are you? I am fine."]:
        glosses = sentence_to_glosses_onnx(dec_session, tokenizer, sentence)
        print(f"  {sentence!r:35s} -> {glosses!r}")

    print("\n--- Edge cases ---")
    edge_cases = [
        ("empty string", ""),
        ("whitespace only", "   "),
        ("single known word", "laundry"),
        ("unknown word mixed in", "mother xyz drive"),
        ("list instead of string", ["mother", "drive", "where"]),
        ("too many words", ("mother father drive home laundry where how fine name "
                             "mother father drive home laundry where how fine name")),
    ]
    for label, case in edge_cases:
        try:
            result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, case)
            print(f"  [{label}] {case!r} -> {result!r}")
        except Exception as e:
            print(f"  [{label}] {case!r} -> RAISED {type(e).__name__}: {e}")

    for bad in [None, 42, {"a": 1}]:
        try:
            sign_to_sentence_onnx(enc_session, dec_session, tokenizer, bad)
            print(f"  [invalid type] {bad!r} -> NO ERROR RAISED (unexpected!)")
        except TypeError as e:
            print(f"  [invalid type] {bad!r} -> correctly raised TypeError: {e}")

    print("\nCP3: ONNX pipeline demo + edge case suite complete", flush=True)


if __name__ == "__main__":
    main()