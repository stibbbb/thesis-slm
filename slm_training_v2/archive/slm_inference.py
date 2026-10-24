"""
Unified PyTorch SLM inference module. Development/testing counterpart
to onnx_inference.py -- same hardening, same behavior, verified to
match line-for-line in every prior parity check.

Demo + edge-case suite: python slm_inference.py
"""

import torch
from scipy.optimize import linear_sum_assignment

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from encoder_distilbert import DistilBertForRestructure
from decoder_tinyllama import TinyLlamaForGeneration
from word_picker import WordPicker, picker_predict
from onnx_config import (
    GLOSS_MAP, normalize_input, sanitize_glosses, enforce_max_len,
    ENCODER_MAX_LEN, DECODER_SRC_LEN,
)

ENCODER_PATH = "trained_distilbert.pt"
DECODER_PATH = "trained_tinyllama.pt"
PICKER_PATH = "trained_picker.pt"

# Only the three deployed models. The runner-up architectures from the
# 6-way comparison (TinyBERT, MobileBERT, BabyLlama, MobiLlama) live in
# training_comparison/ and are not loaded by the pipeline.
MODEL_CLASSES = {
    "distilbert": DistilBertForRestructure,
    "tinyllama":  TinyLlamaForGeneration,
    "picker":     WordPicker,
}


def map_glosses(text):
    text = normalize_input(text)
    return " ".join(GLOSS_MAP.get(w, w) for w in text.lower().split())


def load_model(path, tokenizer, device):
    ckpt = torch.load(path, map_location=device)
    ModelClass = MODEL_CLASSES[ckpt["variant"]]
    cfg = {k: v for k, v in ckpt.items()
           if k not in ("model_state_dict", "vocab_size", "variant")}
    model = ModelClass(
        vocab_size=ckpt["vocab_size"],
        pad_id=tokenizer.pad_id,
        **cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def restructure(model, tokenizer, gloss_text, device):
    """Reorder glosses into English word order. Hungarian assignment
    guarantees every KNOWN input word gets a unique output position.
    Unknown words are dropped before reaching the model (warning
    printed) -- an untrained <unk> embedding would otherwise silently
    corrupt the position prediction rather than crash, which is worse.
    Oversized input is truncated with a warning -- previously
    unguarded here, and confirmed by tracing the embeddings code to
    crash with a tensor shape mismatch above ENCODER_MAX_LEN words."""
    raw_tokens = tokenizer._clean_and_split(gloss_text)
    input_tokens, _dropped = sanitize_glosses(raw_tokens)
    input_tokens = enforce_max_len(input_tokens, ENCODER_MAX_LEN, "Restructure input")
    n = len(input_tokens)
    if n == 0:
        return ""
    if n == 1:
        return input_tokens[0]
    ids = [tokenizer.token_to_id[w] for w in input_tokens]
    input_tensor = torch.tensor([ids]).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
    score = logits[0, :n, :n].cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-score)
    ordered = [None] * n
    for word_idx, pos in zip(row_ind, col_ind):
        ordered[pos] = input_tokens[word_idx]
    return " ".join(ordered)


def generate(model, tokenizer, text, device, max_new_tokens=16, strip_unknown=False):
    """Decoder generation. Glosses in -> sentence out (Complete);
    sentence in -> glosses out (Reduce). strip_unknown=False by
    default -- see the matching note in onnx_inference.py's
    generate_onnx()."""
    text = normalize_input(text)
    if not text:
        return ""
    words = tokenizer._clean_and_split(text)
    if strip_unknown:
        words, _dropped = sanitize_glosses(words)
    words = enforce_max_len(words, DECODER_SRC_LEN, "generate() input")
    if not words:
        return ""
    ids = [tokenizer.token_to_id.get(w, tokenizer.unk_id) for w in words]
    src = torch.tensor([ids]).to(device)
    return model.generate(src, tokenizer, max_new_tokens=max_new_tokens)


def _sentence_covers_signs(sentence, signs):
    """
    Every sign the person made must appear in the sentence, directly or
    as an inflected form of itself. The decoder GENERATES text, so
    unlike the word picker it can produce words nobody signed -- it
    once turned "you sick" into "family home". This is the check that
    catches that.
    """
    out = set(sentence.lower().split())
    # A sign is covered by more than its own spelling. The self-reference
    # sign "my" surfaces as "i am"; "you" surfaces as "your" or "are".
    # Without this, "my sick" -> "i am sick" was wrongly rejected and the
    # raw signs were shown instead of a correct sentence.
    cover_forms = {"my": {"my", "i", "me", "am"},
                   "you": {"you", "your", "are"}}
    for w in signs:
        forms = cover_forms.get(w, {w}) | {
            k for k, v in GLOSS_MAP.items() if v == w}
        if not (forms & out) and not any(o.startswith(w[:4]) for o in out):
            return False
    return True


def _fallback_sentence(signs):
    """
    Used when the decoder's answer does not contain what was signed.
    Showing the signs plainly is worse English but it is HONEST -- the
    hearing person sees what was actually signed rather than a fluent
    sentence about something else.
    """
    return " ".join(signs)


def sign_to_sentence(encoder, decoder, tokenizer, gloss_text, device):
    """Full Direction-1 pipeline: recognition output -> natural sentence."""
    mapped = map_glosses(gloss_text)
    ordered = restructure(encoder, tokenizer, mapped, device)
    if not ordered:
        return ""
    sentence = generate(decoder, tokenizer, ordered, device)

    # Safety net. The rules in generate_dataset_v4.py cover every
    # combination they judge sensible, but nobody can predict every
    # combination a real conversation will produce. When the decoder
    # meets one it was never taught, it guesses -- and a fluent guess
    # about the wrong subject is worse than plain signs.
    signs = ordered.split()
    if not _sentence_covers_signs(sentence, signs):
        print(f"[SLM WARNING] the sentence {sentence!r} does not contain "
              f"what was signed ({ordered!r}); showing the signs instead.")
        return _fallback_sentence(signs)
    return sentence


def sentence_to_glosses(model, tokenizer, sentence, device):
    """Direction-2 pipeline: spoken sentence -> gloss keywords.

    Pass the WORD PICKER here. It decides, for every word, whether to
    keep it and where it goes, so it can only ever output words the
    person actually said. On held-out sentences it scores about 97.8%
    against 88.0% for the decoder that previously did this job.

    A decoder is still accepted for backward compatibility -- older
    callers passed one -- but it is the weaker path and will generate
    words that were never spoken.
    """
    if isinstance(model, WordPicker):
        glosses, _labels, _question = picker_predict(
            model, tokenizer, sentence, device)
        return glosses
    return generate(model, tokenizer, sentence, device, strip_unknown=False)


def main():
    device = "cpu"
    print(f"CP1: device: {device}", flush=True)
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    encoder = load_model(ENCODER_PATH, tokenizer, device)
    decoder = load_model(DECODER_PATH, tokenizer, device)
    picker = load_model(PICKER_PATH, tokenizer, device)
    print(f"CP2: encoder={ENCODER_PATH}  decoder={DECODER_PATH}  "
          f"picker={PICKER_PATH}", flush=True)

    print("\n--- Direction 1: signs -> sentence ---")
    for glosses in ["drive mother where", "home drive father", "papa laundry", "fine how"]:
        sentence = sign_to_sentence(encoder, decoder, tokenizer, glosses, device)
        print(f"  {glosses!r:24s} -> {sentence!r}")

    print("\n--- Direction 2: sentence -> glosses ---")
    for sentence in ["Where is mother? Is she home?", "How are you? I am fine."]:
        glosses = sentence_to_glosses(picker, tokenizer, sentence, device)
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
            result = sign_to_sentence(encoder, decoder, tokenizer, case, device)
            print(f"  [{label}] {case!r} -> {result!r}")
        except Exception as e:
            print(f"  [{label}] {case!r} -> RAISED {type(e).__name__}: {e}")

    for bad in [None, 42, {"a": 1}]:
        try:
            sign_to_sentence(encoder, decoder, tokenizer, bad, device)
            print(f"  [invalid type] {bad!r} -> NO ERROR RAISED (unexpected!)")
        except TypeError as e:
            print(f"  [invalid type] {bad!r} -> correctly raised TypeError: {e}")

    print("\nCP3: pipeline demo + edge case suite complete", flush=True)


if __name__ == "__main__":
    main()