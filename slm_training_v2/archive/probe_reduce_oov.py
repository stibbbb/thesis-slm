"""
probe_reduce_oov.py -- Follow-up to the OOV finding surfaced by
interactive testing of conversation_demo.py ("you will go home" ->
"mother home", logged in slm_progress_notes.md known issue 14).

This is a DIAGNOSTIC / characterization tool, not a permanent pipeline
feature -- it exists to gather a few more data points on Reduce's
behavior across different out-of-vocabulary densities before Phase B,
not to fix anything. Each test sentence's OOV density is computed
automatically against the real trained vocabulary, so the density
labels are always accurate even if the vocabulary changes later.

Run: python probe_reduce_oov.py
"""

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from onnx_config import ENCODER_ONNX_PATH, DECODER_ONNX_PATH
from onnx_inference import sentence_to_glosses_onnx
import onnxruntime as ort

KNOWN = set(ALL_VOCAB)

# (sentence, informal expectation) -- expectation is NOT a strict
# pass/fail target, just what a reasonable Reduce output would contain,
# for eyeballing how far off each result actually is.
PROBES = [
    ("Where is mother? Is she home?", "control -- 0% OOV, already well established"),
    ("Good morning, is your father home?", "mostly in-vocabulary, 2 OOV words at the start"),
    ("Hello, how are you today?", "moderate OOV density, OOV words at both ends"),
    ("Can you help me?", "high OOV density, only 'you' is in-vocabulary"),
    ("I love my family very much.", "very high OOV density, only 'i' is in-vocabulary"),
]


def oov_density(sentence):
    words = [w.strip(".,?!").lower() for w in sentence.split()]
    oov = [w for w in words if w not in KNOWN]
    return len(oov), len(words), oov


def main():
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    dec_session = ort.InferenceSession(DECODER_ONNX_PATH)

    print("=== Reduce OOV density probe ===")
    print("(diagnostic only -- see slm_progress_notes.md known issue 14)\n")

    for sentence, note in PROBES:
        n_oov, n_total, oov_words = oov_density(sentence)
        pct = round(100 * n_oov / n_total)
        result = sentence_to_glosses_onnx(dec_session, tokenizer, sentence)

        print(f"Input:   {sentence!r}")
        print(f"OOV:     {n_oov}/{n_total} words ({pct}%) -- {oov_words if oov_words else 'none'}")
        print(f"Note:    {note}")
        print(f"Output:  {result!r}")
        print()


if __name__ == "__main__":
    main()