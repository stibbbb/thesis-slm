"""
evaluate_context_learning.py -- Tests whether TinyLlama's new
"complete_context" training rows actually get exercised by the real
deployed pipeline, or whether Restructure's reordering already
intercepts the context+current-turn word order before Complete ever
sees it in the format it was newly trained on.

Uses the PyTorch checkpoints directly (slm_inference.py), not ONNX --
this is a diagnostic check before deciding whether ONNX re-export is
even worth doing yet.

Run: python evaluate_context_learning.py
"""

import json

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from slm_inference import load_model, restructure, generate, ENCODER_PATH, DECODER_PATH


def load_held_out():
    rows = []
    with open("dataset_context_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    device = "cpu"
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    encoder = load_model(ENCODER_PATH, tokenizer, device)
    decoder = load_model(DECODER_PATH, tokenizer, device)

    held_out = load_held_out()

    print("=== Held-out context-recovery cases: three views per case ===\n")
    for row in held_out:
        raw_source = row["input"]  # e.g. "mother where" -- context word first, unordered
        expected = row["output"]

        print(f"Case: context={row['context_turn']!r}  current_turn={row['current_turn']!r}")
        print(f"  expected:                          {expected!r}")

        # View 1: what Restructure alone does with the raw context+current string
        restructured = restructure(encoder, tokenizer, raw_source, device)
        print(f"  [1] Restructure({raw_source!r}) -> {restructured!r}")

        # View 2: the REAL deployed pipeline -- Restructure's output feeds Complete
        full_pipeline_result = generate(decoder, tokenizer, restructured, device)
        print(f"  [2] Full pipeline (Restructure -> Complete)  -> {full_pipeline_result!r}")

        # View 3: Complete called DIRECTLY on the raw, unreordered string --
        # exactly the format the new context-training rows taught it to handle.
        direct_result = generate(decoder, tokenizer, raw_source, device)
        print(f"  [3] Complete direct, bypassing Restructure   -> {direct_result!r}")
        print()


if __name__ == "__main__":
    main()