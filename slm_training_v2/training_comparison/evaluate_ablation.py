"""
evaluate_ablation.py -- Closes the attribution gap flagged in
slm_progress_notes.md \u00a712: a second TinyLlama checkpoint, identical
in every way except that it excludes the 16 context-training rows,
is compared against the currently deployed checkpoint on the same 4
held-out context-recovery cases.

Requires trained_tinyllama_ablation.pt to already exist (produced by
train_tinyllama_ablation.py). Does NOT touch, load for writing, or
overwrite the currently deployed trained_tinyllama.pt or its ONNX
export -- this is a read-only comparison of two files.

Run: python evaluate_ablation.py
"""

import json
import re

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from slm_inference import load_model, restructure, generate, ENCODER_PATH


def load_held_out():
    rows = []
    with open("dataset_context_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def normalize(text):
    text = text.strip().lower()
    text = re.sub(r"[.?!,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_checkpoint(decoder_path, encoder, tokenizer, held_out, device="cpu"):
    decoder = load_model(decoder_path, tokenizer, device)
    results = []
    for row in held_out:
        restructured = restructure(encoder, tokenizer, row["input"], device)
        result = generate(decoder, tokenizer, restructured, device)
        ok = normalize(result) == normalize(row["output"])
        results.append((row, result, ok))
    return results


def print_results(label, results, total):
    correct = 0
    for row, result, ok in results:
        status = "OK" if ok else "MISS"
        print(f"  [{status}] context={row['context_turn']!r} "
              f"current={row['current_turn']!r} -> {result!r}")
        correct += ok
    print(f"  {label}: {correct}/{total}\n")
    return correct


def main():
    device = "cpu"
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    encoder = load_model(ENCODER_PATH, tokenizer, device)
    held_out = load_held_out()
    total = len(held_out)

    print("=== Ablation comparison: WITH vs WITHOUT context-training rows ===\n")

    print("--- WITH context rows (currently deployed: trained_tinyllama.pt) ---")
    with_results = evaluate_checkpoint("trained_tinyllama.pt", encoder, tokenizer, held_out, device)
    with_correct = print_results("WITH context rows", with_results, total)

    print("--- WITHOUT context rows (ablation: trained_tinyllama_ablation.pt) ---")
    without_results = evaluate_checkpoint("trained_tinyllama_ablation.pt", encoder, tokenizer, held_out, device)
    without_correct = print_results("WITHOUT context rows", without_results, total)

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if without_correct < with_correct:
        print(f"The context-training rows made a measurable difference: "
              f"{with_correct}/{total} with, vs {without_correct}/{total} without.")
        print("Controlled evidence that the new training rows contributed to the")
        print("held-out result -- not just base dataset coverage.")
    elif without_correct == with_correct == total:
        print(f"Both checkpoints scored {with_correct}/{total} -- base dataset coverage")
        print("alone was already sufficient for these 4 specific cases. The context")
        print("rows did not demonstrably help on THIS held-out set, though they may")
        print("still help on cases outside it. The attribution gap is now answered")
        print("honestly, with evidence, rather than left as an open question.")
    else:
        print(f"WITH={with_correct}/{total}  WITHOUT={without_correct}/{total}")
        print("See individual results above for details.")


if __name__ == "__main__":
    main()