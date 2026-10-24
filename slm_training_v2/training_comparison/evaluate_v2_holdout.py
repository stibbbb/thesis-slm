"""
evaluate_v2_holdout.py -- Evaluates the Phase 2 models on the held-out
test set (dataset_v2_test.jsonl), which contains ENTIRE PATTERNS that
were withheld from training -- every phrasing of a held-out rule was
withheld together, so nothing in this set was seen in any form.

This is the honest test of whether the larger (hidden_size=128) models
actually LEARNED THE RULES or merely MEMORIZED the training rows. The
training loss reaching ~0.0000 makes this test essential rather than
optional: perfect training loss on a small dataset is an overfitting
signal, and only held-out performance can distinguish the two.

Reports per-task accuracy (restructure / complete / reduce) since the
three tasks can fail independently.

Run: python evaluate_v2_holdout.py
"""

import json
import re
from collections import defaultdict

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from slm_inference import load_model, restructure, generate, ENCODER_PATH, DECODER_PATH


def normalize(text):
    """Strips punctuation ANYWHERE (not just trailing) and collapses
    whitespace. The tokenizer produces no punctuation, so comparing
    against punctuated references requires this to avoid false misses."""
    text = text.strip().lower()
    text = re.sub(r"[.?!,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_test_rows():
    rows = []
    with open("dataset_v3_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    device = "cpu"
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    encoder = load_model(ENCODER_PATH, tokenizer, device)
    decoder = load_model(DECODER_PATH, tokenizer, device)

    rows = load_test_rows()
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    print("=== Phase 2 held-out evaluation ===")
    print("(entire patterns withheld -- every phrasing withheld together)\n")

    totals = {}
    for task in ("restructure", "complete", "reduce"):
        task_rows = by_task.get(task, [])
        if not task_rows:
            continue

        print(f"--- {task.upper()}  ({len(task_rows)} held-out rows) ---")
        correct = 0
        misses = []
        for r in task_rows:
            if task == "restructure":
                result = restructure(encoder, tokenizer, r["input"], device)
            else:
                result = generate(decoder, tokenizer, r["input"], device)

            ok = normalize(result) == normalize(r["output"])
            correct += ok
            if not ok:
                misses.append((r["input"], r["output"], result))

        pct = 100.0 * correct / len(task_rows)
        totals[task] = (correct, len(task_rows))
        print(f"  {correct}/{len(task_rows)} correct  ({pct:.1f}%)")

        if misses:
            print(f"  Misses ({len(misses)}):")
            for inp, expected, got in misses[:10]:
                print(f"    input   : {inp!r}")
                print(f"    expected: {expected!r}")
                print(f"    got     : {got!r}")
            if len(misses) > 10:
                print(f"    ... and {len(misses) - 10} more")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_correct = sum(c for c, _ in totals.values())
    total_n = sum(n for _, n in totals.values())
    for task, (c, n) in totals.items():
        print(f"  {task:12s}: {c}/{n}  ({100.0*c/n:.1f}%)")
    print(f"  {'OVERALL':12s}: {total_correct}/{total_n}  "
          f"({100.0*total_correct/total_n:.1f}%)")
    print()
    print("Interpretation: high held-out accuracy means the models learned")
    print("the underlying RULES and generalize to phrasings never seen.")
    print("Low accuracy despite near-zero training loss would indicate")
    print("memorization -- the models are too large for this much data.")


if __name__ == "__main__":
    main()