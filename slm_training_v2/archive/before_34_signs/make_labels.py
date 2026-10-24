"""
make_labels.py -- converts the Reduce examples into PER-WORD labels
plus a sentence-level QUESTION flag.

WHY THIS EXISTS
The adviser's requirement is that the model decide, word by word,
which words to erase and where the survivors go -- rather than mapping
a whole sentence to a whole sentence. This script produces the labels
that approach needs.

TWO THINGS ARE LABELLED PER SENTENCE

1. A decision for EVERY WORD:
       0            -> DROP this word
       1, 2, 3, ...  -> KEEP it, place it at this position

2. A single QUESTION flag for the whole sentence:
       0 -> normal
       1 -> this is a wellbeing QUESTION; put "how" at the front

WHY THE FLAG IS NEEDED
Some correct answers contain a word that appears nowhere in the input:
       "is he fine"   -> "how he"        ("how" is not in the input)
       "he is fine"   -> "he fine"       (no "how" needed)
Same words, different order, opposite answers. Word-picking alone
cannot produce "how" because there is nothing to pick. The flag lets
the model say "this is a question" and supply that one marker word,
while everything else is still chosen from the words actually spoken.

Note the two sentences above ALSO differ in their per-word labels
("fine" is kept in one, dropped in the other) -- so the model has two
consistent signals to learn from, not one.

WHAT STILL CANNOT BE LABELLED
A few rows need a word that is neither in the input nor the question
marker:
       "where is he"  -> "where father"   (needs "father")
That is pronoun-to-person resolution -- a conversation-context job,
already handled by conversation_session.py, not a word-picking job.
Those rows are reported and skipped.

Run: python make_labels.py
Reads:  dataset_v3_train.jsonl, dataset_v3_test.jsonl
Writes: labels_train.jsonl, labels_test.jsonl
"""

import json

# Words that mean the same thing as a trained word. Mirrors
# onnx_config.GLOSS_MAP, plus the possessive/tense forms Reduce needs
# when matching an input word to its output form.
# Imported, not redefined. onnx_config.py is the single source of truth
# for normalisation, shared by training and by both inference paths.
from onnx_config import GLOSS_MAP as NORMALIZE

# The single marker word the question flag can supply.
QUESTION_MARKER = "how"


def normalize_word(w):
    return NORMALIZE.get(w, w)


def derive_labels(input_text, output_words):
    """
    Returns (labels, unmatched_output_words).

    labels[i] is the decision for input word i:
        0 = drop, k>0 = keep and place at output position k (1-based).
    """
    in_words = input_text.split()

    labels = [0] * len(in_words)
    used_input = [False] * len(in_words)
    unmatched = []

    for out_pos, ow in enumerate(output_words, start=1):
        # Gather EVERY input word that could produce this output word,
        # by exact match or by normalization.
        candidates = [
            i for i, iw in enumerate(in_words)
            if not used_input[i] and (iw == ow or normalize_word(iw) == ow)
        ]
        if not candidates:
            unmatched.append(ow)
            continue

        # Prefer the LAST candidate. This matters when a sentence
        # contains both a plain and a possessive form that normalize to
        # the same word -- "tell me my name" has both "me" and "my",
        # and "can you tell me your name" has both "you" and "your".
        # The possessive is the one carrying the meaning, and it comes
        # later. Taking the first candidate instead would label "my" as
        # DROP in one phrasing and KEEP in another, teaching the model
        # contradictory rules about the same word.
        found = candidates[-1]
        used_input[found] = True
        labels[found] = out_pos

    return labels, unmatched


def label_row(input_text, output_text):
    """
    Returns (labels, question_flag), or None if the row cannot be
    expressed as word-picking plus an optional question marker.
    """
    out_words = output_text.split()

    # Attempt 1: pure word-picking, no marker needed.
    labels, unmatched = derive_labels(input_text, out_words)
    if not unmatched:
        return labels, 0

    # Attempt 2: the ONLY missing word is the leading question marker.
    if out_words and out_words[0] == QUESTION_MARKER \
            and set(unmatched) == {QUESTION_MARKER}:
        labels, unmatched = derive_labels(input_text, out_words[1:])
        if not unmatched:
            return labels, 1

    return None


def rebuild(input_text, labels, question):
    """Reconstruct the output from the labels -- used by the check below."""
    in_words = input_text.split()
    n = min(len(in_words), len(labels))
    picked = [(labels[i], normalize_word(in_words[i]))
              for i in range(n) if labels[i] > 0]
    picked.sort(key=lambda pw: pw[0])
    words = [w for _, w in picked]
    if question:
        words = [QUESTION_MARKER] + words
    return " ".join(words)


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build(rows):
    kept, skipped = [], []
    for r in rows:
        if r["task"] != "reduce":
            continue
        result = label_row(r["input"], r["output"])
        if result is None:
            skipped.append((r["input"], r["output"]))
            continue
        labels, question = result
        kept.append({
            "input": r["input"],
            "output": r["output"],
            "labels": labels,
            "question": question,
            "pattern": r["pattern"],
        })
    return kept, skipped


def verify(rows):
    """
    Rebuild each output from its labels and compare to the real answer.
    If the labels are wrong, this catches it BEFORE any training time
    is spent.
    """
    bad = []
    for r in rows:
        got = rebuild(r["input"], r["labels"], r["question"])
        if got != r["output"]:
            bad.append((r["input"], r["output"], got))
    return bad


def main():
    for split in ("train", "test"):
        rows = load_rows(f"dataset_v4_{split}.jsonl")
        kept, skipped = build(rows)
        bad = verify(kept)
        n_q = sum(r["question"] for r in kept)

        out_path = f"labels_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")

        print(f"=== {split.upper()} ===")
        print(f"  rows written            : {len(kept)}  -> {out_path}")
        print(f"    of which QUESTION rows : {n_q}  (model supplies 'how')")
        print(f"  skipped (need a person word, not just a marker): {len(skipped)}")
        for inp, outp in skipped[:6]:
            print(f"      {inp!r:30s} -> {outp!r}")
        if len(skipped) > 6:
            print(f"      ... and {len(skipped)-6} more")
        print(f"  reconstruction check FAILURES: {len(bad)}")
        for inp, expected, got in bad[:6]:
            print(f"      {inp!r} expected {expected!r} rebuilt {got!r}")
        print()

    print("Reconstruction check = rebuild each output from its labels and")
    print("compare to the real answer. 0 failures means the labels encode")
    print("the data faithfully, before any training happens.")


if __name__ == "__main__":
    main()