"""
Dataset generator for the SLM (restructure / complete / reduce tasks).

Uses only the 18-word vocabulary defined in vocab_pool.py:
  9 confirmed gesture words (the only words YOLO can recognize)
  9 helper words (grammar connectors the SLM adds, never signed)

Each entry below is a hand-curated, grammatically valid combination.
Each entry produces THREE dataset rows:
  1. restructure : unordered gesture tokens -> grammatically ordered tokens
  2. complete    : ordered gesture tokens -> full natural sentence
  3. reduce      : full natural sentence -> gesture-vocabulary keywords only

HELD-OUT SPLIT:
ENTRIES are grouped by gloss-combination length (1/2/3 words) and a
fraction of each length-group is held out as TEST, entirely separate
from TRAIN, at the ENTRY level -- not the row level. This guarantees no
gloss combination appears in both files: if a combination's
restructure/complete/reduce rows are used for training, none of that
combination's rows appear in the held-out test set, and vice versa.
A naive row-level split would leak, since e.g. "drive mother" ->
"mother drive" (train) and "mother drive" -> "drive mother" (test) are
the same underlying knowledge in reverse.

Produces dataset_train.jsonl and dataset_test.jsonl. No separate
validation split at this stage -- with only ~37 curated entries,
splitting three ways leaves too little data per split, and no further
architecture tuning is happening on this synthetic dataset. Phase B,
on the real WLASL-grounded gloss set, will use a proper 3-way split.

KNOWN LIMITATION: some vocabulary words (e.g. "your", "going") appear
in only ONE entry each. If that entry lands in the test split, the
model will never have seen that word during training and will likely
miss that one case -- this is a dataset-sparsity artifact, not an
architecture failure, and goes away once Phase B provides more data.
"""

import json
import random

random.seed(42)

# -----------------------------------------------------------------
# CURATED ENTRIES
# Format: (ordered_gesture_tokens, full_sentence)
# -----------------------------------------------------------------
ENTRIES = [
    # --- single word ---
    (["how"],            "How are you?"),
    (["fine"],            "I am fine."),
    (["home"],            "I am going home."),
    (["where"],           "Where are you?"),
    (["drive"],           "I am driving."),
    (["mother"],            "Mother."),
    (["father"],            "Father."),
    (["laundry"],         "I am doing laundry."),
    (["name"],            "What is your name?"),

    # --- two word ---
    (["how", "fine"],         "How are you? I am fine."),
    (["where", "home"],       "Where is home?"),
    (["where", "mother"],       "Where is mother?"),
    (["where", "father"],       "Where is father?"),
    (["mother", "home"],        "Is mother home?"),
    (["father", "home"],        "Is father home?"),
    (["mother", "drive"],       "Mother is driving."),
    (["father", "drive"],       "Father is driving."),
    (["mother", "laundry"],     "Mother is doing laundry."),
    (["father", "laundry"],     "Father is doing laundry."),
    (["fine", "home"],        "I am fine at home."),
    (["drive", "home"],       "I am driving home."),
    (["how", "mother"],         "How is mother?"),
    (["how", "father"],         "How is father?"),
    (["mother", "name"],        "What is mother name?"),
    (["father", "name"],        "What is father name?"),
    (["where", "drive"],      "Where are you driving?"),
    (["where", "laundry"],    "Where are you doing laundry?"),

    # --- three word ---
    (["where", "mother", "home"],   "Where is mother? Is she home?"),
    (["where", "father", "home"],   "Where is father? Is he home?"),
    (["where", "father", "drive"],  "Where is father? Is he driving?"),
    (["where", "mother", "drive"],  "Where is mother? Is she driving?"),
    (["mother", "father", "home"],    "Are mother and father home?"),
    (["how", "mother", "fine"],     "How is mother? Is she fine?"),
    (["how", "father", "fine"],     "How is father? Is he fine?"),
    (["mother", "father", "drive"],   "Are mother and father driving?"),
    (["mother", "laundry", "home"], "mother is doing laundry at home."),
    (["father", "laundry", "home"], "father is doing laundry at home."),
]

GESTURE_WORDS = {
    "how", "fine", "home", "where", "drive",
    "mother", "father", "laundry", "name"
}


def shuffle_tokens(tokens, n_variants=2):
    """Generate scrambled (unordered) versions of the correct token order,
    simulating how a deaf signer naturally produces non-structured signs."""
    variants = []
    seen = {tuple(tokens)}
    attempts = 0
    while len(variants) < n_variants and attempts < 20:
        shuffled = tokens.copy()
        random.shuffle(shuffled)
        if tuple(shuffled) not in seen:
            variants.append(shuffled)
            seen.add(tuple(shuffled))
        attempts += 1
    return variants


def extract_keywords(sentence, vocab):
    """Pull out only the gesture-vocabulary words present in a sentence,
    in the order they appear, for the reduce task."""
    words = sentence.lower().replace("?", "").replace(".", "").replace("'s", "").split()
    return [w for w in words if w in vocab]


def split_entries(entries, test_ratio=0.25, seed=123):
    """
    Split ENTRIES into train/test at the ENTRY level, stratified by
    gloss-combination length (1/2/3 words), so every length category
    is represented in both splits. Uses its own Random instance so it
    never disturbs the global `random` state that shuffle_tokens()
    relies on for scramble generation.
    """
    splitter = random.Random(seed)
    by_length = {}
    for entry in entries:
        by_length.setdefault(len(entry[0]), []).append(entry)

    train, test = [], []
    for length, group in sorted(by_length.items()):
        group = group.copy()
        splitter.shuffle(group)
        n_test = max(1, round(len(group) * test_ratio))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


def build_dataset(entries):
    """Build restructure/complete/reduce rows for a given list of
    entries. Parameterized so it can be called separately for the
    train and test entry sets, keeping combinations fully isolated."""
    rows = []

    for tokens, sentence in entries:
        ordered_str = " ".join(tokens)

        if len(tokens) >= 2:
            scrambled_variants = shuffle_tokens(tokens, n_variants=2)
            for scrambled in scrambled_variants:
                rows.append({
                    "task": "restructure",
                    "instruction": "Restructure these sign tokens into the correct grammatical order.",
                    "input": " ".join(scrambled),
                    "output": ordered_str,
                })
            rows.append({
                "task": "restructure",
                "instruction": "Restructure these sign tokens into the correct grammatical order.",
                "input": ordered_str,
                "output": ordered_str,
            })

        rows.append({
            "task": "complete",
            "instruction": "Convert these ordered sign tokens into a complete natural sentence.",
            "input": ordered_str,
            "output": sentence,
        })

        keywords = extract_keywords(sentence, GESTURE_WORDS)
        if keywords:
            rows.append({
                "task": "reduce",
                "instruction": "Reduce this sentence to simple sign-vocabulary keywords.",
                "input": sentence,
                "output": " ".join(keywords),
            })

    random.shuffle(rows)
    return rows


def summarize(name, entries, rows):
    by_task = {}
    for row in rows:
        by_task[row["task"]] = by_task.get(row["task"], 0) + 1
    by_length = {}
    for tokens, _ in entries:
        by_length[len(tokens)] = by_length.get(len(tokens), 0) + 1
    print(f"\n{name}: {len(entries)} entries -> {len(rows)} rows")
    print(f"  entries by length: {dict(sorted(by_length.items()))}")
    print(f"  rows by task:      {by_task}")


if __name__ == "__main__":
    train_entries, test_entries = split_entries(ENTRIES, test_ratio=0.25)

    train_rows = build_dataset(train_entries)
    test_rows = build_dataset(test_entries)

    with open("dataset_train.jsonl", "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")

    # Deployment dataset: the exact union of train + test, nothing
    # re-randomized. Used ONLY for the final deployment retrain, once
    # no further evaluation decisions remain on this synthetic data.
    # NEVER evaluate a model trained on this file and call the result
    # "held-out" -- it has seen everything. The §7b held-out proof
    # stays tied to the train-only checkpoint, permanently.
    full_rows = train_rows + test_rows
    random.shuffle(full_rows)
    with open("dataset_full.jsonl", "w", encoding="utf-8") as f:
        for row in full_rows:
            f.write(json.dumps(row) + "\n")

    summarize("TRAIN", train_entries, train_rows)
    summarize("TEST (held out)", test_entries, test_rows)
    summarize("FULL (train+test union -- deployment retrain only)", ENTRIES, full_rows)

    print("\nTEST entries (never seen in training, for the record):")
    for tokens, sentence in test_entries:
        print(f"  {tokens} -> {sentence!r}")

    print("\nSaved: dataset_train.jsonl, dataset_test.jsonl, dataset_full.jsonl")
    print("dataset.jsonl (the old combined file) is no longer produced.")
    print("Old file left on disk untouched, but is now superseded.")