"""
generate_context_dataset.py -- Generates multi-turn dialogue training
examples for teaching the Complete task to use conversational context,
addressing the same gap conversation_session.py's rule-based
subject-ellipsis recovery addresses, but as something the MODEL learns
from training examples rather than a fixed Python rule.

RECIPE (applied automatically to every qualifying entry in ENTRIES,
imported directly from generate_dataset.py -- no new sentences are
invented, only existing already-validated entries are restructured):

  Take any entry containing "mother" OR "father" (but not both -- see
  below) with 2 or more gloss tokens. Split it into two conceptual
  turns:
    Turn 1 (context):      the person word alone
    Turn 2 (current turn):  every other word in the entry, in their
                             original relative order
  The training PAIR is: source = [context word] + [current turn
  words], target = the entry's own already-approved sentence.

  Example: ("where", "mother") -> "Where is mother?"
    becomes  source = "mother where"  ->  target = "Where is mother?"

EXCLUDED: entries containing BOTH "mother" and "father" (e.g. "Are
mother and father home?") -- these have two subjects, not one dropped
subject, which is a different pattern from what this recipe teaches.

HELD-OUT SPLIT: of the 20 generated examples, 4 are deliberately held
out of training (2 with a "mother" context, 2 with "father"; 2 with a
one-word current turn, 2 with a two-word current turn). Every one of
these 20 combinations already exists in the base dataset in ordinary
(non-split) form, so this split does not test whether the model has
seen a combination before -- it tests something more specific: whether
the model has learned the general SKILL of using a context word to
complete an otherwise-incomplete current turn, or has only memorized
this exact set of 20 context-formatted pairs. That distinction is the
actual thing worth measuring once training is done.

SCOPE NOTE (deliberate simplification, not an oversight): no special
separator token is used between "context" and "current turn" -- they
are concatenated as one plain source sequence, identical in format to
any other multi-word Complete-task input. This avoids changing the
vocabulary size (which would force re-exporting ONNX and re-verifying
parity across the whole pipeline for an unproven benefit on this small
a dataset). A version where the model explicitly learns to tell
context apart from the current turn is a reasonable future extension
once more data exists to justify it.

Run: python generate_context_dataset.py
Produces: dataset_context_train.jsonl, dataset_context_test.jsonl
"""

import json

from generate_dataset import ENTRIES

PERSON_WORDS = {"mother", "father"}

# Deliberately held out of training -- see HELD-OUT SPLIT note above.
# Identified by (context_word, current_turn) so the choice is explicit
# and reviewable, not a random seed someone has to trust blindly.
HELD_OUT = {
    ("mother", "name"),
    ("father", "how"),
    ("mother", "where drive"),
    ("father", "laundry home"),
}


def build_context_examples(entries):
    rows = []
    skipped_compound = []

    for tokens, sentence in entries:
        person_tokens = [t for t in tokens if t in PERSON_WORDS]
        other_tokens = [t for t in tokens if t not in PERSON_WORDS]

        if len(person_tokens) != 1:
            # 0 person words: nothing to establish as context, skip.
            # 2 person words (mother AND father): compound subject,
            # a different pattern -- explicitly excluded, not silently
            # dropped -- logged below for the record.
            if len(person_tokens) == 2:
                skipped_compound.append((tokens, sentence))
            continue

        if not other_tokens:
            # Single-word entry (just "mother" or "father" alone) --
            # nothing left to be the "current turn", skip.
            continue

        context_word = person_tokens[0]
        current_turn = " ".join(other_tokens)
        source = " ".join([context_word] + other_tokens)

        rows.append({
            "task": "complete_context",
            "instruction": "Given a prior-turn subject and the current turn's glosses, "
                            "generate the complete sentence.",
            "context_turn": context_word,
            "current_turn": current_turn,
            "input": source,
            "output": sentence,
            "held_out": (context_word, current_turn) in HELD_OUT,
        })

    return rows, skipped_compound


def main():
    rows, skipped_compound = build_context_examples(ENTRIES)

    train_rows = [r for r in rows if not r["held_out"]]
    test_rows = [r for r in rows if r["held_out"]]

    with open("dataset_context_train.jsonl", "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")

    with open("dataset_context_test.jsonl", "w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Generated {len(rows)} context-recovery examples from {len(ENTRIES)} source entries.")
    print(f"  train: {len(train_rows)} -> dataset_context_train.jsonl")
    print(f"  held out (test): {len(test_rows)} -> dataset_context_test.jsonl\n")

    print("TRAIN examples:")
    for row in train_rows:
        print(f"  context={row['context_turn']!r:10s} current={row['current_turn']!r:14s} "
              f"-> {row['output']!r}")

    print("\nHELD-OUT (test) examples -- the model will never see these in context-split")
    print("form during training, only as ordinary combined entries elsewhere in the dataset:")
    for row in test_rows:
        print(f"  context={row['context_turn']!r:10s} current={row['current_turn']!r:14s} "
              f"-> {row['output']!r}")

    if skipped_compound:
        print(f"\nExcluded {len(skipped_compound)} compound-subject entries "
              f"(contain both mother and father -- different pattern, not "
              f"silently dropped):")
        for tokens, sentence in skipped_compound:
            print(f"  {tokens} -> {sentence!r}")


if __name__ == "__main__":
    main()