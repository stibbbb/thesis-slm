"""
generate_dataset_v2.py -- Phase 2 dataset generator, built in response
to adviser feedback that the model should learn CONTEXTUAL RULES
rather than memorize specific input/output pairs.

THE CORE IDEA
Instead of one sentence per meaning, each underlying pattern is
expressed in MANY different phrasings. A model trained on a single
phrasing per meaning can memorize it; a model trained on many
phrasings of the same rule has to learn the rule itself, because
memorizing every variant is harder than learning the pattern behind
them.

Example -- all of these teach the SAME Reduce rule (drop possessives,
drop copulas, keep content words):
    "my mother is driving"   -> mother driving
    "mother is driving"      -> mother driving
    "my mom is driving"      -> mother driving
    "is my mother driving"   -> mother driving

WHAT PHASE 2 ADDS OVER THE ORIGINAL 37-ENTRY DATASET
  1. Many phrasing variations per pattern (the main change)
  2. New pronoun vocabulary: his, her, me, my -- so Reduce can preserve
     WHO is being referred to. Directly addresses the adviser's
     scenario: two Deaf people and one hearing person, where asking for
     one person's name must not reduce to a bare "name", which loses
     the critical information about whose name is being asked.
  3. Explicit keep/drop training signal: filler words (my, the, is,
     are, am) are dropped; referential pronouns (his, her, me, you)
     are KEPT.

NO CONTRACTIONS OR POSSESSIVE APOSTROPHES are used in this dataset.
The tokenizer strips non-word characters, so a form like "what's"
would become an unpredictable token. Every such phrasing has a clean
equivalent already present ("what is his name"). Handling contractions
properly is deferred -- real speech-to-text WILL produce them, so this
is a known gap, not a solved problem.

Run: python generate_dataset_v2.py
Produces: dataset_v2_train.jsonl, dataset_v2_test.jsonl
"""

import json
import random

SEED = 42

# ---------------------------------------------------------------- #
# REDUCE PATTERNS -- natural speech -> gloss keywords
# Each entry: (list of input phrasings, single gloss output)
# The MANY input phrasings per single output is the whole point --
# that variety is what forces rule-learning over memorization.
# ---------------------------------------------------------------- #

REDUCE_PATTERNS = [
    # --- Subject + action: drop possessives ("my"), drop copulas ("is")
    (["my mother is driving", "mother is driving", "my mom is driving",
      "mom is driving", "is my mother driving", "my mother drives",
      "mother drives"], "mother driving"),
    (["my father is driving", "father is driving", "my dad is driving",
      "dad is driving", "is my father driving", "my father drives",
      "father drives"], "father driving"),
    (["she is driving", "is she driving", "she drives"], "she driving"),
    (["he is driving", "is he driving", "he drives"], "he driving"),
    (["i am driving", "am i driving", "i drive"], "i driving"),

    # --- Subject + laundry
    (["my mother is doing laundry", "mother is doing laundry",
      "my mom is doing the laundry", "mom is doing laundry",
      "my mother does the laundry"], "mother laundry"),
    (["my father is doing laundry", "father is doing laundry",
      "my dad is doing the laundry", "dad is doing laundry",
      "my father does the laundry"], "father laundry"),
    (["she is doing laundry", "she does the laundry"], "she laundry"),
    (["he is doing laundry", "he does the laundry"], "he laundry"),
    (["i am doing laundry", "i do the laundry"], "i laundry"),

    # --- Location questions
    (["where is my mother", "where is mother", "where is my mom",
      "where is mom", "where is she"], "where mother"),
    (["where is my father", "where is father", "where is my dad",
      "where is dad", "where is he"], "where father"),
    (["where is home", "where is the house", "where is my home",
      "where is my house"], "where home"),

    # --- Home / location statements
    (["my mother is home", "mother is home", "is my mother home",
      "is mother home", "my mom is at home", "mother is at home"],
     "mother home"),
    (["my father is home", "father is home", "is my father home",
      "is father home", "my dad is at home", "father is at home"],
     "father home"),
    (["she is home", "is she home", "she is at home"], "she home"),
    (["he is home", "is he home", "he is at home"], "he home"),
    (["i am home", "i am at home"], "i home"),

    # --- NAME questions: THE KEY ADVISER SCENARIO
    # These must NOT reduce to a bare "name" -- with two Deaf people
    # present, that loses whose name is being asked about.
    (["what is his name", "his name", "tell me his name",
      "what is his name please", "do you know his name"], "name his"),
    (["what is her name", "her name", "tell me her name",
      "what is her name please", "do you know her name"], "name her"),
    (["what is your name", "your name", "tell me your name",
      "what is your name please", "do you know your name"], "name you"),
    (["what is my name", "do you know my name", "tell me my name"],
     "name me"),
    (["what is mother name", "what is my mother name",
      "tell me mother name"], "name mother"),
    (["what is father name", "what is my father name",
      "tell me father name"], "name father"),

    # --- Wellbeing questions
    (["how are you", "how are you doing", "are you fine",
      "are you okay", "how are you today"], "how you"),
    (["how is he", "how is he doing", "is he fine", "is he okay"],
     "how he"),
    (["how is she", "how is she doing", "is she fine", "is she okay"],
     "how she"),
    (["how is my mother", "how is mother", "how is my mom",
      "is mother fine", "is mother okay"], "how mother"),
    (["how is my father", "how is father", "how is my dad",
      "is father fine", "is father okay"], "how father"),

    # --- Wellbeing answers
    (["i am fine", "i am okay", "i am good", "i am alright",
      "i am doing fine"], "i fine"),
    (["he is fine", "he is okay", "he is good", "he is alright"],
     "he fine"),
    (["she is fine", "she is okay", "she is good", "she is alright"],
     "she fine"),
    (["mother is fine", "my mother is fine", "mom is okay",
      "my mother is okay"], "mother fine"),
    (["father is fine", "my father is fine", "dad is okay",
      "my father is okay"], "father fine"),

    # --- Going home
    (["i am going home", "i am going to my house", "i will go home"],
     "i going home"),
    (["she is going home", "she is going to her house",
      "she will go home"], "she going home"),
    (["he is going home", "he is going to his house",
      "he will go home"], "he going home"),
    (["my mother is going home", "mother is going home",
      "my mom is going home"], "mother going home"),
    (["my father is going home", "father is going home",
      "my dad is going home"], "father going home"),
]

# ---------------------------------------------------------------- #
# COMPLETE PATTERNS -- glosses -> natural sentence
# Multiple valid gloss orderings per output, so Complete learns to
# handle whatever order Restructure hands it rather than one fixed
# sequence.
# ---------------------------------------------------------------- #

COMPLETE_PATTERNS = [
    (["mother driving", "driving mother"], "mother is driving"),
    (["father driving", "driving father"], "father is driving"),
    (["she driving", "driving she"], "she is driving"),
    (["he driving", "driving he"], "he is driving"),
    (["i driving", "driving i"], "i am driving"),
    (["mother laundry", "laundry mother"], "mother is doing laundry"),
    (["father laundry", "laundry father"], "father is doing laundry"),
    (["she laundry", "laundry she"], "she is doing laundry"),
    (["he laundry", "laundry he"], "he is doing laundry"),
    (["i laundry", "laundry i"], "i am doing laundry"),
    (["where mother", "mother where"], "where is mother"),
    (["where father", "father where"], "where is father"),
    (["where home", "home where"], "where is home"),
    (["mother home", "home mother"], "is mother home"),
    (["father home", "home father"], "is father home"),
    (["she home", "home she"], "is she home"),
    (["he home", "home he"], "is he home"),
    (["i home", "home i"], "i am home"),
    (["name his", "his name"], "what is his name"),
    (["name her", "her name"], "what is her name"),
    (["name you", "you name"], "what is your name"),
    (["name me", "me name"], "what is my name"),
    (["name mother", "mother name"], "what is mother name"),
    (["name father", "father name"], "what is father name"),
    (["how you", "you how"], "how are you"),
    (["how he", "he how"], "how is he"),
    (["how she", "she how"], "how is she"),
    (["how mother", "mother how"], "how is mother"),
    (["how father", "father how"], "how is father"),
    (["i fine", "fine i"], "i am fine"),
    (["he fine", "fine he"], "he is fine"),
    (["she fine", "fine she"], "she is fine"),
    (["mother fine", "fine mother"], "mother is fine"),
    (["father fine", "fine father"], "father is fine"),
    (["i going home", "going home i"], "i am going home"),
    (["she going home", "going home she"], "she is going home"),
    (["he going home", "going home he"], "he is going home"),
    (["mother going home", "going home mother"], "mother is going home"),
    (["father going home", "going home father"], "father is going home"),
]


def build_reduce_rows():
    rows = []
    for inputs, output in REDUCE_PATTERNS:
        for inp in inputs:
            rows.append({
                "task": "reduce",
                "input": inp,
                "output": output,
                "pattern": "reduce:" + output,
            })
    return rows


def build_complete_rows():
    rows = []
    for inputs, output in COMPLETE_PATTERNS:
        for inp in inputs:
            rows.append({
                "task": "complete",
                "input": inp,
                "output": output,
                "pattern": "complete:" + output,
            })
    return rows


def build_restructure_rows():
    """
    Restructure training data: scrambled gloss order -> correct order.
    Derived from the Complete patterns' gloss sequences, so the two
    tasks stay consistent with each other by construction.
    """
    rows = []
    for inputs, _ in COMPLETE_PATTERNS:
        correct = inputs[0]           # first listed = canonical order
        for variant in inputs:
            rows.append({
                "task": "restructure",
                "input": variant,
                "output": correct,
                "pattern": "restructure:" + correct,
            })
    return rows


def split_by_pattern(rows, test_fraction=0.2, seed=SEED):
    """
    Splits at the PATTERN level, not the row level. Every phrasing of
    a held-out pattern is held out together -- otherwise a phrasing in
    train and a different phrasing of the SAME rule in test would leak,
    since they teach the same thing.
    """
    patterns = sorted({r["pattern"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(patterns)
    n_test = max(1, int(len(patterns) * test_fraction))
    test_patterns = set(patterns[:n_test])

    train = [r for r in rows if r["pattern"] not in test_patterns]
    test = [r for r in rows if r["pattern"] in test_patterns]
    return train, test, test_patterns


def main():
    reduce_rows = build_reduce_rows()
    complete_rows = build_complete_rows()
    restructure_rows = build_restructure_rows()
    all_rows = reduce_rows + complete_rows + restructure_rows

    train, test, test_patterns = split_by_pattern(all_rows)

    with open("dataset_v2_train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open("dataset_v2_test.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")

    print("=== Phase 2 dataset generated ===\n")
    print(f"Reduce rows       : {len(reduce_rows)}")
    print(f"Complete rows     : {len(complete_rows)}")
    print(f"Restructure rows  : {len(restructure_rows)}")
    print(f"TOTAL             : {len(all_rows)}")
    print(f"  (previous dataset: 139 rows)\n")

    n_patterns = len({r["pattern"] for r in all_rows})
    print(f"Distinct patterns : {n_patterns}")
    print(f"Avg phrasings per pattern: {len(all_rows) / n_patterns:.1f}\n")

    print(f"TRAIN rows: {len(train)}  ->  dataset_v2_train.jsonl")
    print(f"TEST rows : {len(test)}  ->  dataset_v2_test.jsonl")
    print(f"\nHeld-out patterns ({len(test_patterns)}) -- ALL phrasings withheld together:")
    for pat in sorted(test_patterns):
        print(f"    {pat}")

    longest_in = max(len(r["input"].split()) for r in all_rows)
    longest_out = max(len(r["output"].split()) for r in all_rows)
    print(f"\nLongest input : {longest_in} words")
    print(f"Longest output: {longest_out} words")

    print("\nSample -- one pattern, many phrasings (this is the point):")
    for r in reduce_rows[:7]:
        print(f"    {r['input']!r:32s} -> {r['output']!r}")


if __name__ == "__main__":
    main()