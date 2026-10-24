"""
generate_dataset_v3.py -- Phase 3 dataset, built to directly satisfy
the thesis adviser's three requirements and to close the specific
coverage gaps the v2 held-out evaluation exposed.

ADVISER REQUIREMENT 1 -- learn RULES, not fixed pairs
  Every pattern appears in many phrasings. A model given one phrasing
  per meaning can memorize it; given many, it must learn the rule.

ADVISER REQUIREMENT 2 -- Reduce must preserve WHO is referred to
  "What is his name?" must reduce to "name his", NOT a bare "name".
  With two Deaf people present, a bare "name" loses whose name is
  being asked. Heavily covered below, including the specific case
  that FAILED in v2: distinguishing "my name" (-> name me) from
  "my mother name" (-> name mother).

ADVISER REQUIREMENT 3 -- clear keep/drop signal
  DROP: my, the, is, are, am, do, does, please, will, to
  KEEP: his, her, you, me + all content words
  Taught by example across many phrasings, not by a runtime rule.

KEY DESIGN CHANGE FROM v2 -- Complete/Restructure inputs are now
restricted to the 9 CONFIRMED_GESTURES only (how, fine, home, where,
drive, mother, father, laundry, name). The recognition module
physically cannot output "she", "he", "i" or "you" -- those signs are
not in its trained set -- so v2 was spending scarce model capacity
learning to handle input that can never occur. Pronouns now appear
only in Complete's OUTPUT, inferred from context:
    "drive"         -> "i am driving"        (no person gesture ->
                                               subject is the signer)
    "mother drive"  -> "mother is driving"   (person signed explicitly)
This is a real, learnable rule AND it shrinks the input space to
something that can be covered systematically.

Reduce is unaffected by that change -- it receives real human speech,
which naturally contains pronouns, and must output them for display.

PHASE 3 BUGFIX -- "your mother"/"your father" phrasings added
throughout. A gap was found where "my mother"/"my father" phrasings
existed everywhere, but the equivalent "your mother"/"your father"
was missing everywhere -- e.g. "what is your father name" had no
training signal at all and incorrectly reduced to "name you" (the
"your...name" pattern won out over the unlearned "father" reference).
Fixed systematically, not just for the one reported case.

Run: python generate_dataset_v3.py
Produces: dataset_v3_train.jsonl, dataset_v3_test.jsonl
"""

import json
import random

SEED = 42

# ---------------------------------------------------------------- #
# COMPLETE -- gloss sequence (9 confirmed gestures only) -> sentence
# Systematically enumerated so every gesture appears in every slot it
# can legitimately occupy. v2's failures clustered exactly where a
# word never appeared in a given slot during training.
# ---------------------------------------------------------------- #

COMPLETE_PATTERNS = [
    # --- single gesture: subject is implied
    (["how"], "how are you"),
    (["fine"], "i am fine"),
    (["home"], "i am home"),
    (["drive"], "i am driving"),
    (["laundry"], "i am doing laundry"),
    (["where"], "where are you"),
    (["name"], "what is your name"),
    (["mother"], "this is mother"),
    (["father"], "this is father"),

    # --- person + action
    (["mother drive", "drive mother"], "mother is driving"),
    (["father drive", "drive father"], "father is driving"),
    (["mother laundry", "laundry mother"], "mother is doing laundry"),
    (["father laundry", "laundry father"], "father is doing laundry"),

    # --- person + state/location
    (["mother home", "home mother"], "is mother home"),
    (["father home", "home father"], "is father home"),
    (["mother fine", "fine mother"], "mother is fine"),
    (["father fine", "fine father"], "father is fine"),

    # --- where + X  (every X that can follow "where")
    (["where mother", "mother where"], "where is mother"),
    (["where father", "father where"], "where is father"),
    (["where home", "home where"], "where is home"),
    (["where drive", "drive where"], "where are you driving"),
    (["where laundry", "laundry where"], "where are you doing laundry"),

    # --- how + X
    (["how mother", "mother how"], "how is mother"),
    (["how father", "father how"], "how is father"),
    (["how fine", "fine how"], "how are you i am fine"),

    # --- name + X
    (["name mother", "mother name"], "what is mother name"),
    (["name father", "father name"], "what is father name"),

    # --- implied-subject combinations (no person gesture present)
    (["drive home", "home drive"], "i am driving home"),
    (["laundry home", "home laundry"], "i am doing laundry at home"),
    (["fine home", "home fine"], "i am fine at home"),

    # --- three-gesture combinations
    (["where mother home", "mother where home", "home where mother"],
     "where is mother is she home"),
    (["where father home", "father where home", "home where father"],
     "where is father is he home"),
    (["where mother drive", "mother where drive", "drive where mother"],
     "where is mother is she driving"),
    (["where father drive", "father where drive", "drive where father"],
     "where is father is he driving"),
    (["mother laundry home", "laundry mother home", "home mother laundry"],
     "mother is doing laundry at home"),
    (["father laundry home", "laundry father home", "home father laundry"],
     "father is doing laundry at home"),
    (["how mother fine", "mother how fine", "fine how mother"],
     "how is mother is she fine"),
    (["how father fine", "father how fine", "fine how father"],
     "how is father is he fine"),
    (["mother drive home", "drive mother home", "home mother drive"],
     "mother is driving home"),
    (["father drive home", "drive father home", "home father drive"],
     "father is driving home"),

    # --- PHASE 4: new gestures (sad, study, teacher, why, what)
    (["sad"], "i am sad"),
    (["study"], "i am studying"),
    (["teacher"], "this is teacher"),
    # "why" and "what" stand alone as short responses -- e.g. the
    # hearing person says "your father is sad" and the Deaf person
    # simply signs "why".
    (["why"], "why"),
    (["what"], "what"),

    # teacher behaves like mother/father (a person gesture)
    (["teacher drive", "drive teacher"], "teacher is driving"),
    (["teacher laundry", "laundry teacher"], "teacher is doing laundry"),
    (["teacher home", "home teacher"], "is teacher home"),
    (["teacher fine", "fine teacher"], "teacher is fine"),
    (["where teacher", "teacher where"], "where is teacher"),
    (["how teacher", "teacher how"], "how is teacher"),
    (["name teacher", "teacher name"], "what is teacher name"),
    (["teacher study", "study teacher"], "teacher is studying"),

    # sad + person
    (["mother sad", "sad mother"], "mother is sad"),
    (["father sad", "sad father"], "father is sad"),
    (["teacher sad", "sad teacher"], "teacher is sad"),
    (["sad home", "home sad"], "i am sad at home"),

    # study + place/person
    (["study home", "home study"], "i am studying at home"),
    (["mother study", "study mother"], "mother is studying"),
    (["father study", "study father"], "father is studying"),
    (["where study", "study where"], "where are you studying"),

    # why + X -- "why" asks about whatever follows it
    (["why sad", "sad why"], "why are you sad"),
    (["why study", "study why"], "why are you studying"),
    (["why laundry", "laundry why"], "why are you doing laundry"),
    (["why drive", "drive why"], "why are you driving"),
    (["why home", "home why"], "why are you home"),
    (["why mother sad", "mother why sad", "sad why mother"],
     "why is mother sad"),
    (["why father sad", "father why sad", "sad why father"],
     "why is father sad"),
]

# ---------------------------------------------------------------- #
# REDUCE -- natural human speech -> gloss keywords for display
# ADVISER PRIORITY. The name/pronoun block below is deliberately the
# largest and most varied section in this file.
# ---------------------------------------------------------------- #

REDUCE_PATTERNS = [
    # ============================================================
    # NAME + PRONOUN -- adviser requirement 2, heaviest coverage.
    # "name" alone is never a correct output for these.
    # ============================================================
    (["what is his name", "his name", "tell me his name",
      "do you know his name", "what is his name please",
      "can you tell me his name", "i want to know his name",
      "please tell me his name",
      "his name please", "and his name"], "name his"),
    (["what is her name", "her name", "tell me her name",
      "do you know her name", "what is her name please",
      "can you tell me her name", "i want to know her name",
      "please tell me her name",
      "her name please", "and her name"], "name her"),
    (["what is your name", "your name", "tell me your name",
      "do you know your name", "what is your name please",
      "can you tell me your name", "i want to know your name",
      "please tell me your name",
      "your name please", "and your name"], "name you"),
    (["what is my name", "do you know my name", "tell me my name",
      "can you tell me my name", "please tell me my name",
      "my name please", "and my name"], "name me"),

    # The v2 FAILURE CASE: "my mother name" must not collapse to
    # "name me" just because "my" is present. Extra phrasings here on
    # purpose, to separate the possessive "my" from the referent.
    # PHASE 3 BUGFIX: "your mother"/"your father" added -- this is
    # the exact case that was reported broken ("what is your father
    # name" incorrectly gave "name you").
    (["what is mother name", "what is my mother name",
      "what is your mother name", "tell me mother name",
      "tell me my mother name", "tell me your mother name",
      "do you know mother name", "do you know my mother name",
      "what is mom name",
      "your mother name", "mother name please"], "name mother"),
    (["what is father name", "what is my father name",
      "what is your father name", "tell me father name",
      "tell me my father name", "tell me your father name",
      "do you know father name", "do you know my father name",
      "what is dad name",
      "your father name", "father name please"], "name father"),

    # ============================================================
    # SUBJECT + ACTION -- drop "my", drop copulas
    # ============================================================
    (["my mother is driving", "mother is driving", "my mom is driving",
      "mom is driving", "is my mother driving", "your mother is driving",
      "is your mother driving", "mother drives", "is mother driving"],
     "mother driving"),
    (["my father is driving", "father is driving", "my dad is driving",
      "dad is driving", "is my father driving", "your father is driving",
      "is your father driving", "father drives", "is father driving"],
     "father driving"),
    (["she is driving", "is she driving", "she drives"], "she driving"),
    (["he is driving", "is he driving", "he drives"], "he driving"),
    (["i am driving", "am i driving", "i drive"], "i driving"),

    (["my mother is doing laundry", "mother is doing laundry",
      "my mom is doing the laundry", "mom is doing laundry",
      "your mother is doing laundry", "is mother doing laundry"],
     "mother laundry"),
    (["my father is doing laundry", "father is doing laundry",
      "my dad is doing the laundry", "dad is doing laundry",
      "your father is doing laundry", "is father doing laundry"],
     "father laundry"),
    (["she is doing laundry", "she does the laundry"], "she laundry"),
    (["he is doing laundry", "he does the laundry"], "he laundry"),
    (["i am doing laundry", "i do the laundry"], "i laundry"),

    # ============================================================
    # WHERE questions -- note "home"/"house" coverage, a v2 gap
    # ============================================================
    (["where is my mother", "where is mother", "where is my mom",
      "where is mom", "where is your mother", "where is she"],
     "where mother"),
    (["where is my father", "where is father", "where is my dad",
      "where is dad", "where is your father", "where is he"],
     "where father"),
    (["where is home", "where is the house", "where is my home",
      "where is my house", "where is the home"], "where home"),

    # ============================================================
    # HOME / location statements
    # ============================================================
    (["my mother is home", "mother is home", "is my mother home",
      "is mother home", "your mother is home", "mother is at home"],
     "mother home"),
    (["my father is home", "father is home", "is my father home",
      "is father home", "your father is home", "father is at home"],
     "father home"),
    (["she is home", "is she home", "she is at home"], "she home"),
    (["he is home", "is he home", "he is at home"], "he home"),
    (["i am home", "i am at home"], "i home"),

    # ============================================================
    # WELLBEING questions
    # ============================================================
    (["how are you", "how are you doing", "how are you today",
      "are you fine", "are you okay", "are you alright",
      "are you good", "are you doing fine", "are you doing okay"],
     "how you"),
    (["how is he", "how is he doing", "how is he today",
      "is he fine", "is he okay", "is he alright",
      "is he good", "is he doing fine"], "how he"),
    (["how is she", "how is she doing", "how is she today",
      "is she fine", "is she okay", "is she alright",
      "is she good", "is she doing fine"], "how she"),
    (["how is my mother", "how is mother", "how is mother doing",
      "how is your mother", "is mother fine", "is mother okay",
      "is mother alright", "is mother doing fine"], "how mother"),
    (["how is my father", "how is father", "how is father doing",
      "how is your father", "is father fine", "is father okay",
      "is father alright", "is father doing fine"], "how father"),

    # ============================================================
    # WELLBEING answers
    # ============================================================
    (["i am fine", "i am okay", "i am good", "i am alright",
      "i am doing fine"], "i fine"),
    (["he is fine", "he is okay", "he is good", "he is alright"],
     "he fine"),
    (["she is fine", "she is okay", "she is good", "she is alright"],
     "she fine"),
    (["mother is fine", "my mother is fine", "mom is okay",
      "your mother is fine", "my mother is okay"], "mother fine"),
    (["father is fine", "my father is fine", "dad is okay",
      "your father is fine", "my father is okay"], "father fine"),

    # ============================================================
    # GOING HOME
    # ============================================================
    (["i am going home", "i am going to my house", "i will go home",
      "i will go to my house", "i am going to my home"], "i going home"),
    (["she is going home", "she is going to her house",
      "she will go home", "she will go to her house",
      "she is going to her home"], "she going home"),
    (["he is going home", "he is going to his house",
      "he will go home", "he will go to his house",
      "he is going to his home"], "he going home"),
    (["my mother is going home", "mother is going home",
      "my mom is going home", "your mother is going home",
      "mother will go home"], "mother going home"),
    (["my father is going home", "father is going home",
      "my dad is going home", "your father is going home",
      "father will go home"], "father going home"),
      (["what are you doing", "tell me what you are doing"], "you doing"),
    (["what is he doing"], "he doing"),
    (["what is she doing"], "she doing"),
    (["what is my mother doing", "what is mother doing",
      "what is your mother doing"], "mother doing"),
    (["what is my father doing", "what is father doing",
      "what is your father doing"], "father doing"),

    # ============================================================
    # PHASE 4 -- sad, study, teacher, why
    # ============================================================
    (["are you sad", "you are sad"], "you sad"),
    (["he is sad", "is he sad"], "he sad"),
    (["she is sad", "is she sad"], "she sad"),
    (["mother is sad", "my mother is sad", "your mother is sad",
      "is mother sad"], "mother sad"),
    (["father is sad", "my father is sad", "your father is sad",
      "is father sad"], "father sad"),
    (["teacher is sad", "is teacher sad", "the teacher is sad"],
     "teacher sad"),

    (["why are you sad", "why you are sad"], "why you sad"),
    (["why is he sad", "why he is sad"], "why he sad"),
    (["why is she sad", "why she is sad"], "why she sad"),
    (["why is mother sad", "why is my mother sad",
      "why is your mother sad"], "why mother sad"),
    (["why is father sad", "why is my father sad",
      "why is your father sad"], "why father sad"),
    (["why are you studying", "why do you study"], "why you studying"),
    (["why are you driving", "why do you drive"], "why you driving"),
    (["why are you doing laundry"], "why you laundry"),

    (["what are you studying", "are you studying"], "you studying"),
    (["he is studying", "is he studying"], "he studying"),
    (["she is studying", "is she studying"], "she studying"),
    (["mother is studying", "my mother is studying",
      "is mother studying"], "mother studying"),
    (["father is studying", "my father is studying",
      "is father studying"], "father studying"),
    (["i am studying", "i study"], "i studying"),

    (["where is teacher", "where is the teacher", "where is my teacher",
      "where is your teacher"], "where teacher"),
    (["how is teacher", "how is the teacher", "how is your teacher",
      "is teacher fine", "is teacher okay"], "how teacher"),
    (["what is teacher name", "what is the teacher name",
      "what is your teacher name", "tell me teacher name",
      "teacher name", "teacher name please"], "name teacher"),
    (["teacher is fine", "teacher is okay", "the teacher is fine"],
     "teacher fine"),
    (["teacher is home", "is teacher home", "the teacher is home"],
     "teacher home"),
    (["teacher is driving", "the teacher is driving"], "teacher driving"),
    (["teacher is doing laundry"], "teacher laundry"),
    (["teacher is studying", "is teacher studying"], "teacher studying"),
]


def build_rows(patterns, task, prefix):
    rows = []
    for inputs, output in patterns:
        for inp in inputs:
            rows.append({
                "task": task,
                "input": inp,
                "output": output,
                "pattern": prefix + output,
            })
    return rows


def build_restructure_rows():
    """
    Scrambled gloss order -> canonical order. Derived from the Complete
    patterns' gloss sequences so the two tasks stay consistent by
    construction. Single-gesture patterns are skipped -- there is
    nothing to reorder in a one-word sequence.
    """
    rows = []
    for inputs, _ in COMPLETE_PATTERNS:
        if len(inputs[0].split()) < 2:
            continue
        correct = inputs[0]
        for variant in inputs:
            rows.append({
                "task": "restructure",
                "input": variant,
                "output": correct,
                "pattern": "restructure:" + correct,
            })
    return rows


def split_by_phrasing(rows, holdout_per_pattern=2, min_train_phrasings=2, seed=SEED):
    """
    Splits by PHRASING within each pattern, not by whole pattern.

    Every pattern keeps at least `min_train_phrasings` examples in
    training, so the model always has signal for that meaning. Held-out
    rows are DIFFERENT WAYS OF SAYING a meaning the model has seen.

    This tests whether the system can handle phrasings it was never
    trained on, rather than whether it can invent a word combination
    it has never seen in any form -- a much harder problem that also
    does not reflect how the deployed model is trained (on all data).
    """
    by_pattern = {}
    for r in rows:
        by_pattern.setdefault(r["pattern"], []).append(r)

    rng = random.Random(seed)
    train, test = [], []
    for pattern, group in sorted(by_pattern.items()):
        group = list(group)
        rng.shuffle(group)
        n_holdout = min(holdout_per_pattern,
                        max(0, len(group) - min_train_phrasings))
        test.extend(group[:n_holdout])
        train.extend(group[n_holdout:])
    return train, test, set()


def main():
    complete_rows = build_rows(COMPLETE_PATTERNS, "complete", "complete:")
    reduce_rows = build_rows(REDUCE_PATTERNS, "reduce", "reduce:")
    restructure_rows = build_restructure_rows()
    all_rows = complete_rows + reduce_rows + restructure_rows

    train, test, _ = split_by_phrasing(all_rows)

    with open("dataset_v3_train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open("dataset_v3_test.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")

    print("=== Phase 3 dataset generated ===\n")
    print(f"Complete rows     : {len(complete_rows)}")
    print(f"Reduce rows       : {len(reduce_rows)}")
    print(f"Restructure rows  : {len(restructure_rows)}")
    print(f"TOTAL             : {len(all_rows)}   (v2: 312, v1: 139)\n")

    n_patterns = len({r["pattern"] for r in all_rows})
    print(f"Distinct patterns : {n_patterns}")
    print(f"Avg phrasings per pattern: {len(all_rows) / n_patterns:.1f}\n")

    name_rows = [r for r in reduce_rows if r["output"].startswith("name")]
    print(f"ADVISER PRIORITY -- name/pronoun Reduce rows: {len(name_rows)}")
    for out in sorted({r["output"] for r in name_rows}):
        n = len([r for r in name_rows if r["output"] == out])
        print(f"    {out!r:16s} <- {n} different phrasings")

    print(f"\nTRAIN rows: {len(train)}  ->  dataset_v3_train.jsonl")
    print(f"TEST rows : {len(test)}  ->  dataset_v3_test.jsonl")
    print(f"\nHeld-out PHRASINGS (every pattern still has training signal):")
    for r in test[:12]:
        print(f"    [{r['task']}] {r['input']!r} -> {r['output']!r}")
    print(f"    ... {len(test)} held-out phrasings total")

    longest_in = max(len(r["input"].split()) for r in all_rows)
    longest_out = max(len(r["output"].split()) for r in all_rows)
    print(f"\nLongest input : {longest_in} words")
    print(f"Longest output: {longest_out} words")


if __name__ == "__main__":
    main()