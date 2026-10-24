"""
generate_dataset_v4.py -- builds the training data for the full
39-sign vocabulary confirmed by the recognition team.

WHY THIS IS A GENERATOR AND NOT A LIST
With 14 signs the patterns could be written out by hand. With 39 they
cannot, and hand-writing them would drift back toward exactly the
"fixed set of sentences" the adviser objected to. Instead the signs
are grouped into CATEGORIES, and the patterns are produced by RULES
that combine those categories:

    PERSON + STATE      -> "<person> is <state>"
    where  + PERSON     -> "where is <person>"
    want   + THING      -> "i want <thing>"

Adding a new sign later means adding one word to one category list --
every rule then covers it automatically. That is the difference
between a rule-driven dataset and a hand-written one.

IMPORTANT ASYMMETRY BETWEEN THE TWO DIRECTIONS
  COMPLETE (Deaf -> hearing): the INPUT is constrained to the 39 signs,
      because that is all the camera can recognize.
  REDUCE (hearing -> Deaf): the INPUT is ordinary spoken English and is
      NOT constrained -- the hearing person talks into a microphone.
      Only the OUTPUT is limited to signs the Deaf reader knows.
So the sign list shapes Complete's inputs, and Reduce's outputs.

Run: python generate_dataset_v4.py
Produces: dataset_v4_train.jsonl, dataset_v4_test.jsonl
"""

import json
import random

SEED = 42

# the one word Reduce may add that was not spoken
QUESTION_MARKER_WORD = "how"

# ---------------------------------------------------------------- #
# THE 39 CONFIRMED SIGNS, grouped by how they behave grammatically.
# ---------------------------------------------------------------- #

PERSON   = ["mother", "father", "teacher", "family", "friend"]
STATE    = ["fine", "sad", "tired", "sick", "good", "deaf"]
ACTIVITY = ["drive", "laundry", "study", "learn", "eat"]
PLACE    = ["home", "school", "bathroom"]
THING    = ["water"]
QUESTION = ["what", "where", "why", "how", "who", "when"]
SOCIAL   = ["hello", "sorry", "please", "nice", "yes", "no",
            "understand", "finish"]
NEED     = ["want", "help"]
PRONOUN  = ["you", "my"]
# Possessive signs. These let the system say WHOSE family member is
# being discussed -- "her mother" vs "his mother" -- which is the
# adviser's two-Deaf-people scenario, solved without fingerspelling.
POSSESSIVE = ["his", "her", "my"]

ALL_SIGNS = (PERSON + STATE + ACTIVITY + PLACE + THING + QUESTION
             + SOCIAL + NEED + PRONOUN
             + ["his", "her"])

# How each sign reads inside a natural sentence.
VERB = {"drive": "driving", "laundry": "doing laundry",
        "study": "studying", "learn": "learning", "eat": "eating"}
PLACE_PHRASE = {"home": "home", "school": "school",
                "bathroom": "the bathroom"}
# Only mother/father carry a gendered follow-up pronoun; the others
# are deliberately left without one rather than assuming a gender.
PRONOUN_OF = {"mother": "she", "father": "he"}
PERSON_ARTICLE = {"mother": "mother", "father": "father",
                  "teacher": "teacher", "family": "my family",
                  "friend": "my friend"}

# ---------------------------------------------------------------- #
# SINGLE SIGNS -- what one sign alone means.
# ---------------------------------------------------------------- #

SINGLE = {
    # social words that stand alone as-is
    "hello": "hello",
    "yes": "yes",
    "no": "no",
    "please": "please",
    "sorry": "i am sorry",
    "nice": "nice to meet you",
    "understand": "i understand",
    "finish": "i am finished",
    # states -- subject is the signer
    "fine": "i am fine",
    "sad": "i am sad",
    "tired": "i am tired",
    "sick": "i am sick",
    "good": "i am good",
    "deaf": "i am deaf",
    # activities
    "drive": "i am driving",
    "laundry": "i am doing laundry",
    "study": "i am studying",
    "learn": "i am learning",
    "eat": "i am eating",
    # places
    "home": "i am home",
    "school": "i am at school",
    "bathroom": "i need the bathroom",
    # things and needs
    "water": "i want water",
    "want": "i want",
    "help": "help me",
    # people -- introducing someone
    "mother": "this is mother",
    "father": "this is father",
    "teacher": "this is teacher",
    "family": "this is my family",
    "friend": "this is my friend",
    # identity and pronouns
    "you": "you",
    "my": "my",
    # question words alone, used as short responses
    "what": "what",
    "where": "where",
    "why": "why",
    "how": "how",
    "who": "who",
    "when": "when",
}


def orderings(words):
    """The gesture orders the recognizer might plausibly deliver.
    Canonical order first, then the reverse for two-sign phrases."""
    if len(words) == 2:
        return [" ".join(words), " ".join(reversed(words))]
    return [" ".join(words)]


def build_complete():
    """
    Gloss sequence (signs only) -> natural sentence.

    EXHAUSTIVE BY CONSTRUCTION. Every rule below enumerates EVERY
    combination its categories allow, rather than the handful someone
    thought to write down. That is the difference that matters: the
    earlier version used the same categories but only for the pairings
    that had been listed by hand, so ordinary things like "hello friend"
    (SOCIAL + PERSON) and "you want help" (PRONOUN + NEED + THING) had
    no rule at all and the model had to guess.

    A combination is generated when it is grammatically sensible.
    Combinations that cannot mean anything -- a thing being sad, a
    greeting aimed at water -- are deliberately not generated, because
    teaching nonsense is worse than leaving a gap.
    """
    pats = []

    def add(words, sentence):
        pats.append((orderings(words), sentence))

    # ---------- one sign ----------
    for sign, sentence in SINGLE.items():
        pats.append(([sign], sentence))

    # ---------- PERSON + something ----------
    for pr in PERSON:
        who = PERSON_ARTICLE[pr]
        for st in STATE:
            add([pr, st], f"{who} is {st}")
        for a in ACTIVITY:
            add([pr, a], f"{who} is {VERB[a]}")
        for pl in PLACE:
            loc = "home" if pl == "home" else f"at {PLACE_PHRASE[pl]}"
            add([pr, pl], f"is {who} {loc}")
        add(["where", pr], f"where is {who}")
        add(["how", pr], f"how is {who}")
        add(["who", pr], f"who is {who}")
        for so in SOCIAL:
            if so in ("hello", "sorry", "nice", "please"):
                greet = {"hello": f"hello {pr}",
                         "sorry": f"i am sorry {pr}",
                         "nice": f"nice to meet you {pr}",
                         "please": f"please {pr}"}[so]
                add([so, pr], greet)

    # ---------- POSSESSIVE + PERSON, and what can be said about them ----------
    for po in POSSESSIVE:
        for pr in PERSON:
            add([po, pr], f"this is {po} {pr}")
            add(["where", po, pr], f"where is {po} {pr}")
            add(["how", po, pr], f"how is {po} {pr}")
            add(["who", po, pr], f"who is {po} {pr}")
            for st in STATE:
                add([po, pr, st], f"{po} {pr} is {st}")
            for a in ACTIVITY:
                add([po, pr, a], f"{po} {pr} is {VERB[a]}")
            for pl in PLACE:
                loc = "home" if pl == "home" else f"at {PLACE_PHRASE[pl]}"
                add([po, pr, pl], f"is {po} {pr} {loc}")

    # ---------- QUESTION + thing being asked about ----------
    for pl in PLACE:
        add(["where", pl], f"where is {PLACE_PHRASE[pl]}")
    for a in ACTIVITY:
        add(["where", a], f"where are you {VERB[a]}")
        add(["when", a], f"when are you {VERB[a]}")
        add(["why", a], f"why are you {VERB[a]}")
        add(["what", a], f"what are you {VERB[a]}")
    for st in STATE:
        add(["why", st], f"why are you {st}")
    for t in THING:
        add(["where", t], f"where is the {t}")

    # ---------- why + PERSON + STATE ----------
    for pr in PERSON:
        for st in STATE:
            add(["why", pr, st], f"why is {PERSON_ARTICLE[pr]} {st}")

    # ---------- QUESTION + PERSON + ACTIVITY ----------
    # "when father finish" was not covered: questions about a PERSON
    # doing an ACTIVITY only existed without the person.
    for pr in PERSON:
        who = PERSON_ARTICLE[pr]
        for a in ACTIVITY:
            add(["where", pr, a], f"where is {who} {VERB[a]}")
            add(["when", pr, a], f"when will {who} {a}")
            add(["why", pr, a], f"why is {who} {VERB[a]}")
            add(["what", pr, a], f"what is {who} {VERB[a]}")
        add(["when", pr, "finish"], f"when will {who} finish")
        add(["what", pr, "finish"], f"has {who} finished")
        add([pr, "finish"], f"{who} is finished")
        for a in ACTIVITY:
            add([pr, "finish", a], f"{who} is finished {VERB[a]}")

    # ---------- PRONOUN + something ----------
    # "you" is the person being addressed; "my" is the signer.
    for st in STATE:
        add(["you", st], f"you are {st}")
        add(["my", st], f"i am {st}")
    for a in ACTIVITY:
        add(["you", a], f"are you {VERB[a]}")
        add(["my", a], f"i am {VERB[a]}")
    for pl in PLACE:
        loc = "home" if pl == "home" else f"at {PLACE_PHRASE[pl]}"
        add(["you", pl], f"are you {loc}")
        add(["my", pl], f"i am {loc}")

    # ---------- NEED, alone and with a subject ----------
    for t in THING:
        add(["want", t], f"i want {t}")
        add(["you", "want", t], f"do you want {t}")
        add(["my", "want", t], f"i want {t}")
    for a in ACTIVITY:
        add(["want", a], f"i want to {a}")
        add(["you", "want", a], f"do you want to {a}")
        add(["my", "want", a], f"i want to {a}")
    for pl in PLACE:
        add(["want", pl], f"i want to go to {PLACE_PHRASE[pl]}")
        add(["you", "want", pl], f"do you want to go to {PLACE_PHRASE[pl]}")
        add(["my", "want", pl], f"i want to go to {PLACE_PHRASE[pl]}")
    add(["want", "help"], "i want help")
    add(["you", "want", "help"], "do you want help")
    add(["my", "want", "help"], "i want help")
    add(["you", "help"], "can i help you")
    add(["my", "help"], "i need help")
    for a in ACTIVITY:
        add(["help", a], f"help me {a}")
    for pl in PLACE:
        add(["help", pl], f"help me go to {PLACE_PHRASE[pl]}")

    # ---------- ACTIVITY + PLACE, and with a person ----------
    for a in ACTIVITY:
        for pl in PLACE:
            if pl == "bathroom":
                continue
            loc = "at home" if pl == "home" else f"at {PLACE_PHRASE[pl]}"
            add([a, pl], f"i am {VERB[a]} {loc}")
            for pr in PERSON:
                add([pr, a, pl],
                    f"{PERSON_ARTICLE[pr]} is {VERB[a]} {loc}")

    # ---------- finish + activity ----------
    for a in ACTIVITY:
        add(["finish", a], f"i am finished {VERB[a]}")
        add(["you", "finish", a], f"are you finished {VERB[a]}")

    # ---------- SOCIAL pairs that occur together ----------
    add(["yes", "please"], "yes please")
    add(["no", "sorry"], "no i am sorry")
    add(["yes", "understand"], "yes i understand")
    add(["no", "understand"], "no i do not understand")
    add(["yes", "finish"], "yes i am finished")
    add(["hello", "friend"], "hello friend")

    return pats



def build_reduce():
    """Natural spoken English -> gloss keywords for the Deaf reader.

    The hearing person speaks freely, so several phrasings are given
    for each meaning -- that variety is what forces the model to learn
    the underlying keep/drop rule instead of memorizing sentences.
    """
    pats = []

    # PERSON + STATE
    for p in PERSON:
        for s in STATE:
            pats.append(([f"{p} is {s}", f"my {p} is {s}",
                          f"your {p} is {s}", f"{p} is very {s}"],
                         f"{p} {s}"))

    # PERSON + ACTIVITY
    for p in PERSON:
        for a in ACTIVITY:
            v = VERB[a]
            pats.append(([f"{p} is {v}", f"my {p} is {v}",
                          f"your {p} is {v}", f"is {p} {v}"],
                         f"{p} {a}"))

    # where + PERSON
    for p in PERSON:
        pats.append(([f"where is {p}", f"where is my {p}",
                      f"where is your {p}"], f"where {p}"))
    # where + PLACE
    for pl in PLACE:
        pats.append(([f"where is {pl}", f"where is the {pl}",
                      f"where is my {pl}"], f"where {pl}"))

    # how + PERSON  (wellbeing question -- some phrasings need "how"
    # supplied by the model because it is absent from the input)
    for p in PERSON:
        pats.append(([f"how is {p}", f"how is my {p}", f"how is your {p}",
                      f"is {p} fine", f"is {p} okay"], f"how {p}"))
    pats.append((["how are you", "how are you doing", "are you fine",
                  "are you okay", "are you alright"], "how you"))

    # name + PERSON

    # why + STATE
    for s in STATE:
        pats.append(([f"why are you {s}", f"why you are {s}",
                      f"why are you so {s}", f"tell me why you are {s}"],
                     f"why you {s}"))
    for p in PERSON:
        for s in STATE:
            pats.append(([f"why is {p} {s}", f"why is my {p} {s}",
                          f"why is your {p} {s}"], f"why {p} {s}"))

    # POSSESSIVE + PERSON -- preserves WHOSE relative is meant.
    # Only third-person here: "my father" said by the hearing person
    # drops the "my" (handled by the plain PERSON patterns above),
    # while "his/her father" must keep it or the referent is lost.
    for poss in ["his", "her"]:
        for p in PERSON:
            pats.append(([f"where is {poss} {p}", f"where {poss} {p} is"],
                         f"where {poss} {p}"))
            pats.append(([f"how is {poss} {p}", f"how is {poss} {p} doing",
                          f"is {poss} {p} okay",
                          f"tell me how {poss} {p} is"],
                         f"how {poss} {p}"))
            for st in STATE:
                pats.append(([f"{poss} {p} is {st}", f"is {poss} {p} {st}"],
                             f"{poss} {p} {st}"))

    # SELF + STATE  ->  just the state.
    # Symmetric with Complete, where a lone state sign means the signer
    # ("fine" -> "i am fine"). When the hearing person says "i am fine"
    # the speaker is obvious, so the pronoun carries no information and
    # is dropped -- the same reason "my father" drops the "my".
    for st in STATE:
        pats.append(([f"i am {st}", f"i am very {st}", f"i am so {st}",
                      f"i am still {st}"], st))

    # SELF + ACTIVITY  ->  just the activity
    for a in ACTIVITY:
        pats.append(([f"i am {VERB[a]}", f"i am {VERB[a]} now",
                      f"i am still {VERB[a]}"], a))

    # YOU + STATE, stated rather than asked. The question forms
    # ("are you fine") belong to the how-question patterns instead.
    for st in STATE:
        pats.append(([f"you are {st}", f"you are very {st}",
                      f"you look {st}"], f"you {st}"))

    # when + ACTIVITY
    for a in ACTIVITY:
        pats.append(([f"when are you {VERB[a]}", f"when do you {a}",
                      f"when will you {a}", f"tell me when you {a}"],
                     f"when you {a}"))

    # who + PERSON
    for p in PERSON:
        pats.append(([f"who is {p}", f"who is my {p}", f"who is your {p}"],
                     f"who {p}"))

    # want / need
    for t in THING:
        pats.append(([f"do you want {t}", f"do you need {t}",
                      f"i want {t}"], f"want {t}"))
    for a in ACTIVITY:
        pats.append(([f"do you want to {a}", f"i want to {a}",
                      f"do you need to {a}", f"i need to {a}"],
                     f"want {a}"))
    for pl in PLACE:
        pats.append(([f"do you want to go to the {pl}",
                      f"i want to go to the {pl}",
                      f"i need to go to the {pl}",
                      f"do you need the {pl}"], f"want {pl}"))

    # help -- split by whether the sentence names who is being helped
    pats.append((["do you need help", "i need help", "i want help",
                  "do you want help"], "help"))
    pats.append((["can i help you", "i can help you", "i will help you"],
                 "help you"))

    # social
    pats.append((["hello", "hello there", "hi", "hi there"], "hello"))
    pats.append((["yes", "yes i do", "yes i am", "yes it is"], "yes"))
    pats.append((["no", "no i do not", "no i am not"], "no"))
    pats.append((["sorry", "i am sorry", "i am so sorry",
                  "i am really sorry"], "sorry"))
    pats.append((["nice to meet you", "it is nice to meet you",
                  "so nice to meet you", "very nice to meet you",
                  "nice"], "nice"))
    pats.append((["do you understand", "i understand",
                  "do you understand me", "i do not understand",
                  "yes i understand", "i understand now",
                  "do you understand it"], "understand"))
    pats.append((["are you finished", "i am finished", "are you done",
                  "i am done", "i finished", "are you finished now",
                  "yes i am finished"], "finish"))
    pats.append((["please", "please do", "yes please do"], "please"))
    pats.append((["please help me", "please help", "help me please"],
                 "please help"))

    # PERSON + PLACE
    for p in PERSON:
        for pl in PLACE:
            pats.append(([f"{p} is at {pl}", f"my {p} is at {pl}",
                          f"is {p} at {pl}"], f"{p} {pl}"))

    # ACTIVITY + PLACE
    for a in ACTIVITY:
        for pl in PLACE:
            if pl == "bathroom":
                continue
            pats.append(([f"i am {VERB[a]} at {pl}",
                          f"are you {VERB[a]} at {pl}",
                          f"i am {VERB[a]} at {pl} now",
                          f"are you still {VERB[a]} at {pl}"],
                         f"{a} {pl}"))

    return pats


def build_rows(patterns, task, prefix):
    rows = []
    for inputs, output in patterns:
        for inp in inputs:
            rows.append({"task": task, "input": inp, "output": output,
                         "pattern": prefix + output})
    return rows


def build_restructure(complete_pats):
    """Scrambled sign order -> canonical order, derived from the
    Complete patterns so the two tasks stay consistent by construction."""
    rows = []
    for inputs, _ in complete_pats:
        if len(inputs[0].split()) < 2:
            continue
        correct = inputs[0]
        for variant in inputs:
            rows.append({"task": "restructure", "input": variant,
                         "output": correct,
                         "pattern": "restructure:" + correct})
    return rows


def split_by_phrasing(rows, holdout_per_pattern=2, min_train=2, seed=SEED):
    """
    NEVER hold out a pattern that has only ONE phrasing.

    A single sign has exactly one way of being signed, so holding it
    out removes the only example of what that sign means alone. That is
    not testing generalisation, it is deleting a fact. It happened:
    "water" alone was held out, so the model never learned it means
    "i want water" and answered "this is the water" instead, borrowing
    the pattern it knew from mother and father.

    Patterns with 2+ phrasings are still split normally -- holding out
    a DIFFERENT WAY of saying something the model has seen is a fair
    test. Holding out the only way is not.
    """
    """Hold out DIFFERENT PHRASINGS of meanings the model has seen,
    keeping at least `min_train` phrasings of every pattern in
    training. This tests whether the system handles new ways of saying
    known things -- the adviser's actual concern -- rather than asking
    it to invent combinations it has never seen in any form."""
    by_pattern = {}
    for r in rows:
        by_pattern.setdefault(r["pattern"], []).append(r)
    rng = random.Random(seed)
    train, test = [], []
    for _, group in sorted(by_pattern.items()):
        group = list(group)

        # A SINGLE-SIGN input is a primitive fact, not a phrasing.
        # "water" alone is the only way to sign that meaning, so holding
        # it out deletes the fact rather than testing generalisation --
        # the model then answered "this is the water" because it had
        # never been told otherwise. Longer inputs that mean the same
        # thing ("want water") do not teach what the bare sign means.
        primitives = [r for r in group if len(r["input"].split()) == 1]
        rest = [r for r in group if len(r["input"].split()) > 1]
        train.extend(primitives)

        if len(rest) < 2:
            train.extend(rest)
            continue
        rng.shuffle(rest)
        n = min(holdout_per_pattern, max(0, len(rest) - min_train))
        test.extend(rest[:n])
        train.extend(rest[n:])
    return train, test


def check_task_shapes(rows):
    """
    Complete INPUTS must be signs only -- that is all a camera can
    produce. Reduce OUTPUTS must be signs only -- that is all the Deaf
    reader can be shown. Violations mean a block of patterns ended up
    in the wrong builder, which produced a model that answered the
    wrong question entirely ("you sick" -> "family home"). The
    contradiction check cannot see this, because each row is
    internally consistent; only the SHAPE is wrong.
    """
    signs = set(ALL_SIGNS)
    problems = []
    for r in rows:
        if r["task"] == "complete":
            bad = [w for w in r["input"].split() if w not in signs]
            if bad:
                problems.append(("complete input has non-signs",
                                 r["input"], bad))
        elif r["task"] == "reduce":
            bad = [w for w in r["output"].split()
                   if w not in signs and w != QUESTION_MARKER_WORD]
            if bad:
                problems.append(("reduce output has non-signs",
                                 r["output"], bad))
    return problems


def check_no_contradictions(rows):
    """
    The same input must never be paired with two different answers.
    Contradictory rows cannot be learned -- the model is forced to
    guess between them -- and the reconstruction check cannot catch
    this, because each row is individually valid. This guard exists
    because exactly that bug reached training once.
    """
    seen = {}
    clashes = []
    for r in rows:
        key = (r["task"], r["input"])
        if key in seen and seen[key] != r["output"]:
            clashes.append((r["task"], r["input"], seen[key], r["output"]))
        seen[key] = r["output"]
    return clashes


def main():
    complete_pats = build_complete()
    reduce_pats = build_reduce()

    complete_rows = build_rows(complete_pats, "complete", "complete:")
    reduce_rows = build_rows(reduce_pats, "reduce", "reduce:")
    restructure_rows = build_restructure(complete_pats)
    all_rows = complete_rows + reduce_rows + restructure_rows

    shape_problems = check_task_shapes(all_rows)
    if shape_problems:
        print(f"ABORTING -- {len(shape_problems)} rows have the wrong shape "
              f"for their task:\n")
        for kind, text, bad in shape_problems[:12]:
            print(f"   {kind}: {text!r}  offending: {bad}")
        raise SystemExit("A block of patterns is in the wrong builder.")

    clashes = check_no_contradictions(all_rows)
    if clashes:
        print(f"ABORTING -- {len(clashes)} inputs are taught two different "
              f"answers:\n")
        for task, inp, a, b in clashes[:15]:
            print(f"   [{task}] {inp!r}  ->  {a!r}  AND  {b!r}")
        raise SystemExit("Fix the contradictions before training.")

    train, test = split_by_phrasing(all_rows)

    with open("dataset_v4_train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open("dataset_v4_test.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")

    print("=== Phase 5 dataset (39 signs) ===\n")
    print(f"Signs covered     : {len(ALL_SIGNS)}")
    print(f"Complete rows     : {len(complete_rows)}")
    print(f"Reduce rows       : {len(reduce_rows)}")
    print(f"Restructure rows  : {len(restructure_rows)}")
    print(f"TOTAL             : {len(all_rows)}   (v3 was 568)\n")

    n_pat = len({r["pattern"] for r in all_rows})
    print(f"Distinct patterns : {n_pat}")
    print(f"Avg phrasings/pattern: {len(all_rows)/n_pat:.1f}\n")
    print(f"TRAIN rows: {len(train)}  ->  dataset_v4_train.jsonl")
    print(f"TEST rows : {len(test)}  ->  dataset_v4_test.jsonl\n")

    # every sign must appear in at least one Complete input
    used = set()
    for r in complete_rows:
        used.update(r["input"].split())
    missing = [s for s in ALL_SIGNS if s not in used]
    print("Signs never appearing in a Complete input:",
          missing if missing else "none -- all 39 covered")

    longest_in = max(len(r["input"].split()) for r in all_rows)
    longest_out = max(len(r["output"].split()) for r in all_rows)
    print(f"\nLongest input : {longest_in} words")
    print(f"Longest output: {longest_out} words")


if __name__ == "__main__":
    main()