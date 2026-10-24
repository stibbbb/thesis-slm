"""
generate_dataset_v4.py -- builds the training data for the 34 signs
confirmed by the recognition team.

WHY THIS IS A GENERATOR AND NOT A LIST
The signs are grouped into CATEGORIES, and the patterns are produced by
RULES that combine those categories:

    PERSON + STATE      -> "<person> is <state>"
    where  + PERSON     -> "where is <person>"
    ACTIVITY + TIME     -> "i will study tomorrow"

Adding a new sign later means adding one word to one category list --
every rule then covers it automatically. That is the difference
between a rule-driven dataset and a hand-written one.

IMPORTANT ASYMMETRY BETWEEN THE TWO DIRECTIONS
  COMPLETE (Deaf -> hearing): the INPUT is constrained to the 34 signs,
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

from vocab_pool import CONFIRMED_GESTURES

SEED = 42

# the one word Reduce may add that was not spoken
QUESTION_MARKER_WORD = "how"

# ---------------------------------------------------------------- #
# THE 34 CONFIRMED SIGNS, grouped by how they behave grammatically.
# ---------------------------------------------------------------- #

PERSON   = ["mother", "father", "teacher", "family", "friend"]
STATE    = ["fine", "good", "deaf"]
ACTIVITY = ["study", "learn", "eat"]
PLACE    = ["home", "school", "class", "here"]
THING    = ["water"]
QUESTION = ["what", "where", "why", "how"]
SOCIAL   = ["hello", "sorry", "please", "nice", "yes", "understand", "again"]
NEED     = ["help"]
PRONOUN  = ["you", "my"]
TIME     = ["time", "now", "today", "tomorrow"]
WHEN     = ["now", "today", "tomorrow"]     # the time words that say WHEN

ALL_SIGNS = (PERSON + STATE + ACTIVITY + PLACE + THING + QUESTION
             + SOCIAL + NEED + PRONOUN + TIME)
assert sorted(ALL_SIGNS) == sorted(CONFIRMED_GESTURES), "sign lists out of sync"

# How each sign reads inside a natural sentence.
VERB = {"study": "studying", "learn": "learning", "eat": "eating"}
# being AT a place, and going TO it
AT = {"home": "home", "school": "at school", "class": "in class", "here": "here"}
TO = {"home": "home", "school": "to school", "class": "to class"}
PERSON_ARTICLE = {"mother": "mother", "father": "father",
                  "teacher": "teacher", "family": "my family",
                  "friend": "my friend"}

# ---------------------------------------------------------------- #
# SINGLE SIGNS -- what one sign alone means.
# ---------------------------------------------------------------- #

SINGLE = {
    # social words
    "hello": "hello",
    "yes": "yes",
    "please": "please",
    "sorry": "i am sorry",
    "nice": "nice to meet you",
    "understand": "i understand",
    "again": "say it again",
    # states -- subject is the signer
    "fine": "i am fine",
    "good": "i am good",
    "deaf": "i am deaf",
    # activities
    "study": "i am studying",
    "learn": "i am learning",
    "eat": "i am eating",
    # places
    "home": "i am home",
    "school": "i am at school",
    "class": "i am in class",
    "here": "i am here",
    # things and needs
    "water": "i want water",
    "help": "help me",
    # people -- introducing someone
    "mother": "this is mother",
    "father": "this is father",
    "teacher": "this is teacher",
    "family": "this is my family",
    "friend": "this is my friend",
    # pronouns
    "you": "you",
    "my": "my",
    # question words alone, used as short responses
    "what": "what",
    "where": "where",
    "why": "why",
    "how": "how",
    # time
    "time": "what time is it",
    "now": "now",
    "today": "today",
    "tomorrow": "tomorrow",
}


def orderings(words):
    """The gesture orders the recognizer might plausibly deliver.
    Canonical order first, then the reverse for two-sign phrases."""
    if len(words) == 2:
        return [" ".join(words), " ".join(reversed(words))]
    return [" ".join(words)]


def when_phrase(subject, activity, t):
    """'i am studying today' / 'i will study tomorrow' for a subject."""
    be = {"i": "i am", "you": "are you"}.get(subject, f"{subject} is")
    will = {"i": "i will", "you": "will you"}.get(subject, f"{subject} will")
    if t == "tomorrow":
        return f"{will} {activity} tomorrow"
    return f"{be} {VERB[activity]} {t}"


def build_complete():
    """
    Gloss sequence (signs only) -> natural sentence.

    EXHAUSTIVE BY CONSTRUCTION: every rule enumerates every combination
    its categories allow. Combinations that cannot mean anything -- a
    thing being sad, a greeting aimed at water -- are deliberately not
    generated, because teaching nonsense is worse than leaving a gap.
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
            add([pr, pl], f"is {who} {AT[pl]}")
        add(["where", pr], f"where is {who}")
        add(["how", pr], f"how is {who}")
        add(["hello", pr], f"hello {pr}")
        add(["sorry", pr], f"i am sorry {pr}")
        add(["nice", pr], f"nice to meet you {pr}")
        add(["please", pr], f"please {pr}")

    # ---------- my + PERSON, and what can be said about them ----------
    for pr in PERSON:
        add(["my", pr], f"this is my {pr}")
        add(["where", "my", pr], f"where is my {pr}")
        add(["how", "my", pr], f"how is my {pr}")
        for st in STATE:
            add(["my", pr, st], f"my {pr} is {st}")
        for a in ACTIVITY:
            add(["my", pr, a], f"my {pr} is {VERB[a]}")
        for pl in PLACE:
            add(["my", pr, pl], f"is my {pr} {AT[pl]}")

    # ---------- QUESTION + thing being asked about ----------
    for pl in ["home", "school", "class"]:
        add(["where", pl], f"where is {'the ' if pl == 'class' else ''}{pl}")
    for a in ACTIVITY:
        add(["where", a], f"where are you {VERB[a]}")
        add(["why", a], f"why are you {VERB[a]}")
        add(["what", a], f"what are you {VERB[a]}")
    for st in STATE:
        add(["why", st], f"why are you {st}")
    for t in THING:
        add(["where", t], f"where is the {t}")
    add(["where", "you"], "where are you")
    add(["how", "you"], "how are you")

    # ---------- QUESTION + PERSON + STATE / ACTIVITY ----------
    for pr in PERSON:
        who = PERSON_ARTICLE[pr]
        for st in STATE:
            add(["why", pr, st], f"why is {who} {st}")
        for a in ACTIVITY:
            add(["where", pr, a], f"where is {who} {VERB[a]}")
            add(["why", pr, a], f"why is {who} {VERB[a]}")
            add(["what", pr, a], f"what is {who} {VERB[a]}")

    # ---------- PRONOUN + something ----------
    # "you" is the person being addressed; "my" is the signer.
    for st in STATE:
        add(["you", st], f"you are {st}")
        add(["my", st], f"i am {st}")
    for a in ACTIVITY:
        add(["you", a], f"are you {VERB[a]}")
        add(["my", a], f"i am {VERB[a]}")
    for pl in PLACE:
        add(["you", pl], f"are you {AT[pl]}")
        add(["my", pl], f"i am {AT[pl]}")

    # ---------- water and help ----------
    add(["you", "water"], "do you want water")
    add(["please", "water"], "water please")
    add(["you", "help"], "can i help you")
    add(["my", "help"], "i need help")
    add(["please", "help"], "please help me")
    for a in ACTIVITY:
        add(["help", a], f"help me {a}")
    for pl in TO:
        add(["help", pl], f"help me go {TO[pl]}")

    # ---------- ACTIVITY + PLACE, and with a person ----------
    for a in ACTIVITY:
        for pl in PLACE:
            loc = "at home" if pl == "home" else AT[pl]
            add([a, pl], f"i am {VERB[a]} {loc}")
            for pr in PERSON:
                add([pr, a, pl], f"{PERSON_ARTICLE[pr]} is {VERB[a]} {loc}")

    # ---------- TIME: now / today / tomorrow ----------
    for t in WHEN:
        for a in ACTIVITY:
            add([a, t], when_phrase("i", a, t))
            add(["you", a, t], when_phrase("you", a, t))
            for pr in PERSON:
                add([pr, a, t], when_phrase(PERSON_ARTICLE[pr], a, t))
        for pl in TO:
            go = "i will go" if t == "tomorrow" else "i am going"
            add([pl, t], f"{go} {TO[pl]} {t}")
            you_go = "will you go" if t == "tomorrow" else "are you going"
            add(["you", pl, t], f"{you_go} {TO[pl]} {t}")
        for st in STATE:
            add([st, t], f"i am {st} {t}")
        add(["how", "you", t], f"how are you {t}")
    add(["where", "you", "now"], "where are you now")
    add(["understand", "now"], "i understand now")

    # ---------- TIME: the sign "time" ----------
    add(["what", "time"], "what time is it")
    add(["what", "time", "now"], "what time is it now")
    for pl in ["school", "class"]:
        add(["what", "time", pl], f"what time is {pl}")
        add(["time", pl], f"it is time for {pl}")
    for a in ACTIVITY:
        add(["what", "time", a], f"what time do we {a}")
        add(["time", a], f"it is time to {a}")
    add(["time", "home"], "it is time to go home")

    # ---------- again ----------
    add(["please", "again"], "please say it again")
    add(["sorry", "again"], "sorry can you say it again")
    for a in ACTIVITY:
        add([a, "again"], f"i will {a} again")

    # ---------- SOCIAL pairs that occur together ----------
    add(["yes", "please"], "yes please")
    add(["yes", "understand"], "yes i understand")

    return pats


def build_reduce():
    """Natural spoken English -> gloss keywords for the Deaf reader.

    The hearing person speaks freely, so several phrasings are given
    for each meaning -- that variety is what forces the model to learn
    the underlying keep/drop rule instead of memorizing sentences.
    Words with no sign ("his", "who", "when", "want") appear in the
    phrasings so the model learns to DROP them.
    """
    pats = []

    # PERSON + STATE
    for p in PERSON:
        for s in STATE:
            pats.append(([f"{p} is {s}", f"my {p} is {s}", f"your {p} is {s}",
                          f"{p} is very {s}", f"her {p} is {s}"], f"{p} {s}"))

    # PERSON + ACTIVITY
    for p in PERSON:
        for a in ACTIVITY:
            v = VERB[a]
            pats.append(([f"{p} is {v}", f"my {p} is {v}",
                          f"your {p} is {v}", f"is {p} {v}",
                          f"his {p} is {v}"], f"{p} {a}"))

    # where + PERSON  (his / her have no sign, so they are dropped)
    for p in PERSON:
        pats.append(([f"where is {p}", f"where is my {p}", f"where is your {p}",
                      f"where is her {p}", f"where is his {p}"], f"where {p}"))
    # where + PLACE
    for pl in ["home", "school", "class"]:
        pats.append(([f"where is {pl}", f"where is the {pl}",
                      f"where is my {pl}", f"where is your {pl}"], f"where {pl}"))
    pats.append((["where are you", "tell me where you are",
                  "where are you going"], "where you"))
    pats.append((["where are you now", "where are you right now"],
                 "where you now"))

    # how + PERSON  (wellbeing question -- some phrasings need "how"
    # supplied by the model because it is absent from the input)
    for p in PERSON:
        pats.append(([f"how is {p}", f"how is my {p}", f"how is your {p}",
                      f"how is his {p}", f"is {p} fine", f"is {p} okay"],
                     f"how {p}"))
    pats.append((["how are you", "how are you doing", "are you fine",
                  "are you okay", "are you alright"], "how you"))
    pats.append((["how are you today", "how are you doing today",
                  "are you okay today"], "how you today"))

    # why + STATE
    for s in STATE:
        pats.append(([f"why are you {s}", f"why you are {s}",
                      f"why are you so {s}", f"tell me why you are {s}"],
                     f"why you {s}"))
    for p in PERSON:
        for s in STATE:
            pats.append(([f"why is {p} {s}", f"why is my {p} {s}",
                          f"why is your {p} {s}"], f"why {p} {s}"))

    # SELF + STATE  ->  just the state (the speaker is obvious)
    for st in STATE:
        pats.append(([f"i am {st}", f"i am very {st}", f"i am so {st}",
                      f"i am still {st}"], st))

    # SELF + ACTIVITY  ->  just the activity
    for a in ACTIVITY:
        pats.append(([f"i am {VERB[a]}", f"i am still {VERB[a]}",
                      f"i am really {VERB[a]}"], a))

    # YOU + STATE, stated rather than asked
    for st in STATE:
        pats.append(([f"you are {st}", f"you are very {st}",
                      f"you look {st}"], f"you {st}"))

    # what + ACTIVITY  ("when" has no sign: the question is asked with "what time")
    for a in ACTIVITY:
        pats.append(([f"what time do you {a}", f"what time will you {a}",
                      f"what time are you {VERB[a]}"], f"what time you {a}"))

    # water / help  ("want" and "need" have no sign, so they are dropped)
    pats.append((["do you want water", "do you need water",
                  "do you want some water"], "you water"))
    pats.append((["i want water", "i need water", "i want some water"], "water"))
    # "you" is kept when the help is for the listener, as in "you water"
    pats.append((["i need help", "i want help", "i need some help"], "help"))
    pats.append((["do you need help", "do you want help", "can i help you",
                  "i can help you", "i will help you"], "help you"))
    pats.append((["please help me", "please help", "help me please"],
                 "please help"))

    # social
    pats.append((["hello", "hello there", "hi", "hi there"], "hello"))
    pats.append((["yes", "yes i do", "yes i am", "yes it is"], "yes"))
    pats.append((["sorry", "i am sorry", "i am so sorry",
                  "i am really sorry"], "sorry"))
    pats.append((["nice to meet you", "it is nice to meet you",
                  "so nice to meet you", "very nice to meet you",
                  "nice"], "nice"))
    # "i do not understand" keeps "understand"; the negation is added on
    # screen by word_picker.restrict_to_signs, not learned here.
    pats.append((["do you understand", "i understand",
                  "do you understand me", "i do not understand",
                  "yes i understand", "do you understand it"], "understand"))
    pats.append((["i understand now", "do you understand now"],
                 "understand now"))
    pats.append((["please", "please do", "yes please do"], "please"))

    # again
    pats.append((["again", "say it again", "can you say that again",
                  "say that again"], "again"))
    pats.append((["please say it again", "please say that again",
                  "can you say that again please"], "please again"))
    for a in ACTIVITY:
        pats.append(([f"i will {a} again", f"i want to {a} again"],
                     f"{a} again"))

    # PERSON + PLACE  (here: "mother is here")
    for p in PERSON:
        for pl in PLACE:
            at = AT[pl] if pl != "home" else "at home"
            pats.append(([f"{p} is {at}", f"my {p} is {at}",
                          f"is {p} {at}"], f"{p} {pl}"))

    # come / be here
    pats.append((["come here", "come here please", "i am here"], "here"))
    pats.append((["are you here", "you are here"], "you here"))

    # ACTIVITY + PLACE
    for a in ACTIVITY:
        for pl in PLACE:
            at = AT[pl] if pl != "home" else "at home"
            pats.append(([f"i am {VERB[a]} {at}",
                          f"are you {VERB[a]} {at}",
                          f"are you still {VERB[a]} {at}"], f"{a} {pl}"))
            pats.append(([f"i am {VERB[a]} {at} now"], f"{a} {pl} now"))

    # TIME: now / today / tomorrow kept, because each is a sign
    for a in ACTIVITY:
        v = VERB[a]
        pats.append(([f"i am {v} now", f"i am {v} right now"], f"{a} now"))
        pats.append(([f"i am {v} today", f"i will {a} today"], f"{a} today"))
        pats.append(([f"i will {a} tomorrow", f"i am going to {a} tomorrow"],
                     f"{a} tomorrow"))
        pats.append(([f"are you {v} now", f"are you {v} right now"],
                     f"you {a} now"))
        pats.append(([f"are you {v} today", f"will you {a} today"],
                     f"you {a} today"))
        pats.append(([f"will you {a} tomorrow",
                      f"are you going to {a} tomorrow"], f"you {a} tomorrow"))
        for p in PERSON:
            pats.append(([f"{p} will {a} tomorrow", f"my {p} will {a} tomorrow"],
                         f"{p} {a} tomorrow"))
    for pl in TO:
        for t in WHEN:
            pats.append(([f"i am going {TO[pl]} {t}", f"i will go {TO[pl]} {t}"],
                         f"{pl} {t}"))
    pats.append((["i have class today", "we have class today"], "class today"))
    pats.append((["do you have class today", "will you have class today"],
                 "you class today"))
    pats.append((["do you have class tomorrow", "will you have class tomorrow"],
                 "you class tomorrow"))

    # TIME: the sign "time"
    pats.append((["what time is it", "what is the time",
                  "do you know what time it is"], "what time"))
    pats.append((["what time is it now", "what is the time now"],
                 "what time now"))
    for pl in ["school", "class"]:
        pats.append(([f"what time is {pl}", f"what time is your {pl}",
                      f"what time is the {pl}"], f"what time {pl}"))
        pats.append(([f"it is time for {pl}", f"time for {pl}"], f"time {pl}"))
    for a in ACTIVITY:
        pats.append(([f"it is time to {a}", f"time to {a}"], f"time {a}"))

    return pats


def build_rows(patterns, task, prefix):
    rows = []
    for inputs, output in patterns:
        for inp in inputs:
            rows.append({"task": task, "input": inp, "output": output,
                         "pattern": prefix + output})
    return rows


def split_by_phrasing(rows, holdout_per_pattern=2, min_train=2, seed=SEED):
    """
    Hold out DIFFERENT PHRASINGS of meanings the model has seen, keeping
    at least `min_train` phrasings of every pattern in training. This
    tests whether the system handles new ways of saying known things --
    the adviser's actual concern.

    A SINGLE-SIGN input is never held out: it is the only way to sign
    that meaning, so holding it out deletes the fact rather than testing
    generalisation ("water" alone once came out as "this is the water").
    """
    by_pattern = {}
    for r in rows:
        by_pattern.setdefault(r["pattern"], []).append(r)
    rng = random.Random(seed)
    train, test = [], []
    for _, group in sorted(by_pattern.items()):
        group = list(group)
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
    reader can be shown.
    """
    signs = set(ALL_SIGNS)
    problems = []
    for r in rows:
        if r["task"] == "complete":
            bad = [w for w in r["input"].split() if w not in signs]
            if bad:
                problems.append(("complete input has non-signs", r["input"], bad))
        elif r["task"] == "reduce":
            bad = [w for w in r["output"].split()
                   if w not in signs and w != QUESTION_MARKER_WORD]
            if bad:
                problems.append(("reduce output has non-signs", r["output"], bad))
    return problems


def check_vocabulary(rows):
    """Every word in every row must be in the lookup table, or the
    tokenizer turns it into <unk> and the model cannot learn it."""
    from vocab_pool import ALL_VOCAB
    known = set(ALL_VOCAB)
    missing = set()
    for r in rows:
        missing.update(w for w in (r["input"] + " " + r["output"]).split()
                       if w not in known)
    return sorted(missing)


def check_no_contradictions(rows):
    """
    The same input must never be paired with two different answers.
    Contradictory rows cannot be learned -- the model is forced to
    guess between them.
    """
    seen = {}
    clashes = []
    for r in rows:
        key = (r["task"], r["input"])
        if key in seen and seen[key] != r["output"]:
            clashes.append((r["task"], r["input"], seen[key], r["output"]))
        seen[key] = r["output"]
    return clashes


def dedupe(rows):
    seen, out = set(), []
    for r in rows:
        key = (r["task"], r["input"], r["output"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    complete_rows = build_rows(build_complete(), "complete", "complete:")
    reduce_rows = build_rows(build_reduce(), "reduce", "reduce:")
    all_rows = dedupe(complete_rows + reduce_rows)

    shape_problems = check_task_shapes(all_rows)
    if shape_problems:
        print(f"ABORTING -- {len(shape_problems)} rows have the wrong shape:\n")
        for kind, text, bad in shape_problems[:12]:
            print(f"   {kind}: {text!r}  offending: {bad}")
        raise SystemExit("A block of patterns is in the wrong builder.")

    missing = check_vocabulary(all_rows)
    if missing:
        raise SystemExit(f"ABORTING -- words not in vocab_pool.py: {missing}")

    clashes = check_no_contradictions(all_rows)
    if clashes:
        print(f"ABORTING -- {len(clashes)} inputs are taught two different answers:\n")
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

    n_complete = sum(r["task"] == "complete" for r in all_rows)
    print(f"=== dataset ({len(ALL_SIGNS)} signs) ===\n")
    print(f"Complete rows     : {n_complete}")
    print(f"Reduce rows       : {len(all_rows) - n_complete}")
    print(f"TOTAL             : {len(all_rows)}\n")
    n_pat = len({r["pattern"] for r in all_rows})
    print(f"Distinct patterns : {n_pat}")
    print(f"TRAIN rows: {len(train)}  ->  dataset_v4_train.jsonl")
    print(f"TEST rows : {len(test)}  ->  dataset_v4_test.jsonl\n")

    used_in = {w for r in all_rows if r["task"] == "complete" for w in r["input"].split()}
    used_out = {w for r in all_rows if r["task"] == "reduce" for w in r["output"].split()}
    print("Signs never in a Complete input:",
          sorted(set(ALL_SIGNS) - used_in) or "none")
    print("Signs never in a Reduce output :",
          sorted(set(ALL_SIGNS) - used_out) or "none")
    longest_in = max(len(r["input"].split()) for r in all_rows)
    longest_out = max(len(r["output"].split()) for r in all_rows)
    print(f"\nLongest input : {longest_in} words")
    print(f"Longest output: {longest_out} words")


if __name__ == "__main__":
    main()
