"""
generate_dataset_v4.py -- builds the training data for the 44 signs
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
  COMPLETE (Deaf -> hearing): the INPUT is constrained to the 44 signs,
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
# THE 44 CONFIRMED SIGNS, grouped by how they behave grammatically.
# ---------------------------------------------------------------- #

PERSON   = ["mother", "father", "teacher", "family", "friend"]
SOMEONE  = ["person"]       # any person: not greeted by name, never "my person"
# The camera cannot tell MOTHER from FATHER: the Deaf person's signs say
# PARENT (Complete inputs), while the hearing person still says mother,
# father or parent (Reduce inputs).
SIGNED_PERSON = ["parent", "teacher", "family", "friend"]
HEARD_PERSON = PERSON + ["parent"]
STATE    = ["fine", "good", "deaf"]
ACTIVITY = ["study", "learn", "eat", "sleep", "sign"]
MOTION   = ["go"]           # needs a place: "go school" = "i am going to school"
PLACE    = ["home", "school", "class", "here"]
THING    = ["water", "phone"]
QUESTION = ["what", "where", "why", "how"]
SOCIAL   = ["hello", "sorry", "please", "nice", "yes", "understand", "again", "thank"]
NEED     = ["help", "want"]
PRONOUN  = ["you", "my"]
TIME     = ["time", "now", "today", "tomorrow", "yesterday"]
WHEN     = ["now", "today", "tomorrow"]     # the time words that say WHEN (not past)
NEGATION = ["not"]          # put in by a rule (slm_pipeline.negate), never learned

ALL_SIGNS = (PERSON + ["parent"] + SOMEONE + STATE + ACTIVITY + MOTION + PLACE + THING
             + QUESTION + SOCIAL + NEED + PRONOUN + TIME + NEGATION)
assert sorted(ALL_SIGNS) == sorted(CONFIRMED_GESTURES), "sign lists out of sync"

PEOPLE = PERSON + SOMEONE   # everyone a sentence can be about

# How each sign reads inside a natural sentence.
VERB = {"study": "studying", "learn": "learning", "eat": "eating", "sleep": "sleeping",
        "sign": "signing"}
PAST = {"study": "studied", "learn": "learned", "eat": "ate", "sleep": "slept", "sign": "signed"}
# being AT a place, and going TO it
AT = {"home": "home", "school": "at school", "class": "in class", "here": "here"}
TO = {"home": "home", "school": "to school", "class": "to class"}
# Going somewhere TO DO something: only the activities that make sense at
# each place ("go school study", not "go class sleep").
PURPOSE = {"school": ["study", "learn", "sign", "eat"],
           "class": ["study", "learn", "sign"],
           "home": ["eat", "sleep", "study"]}
PERSON_ARTICLE = {"mother": "mother", "father": "father",
                  "teacher": "teacher", "family": "my family",
                  "friend": "my friend", "person": "the person",
                  "parent": "my parent"}
# "what are you sleeping" means nothing: SLEEP is not asked with WHAT
WHAT_ACTIVITY = [a for a in ACTIVITY if a != "sleep"]

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
    "thank": "thank you",
    # states -- subject is the signer
    "fine": "i am fine",
    "good": "i am good",
    "deaf": "i am deaf",
    # activities
    "study": "i am studying",
    "learn": "i am learning",
    "eat": "i am eating",
    "sleep": "i am going to sleep",
    "sign": "i am signing",
    "go": "i am going",
    # places
    "home": "i am home",
    "school": "i am at school",
    "class": "i am in class",
    "here": "i am here",
    # things and needs
    # WATER alone is not "i want water": WANT is a sign of its own now
    "water": "water",
    "phone": "phone",
    "help": "help me",
    "want": "i want that",
    # people -- introducing someone
    "parent": "this is my parent",
    "teacher": "this is teacher",
    "family": "this is my family",
    "friend": "this is my friend",
    "person": "that person",
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
    "yesterday": "yesterday",
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
    PERSON = SIGNED_PERSON              # the camera signs PARENT, not mother/father
    PEOPLE = PERSON + SOMEONE
    pats = []

    def add(words, sentence):
        pats.append((orderings(words), sentence))

    # ---------- one sign ----------
    for sign, sentence in SINGLE.items():
        pats.append(([sign], sentence))

    # ---------- PERSON + something ----------
    for pr in PEOPLE:
        who = PERSON_ARTICLE[pr]
        for st in STATE:
            add([pr, st], f"{who} is {st}")
        for a in ACTIVITY:
            add([pr, a], f"{who} is {VERB[a]}")
        for pl in PLACE:
            add([pr, pl], f"is {who} {AT[pl]}")
        add(["where", pr], f"where is {who}")
        add(["how", pr], f"how is {who}")
        if pr in SOMEONE:
            continue
        add(["hello", pr], f"hello {pr}")
        add(["sorry", pr], f"i am sorry {pr}")
        add(["nice", pr], f"nice to meet you {pr}")
        add(["please", pr], f"please {pr}")
        add(["thank", pr], f"thank you {pr}")
    add(["nice", "person"], "the person is nice")
    add(["help", "person"], "help the person")

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
    for a in WHAT_ACTIVITY:
        add(["what", a], f"what are you {VERB[a]}")
    for st in STATE:
        add(["why", st], f"why are you {st}")
    for t in THING:
        add(["where", t], f"where is the {t}")
    add(["where", "you"], "where are you")
    add(["how", "you"], "how are you")

    # ---------- QUESTION + PERSON + STATE / ACTIVITY ----------
    for pr in PEOPLE:
        who = PERSON_ARTICLE[pr]
        for st in STATE:
            add(["why", pr, st], f"why is {who} {st}")
        for a in ACTIVITY:
            add(["where", pr, a], f"where is {who} {VERB[a]}")
            add(["why", pr, a], f"why is {who} {VERB[a]}")
        for a in WHAT_ACTIVITY:
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
    add(["you", "water"], "do you have water")
    add(["please", "water"], "water please")
    add(["you", "help"], "can i help you")
    add(["my", "help"], "i need help")
    add(["please", "help"], "please help me")
    for a in ACTIVITY:
        add(["help", a], f"help me {a}")
    for pl in TO:
        add(["help", pl], f"help me go {TO[pl]}")

    # ---------- phone ----------
    add(["my", "phone"], "this is my phone")
    add(["where", "my", "phone"], "where is my phone")
    add(["you", "phone"], "do you have your phone")
    add(["please", "phone"], "phone please")

    # ---------- want ----------
    for t in THING:
        mine, yours = ("my ", "your ") if t == "phone" else ("", "")
        add(["want", t], f"i want {mine}{t}")
        add(["you", "want", t], f"do you want {yours}{t}")
    for a in ACTIVITY:
        add(["want", a], f"i want to {a}")
        add(["you", "want", a], f"do you want to {a}")
    for pl in TO:
        add(["want", pl], f"i want to go {TO[pl]}")
        add(["want", "go", pl], f"i want to go {TO[pl]}")
        add(["you", "want", "go", pl], f"do you want to go {TO[pl]}")
    add(["want", "help"], "i want help")
    add(["what", "want"], "what do you want")
    add(["what", "you", "want"], "what do you want")
    for pr in PERSON:
        who = PERSON_ARTICLE[pr]
        add(["want", pr], f"i want {who}")
        for pl in PLACE:
            add(["want", pr, pl], f"i want {who} to be {AT[pl]}")
            for t in WHEN:
                add(["want", pr, pl, t], f"i want {who} to be {AT[pl]} {t}")

    # ---------- want + activity + place / time ----------
    for a in ACTIVITY:
        for pl in PLACE:
            loc = "at home" if pl == "home" else AT[pl]
            add(["want", a, pl], f"i want to {a} {loc}")
            add(["you", "want", a, pl], f"do you want to {a} {loc}")
        for t in WHEN:
            add(["want", a, t], f"i want to {a} {t}")
            add(["you", "want", a, t], f"do you want to {a} {t}")

    # ---------- a person who WANTS something: "teacher wants to learn" ----------
    for pr in PEOPLE:
        who = PERSON_ARTICLE[pr]
        for a in ACTIVITY:
            add([pr, "want", a], f"{who} wants to {a}")
            for pl in PLACE:
                loc = "at home" if pl == "home" else AT[pl]
                add([pr, "want", a, pl], f"{who} wants to {a} {loc}")
        for pl in TO:
            add([pr, "want", "go", pl], f"{who} wants to go {TO[pl]}")
        add([pr, "want", "water"], f"{who} wants water")
        add([pr, "want", "phone"], f"{who} wants the phone")

    # ---------- going somewhere TO DO something ----------
    # "want go school study" -> "i want to go to school to study"
    # "go school study tomorrow" -> "i will go to school to study tomorrow"
    def going(subject, t):
        """The 'go' verb for a subject at a time (None = right now)."""
        if subject == "i":
            return {"tomorrow": "i will go", "yesterday": "i went"}.get(t, "i am going")
        if subject == "you":
            return {"tomorrow": "will you go", "yesterday": "did you go"}.get(t, "are you going")
        return {"tomorrow": f"{subject} will go", "yesterday": f"{subject} went"}.get(
            t, f"{subject} is going")

    subjects = [([], "i"), (["you"], "you")] + [([pr], PERSON_ARTICLE[pr]) for pr in PEOPLE]
    for pl, acts in PURPOSE.items():
        for a in acts:
            goal = f"{TO[pl]} to {a}"
            for signed, subject in subjects:
                wants = {"i": "i want", "you": "do you want"}.get(subject, f"{subject} wants")
                add(signed + ["go", pl, a], f"{going(subject, None)} {goal}")
                add(signed + ["want", "go", pl, a], f"{wants} to go {goal}")
                for t in WHEN + ["yesterday"]:
                    add(signed + ["go", pl, a, t], f"{going(subject, t)} {goal} {t}")
                for t in WHEN:
                    add(signed + ["want", "go", pl, a, t], f"{wants} to go {goal} {t}")

    # ---------- thank you ----------
    add(["thank", "you"], "thank you")
    add(["thank", "help"], "thank you for your help")
    add(["thank", "water"], "thank you for the water")
    add(["thank", "again"], "thank you again")
    add(["yes", "thank"], "yes thank you")

    # ---------- ACTIVITY + PLACE, and with a person ----------
    for a in ACTIVITY:
        for pl in PLACE:
            loc = "at home" if pl == "home" else AT[pl]
            add([a, pl], f"i am {VERB[a]} {loc}")
            for pr in PEOPLE:
                add([pr, a, pl], f"{PERSON_ARTICLE[pr]} is {VERB[a]} {loc}")

    # ---------- go: always to a place, or to do something ----------
    for pl in TO:
        add(["go", pl], f"i am going {TO[pl]}")
        add(["you", "go", pl], f"are you going {TO[pl]}")
        for pr in PEOPLE:
            add([pr, "go", pl], f"{PERSON_ARTICLE[pr]} is going {TO[pl]}")
    for a in ACTIVITY:
        add(["go", a], f"i am going to {a}")
    add(["where", "go"], "where are you going")
    add(["where", "you", "go"], "where are you going")
    for pr in PEOPLE:
        add(["where", pr, "go"], f"where is {PERSON_ARTICLE[pr]} going")
    # where + go + a time: "where will you go tomorrow"
    for t in WHEN:
        you_go = "where will you go" if t == "tomorrow" else "where are you going"
        add(["where", "go", t], f"{you_go} {t}")
        add(["where", "you", "go", t], f"{you_go} {t}")
        for pr in PEOPLE:
            who = PERSON_ARTICLE[pr]
            they_go = f"where will {who} go" if t == "tomorrow" else f"where is {who} going"
            add(["where", pr, "go", t], f"{they_go} {t}")
    add(["where", "go", "yesterday"], "where did you go yesterday")
    add(["where", "you", "go", "yesterday"], "where did you go yesterday")
    for pr in PEOPLE:
        add(["where", pr, "go", "yesterday"], f"where did {PERSON_ARTICLE[pr]} go yesterday")

    # ---------- where + an activity, and when: "where did you study yesterday" ----------
    for a in ACTIVITY:
        add(["where", "you", a], f"where are you {VERB[a]}")
        for t in WHEN:
            you_do = f"where will you {a}" if t == "tomorrow" else f"where are you {VERB[a]}"
            add(["where", a, t], f"{you_do} {t}")
            add(["where", "you", a, t], f"{you_do} {t}")
            for pr in PEOPLE:
                who = PERSON_ARTICLE[pr]
                they_do = f"where will {who} {a}" if t == "tomorrow" else f"where is {who} {VERB[a]}"
                add(["where", pr, a, t], f"{they_do} {t}")
        add(["where", a, "yesterday"], f"where did you {a} yesterday")
        add(["where", "you", a, "yesterday"], f"where did you {a} yesterday")
        for pr in PEOPLE:
            add(["where", pr, a, "yesterday"], f"where did {PERSON_ARTICLE[pr]} {a} yesterday")
    add(["why", "go"], "why are you going")
    add(["time", "go"], "it is time to go")
    add(["what", "time", "go"], "what time do we go")
    add(["please", "go"], "please go")

    # ---------- TIME: now / today / tomorrow ----------
    for t in WHEN:
        for a in ACTIVITY:
            add([a, t], when_phrase("i", a, t))
            add(["you", a, t], when_phrase("you", a, t))
            for pr in PEOPLE:
                add([pr, a, t], when_phrase(PERSON_ARTICLE[pr], a, t))
        for pl in TO:
            go = "i will go" if t == "tomorrow" else "i am going"
            add([pl, t], f"{go} {TO[pl]} {t}")
            add(["go", pl, t], f"{go} {TO[pl]} {t}")
            you_go = "will you go" if t == "tomorrow" else "are you going"
            add(["you", pl, t], f"{you_go} {TO[pl]} {t}")
            add(["you", "go", pl, t], f"{you_go} {TO[pl]} {t}")
        for st in STATE:
            add([st, t], f"i am {st} {t}")
        add(["how", "you", t], f"how are you {t}")
    add(["where", "you", "now"], "where are you now")
    add(["understand", "now"], "i understand now")

    # ---------- TIME: yesterday, in the past tense ----------
    for a in ACTIVITY:
        add([a, "yesterday"], f"i {PAST[a]} yesterday")
        add(["you", a, "yesterday"], f"did you {a} yesterday")
        for pr in PEOPLE:
            add([pr, a, "yesterday"], f"{PERSON_ARTICLE[pr]} {PAST[a]} yesterday")
    for pl in TO:
        add([pl, "yesterday"], f"i went {TO[pl]} yesterday")
        add(["go", pl, "yesterday"], f"i went {TO[pl]} yesterday")
        add(["you", pl, "yesterday"], f"did you go {TO[pl]} yesterday")
        add(["you", "go", pl, "yesterday"], f"did you go {TO[pl]} yesterday")
    for st in ["fine", "good"]:
        add([st, "yesterday"], f"i was {st} yesterday")
    add(["how", "you", "yesterday"], "how were you yesterday")
    add(["where", "you", "yesterday"], "where were you yesterday")

    # ---------- PERSON + PLACE + TIME: where someone is, and when ----------
    for pr in PEOPLE:
        who = PERSON_ARTICLE[pr]
        for pl in PLACE:
            add([pr, pl, "now"], f"{who} is {AT[pl]} now")
            add([pr, pl, "today"], f"{who} is {AT[pl]} today")
            add([pr, pl, "tomorrow"], f"{who} will be {AT[pl]} tomorrow")
            add([pr, pl, "yesterday"], f"{who} was {AT[pl]} yesterday")

    # ---------- TIME: the sign "time" ----------
    add(["what", "time"], "what time is it")
    add(["what", "time", "now"], "what time is it now")
    for pl in ["school", "class"]:
        add(["what", "time", pl], f"what time is {pl}")
        add(["time", pl], f"it is time for {pl}")
    for a in ACTIVITY:
        add(["what", "time", a], f"what time do we {a}")
        add(["what", "time", "you", a], f"what time do you {a}")
        for pr in PEOPLE:
            add(["what", "time", pr, a], f"what time does {PERSON_ARTICLE[pr]} {a}")
        add(["time", a], f"it is time to {a}")
    add(["what", "time", "you", "go"], "what time do you go")
    add(["time", "home"], "it is time to go home")

    # ---------- again ----------
    add(["please", "again"], "please say it again")
    add(["sorry", "again"], "sorry can you say it again")
    for a in ACTIVITY:
        add([a, "again"], f"i will {a} again")

    # ---------- SOCIAL pairs that occur together ----------
    add(["yes", "please"], "yes please")
    add(["yes", "understand"], "yes i understand")

    # ---------- learning sign language: LEARN / STUDY + SIGN ----------
    for a in ["learn", "study"]:
        add([a, "sign"], f"i am {VERB[a]} sign language")
        add(["you", a, "sign"], f"are you {VERB[a]} sign language")
        add(["want", a, "sign"], f"i want to {a} sign language")
        add(["you", "want", a, "sign"], f"do you want to {a} sign language")
        for t in WHEN:
            doing = f"i will {a}" if t == "tomorrow" else f"i am {VERB[a]}"
            add([a, "sign", t], f"{doing} sign language {t}")
        add([a, "sign", "yesterday"], f"i {PAST[a]} sign language yesterday")
        for pl in ["school", "class"]:
            add([a, "sign", pl], f"i am {VERB[a]} sign language {AT[pl]}")
        for pr in PEOPLE:
            who = PERSON_ARTICLE[pr]
            add([pr, a, "sign"], f"{who} is {VERB[a]} sign language")
            add([pr, "want", a, "sign"], f"{who} wants to {a} sign language")

    # ---------- YES + the answer: "yes study tomorrow" -> "yes i will study tomorrow" ----------
    for st in STATE:
        add(["yes", st], f"yes i am {st}")
    for a in ACTIVITY:
        add(["yes", a], f"yes i am {VERB[a]}")
        add(["yes", "want", a], f"yes i want to {a}")
        for t in WHEN:
            add(["yes", a, t], f"yes {when_phrase('i', a, t)}")
        add(["yes", a, "yesterday"], f"yes i {PAST[a]} yesterday")
    for t in THING:
        add(["yes", "want", t], f"yes i want {'my ' if t == 'phone' else ''}{t}")
    add(["yes", "my", "phone"], "yes this is my phone")
    for pl in PLACE:
        add(["yes", pl], f"yes i am {AT[pl]}")
    for pl in TO:
        add(["yes", "go", pl], f"yes i am going {TO[pl]}")
        for t in WHEN:
            go = "i will go" if t == "tomorrow" else "i am going"
            add(["yes", "go", pl, t], f"yes {go} {TO[pl]} {t}")
        add(["yes", "go", pl, "yesterday"], f"yes i went {TO[pl]} yesterday")
    for pr in PEOPLE:
        who = PERSON_ARTICLE[pr]
        for pl in PLACE:
            add(["yes", pr, pl], f"yes {who} is {AT[pl]}")
        for st in STATE:
            add(["yes", pr, st], f"yes {who} is {st}")

    # ---------- offering help: "my help you learn sign" ----------
    add(["my", "help", "you"], "i will help you")
    for a in ACTIVITY:
        add(["help", "you", a], f"i will help you {a}")
        add(["my", "help", "you", a], f"i will help you {a}")
    for a in ["learn", "study"]:
        add(["help", "you", a, "sign"], f"i will help you {a} sign language")
        add(["my", "help", "you", a, "sign"], f"i will help you {a} sign language")

    return pats


def build_reduce():
    """Natural spoken English -> gloss keywords for the Deaf reader.

    The hearing person speaks freely, so several phrasings are given
    for each meaning -- that variety is what forces the model to learn
    the underlying keep/drop rule instead of memorizing sentences.
    Words with no sign ("his", "who", "when", "need") appear in the
    phrasings so the model learns to DROP them.
    """
    PERSON = HEARD_PERSON               # the hearing person says mother, father, parent
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
    pats.append((["where are you", "tell me where you are"], "where you"))
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

    # water / help  ("need" has no sign, so it is dropped; WANT is a sign)
    pats.append((["do you need water", "do you need some water"], "you water"))
    pats.append((["do you want water", "do you want some water"], "you want water"))
    pats.append((["i need water", "i need some water"], "water"))
    pats.append((["i want water", "i want some water"], "want water"))
    # "you" is kept when the help is for the listener, as in "you water"
    pats.append((["i need help", "i need some help"], "help"))
    pats.append((["i want help", "i want some help"], "want help"))
    pats.append((["do you need help", "can i help you",
                  "i can help you", "i will help you"], "help you"))
    pats.append((["do you want help", "do you want some help"], "you want help"))
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
        pats.append(([f"i will {a} again"], f"{a} again"))
        pats.append(([f"i want to {a} again", f"i really want to {a} again"],
                     f"want {a} again"))

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
                         f"go {pl} {t}"))
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

    # GO is a sign when it moves someone to a place. "going to" + an
    # activity is only the future ("i am going to study tomorrow" ->
    # "study tomorrow" above), so the picker learns to drop it there.
    for pl in TO:
        pats.append(([f"i am going {TO[pl]}", f"i will go {TO[pl]}", f"i go {TO[pl]}"],
                     f"go {pl}"))
        pats.append(([f"are you going {TO[pl]}", f"will you go {TO[pl]}"], f"you go {pl}"))
        pats.append(([f"do you want to go {TO[pl]}", f"you want to go {TO[pl]}"],
                     f"you want go {pl}"))
        for p in PERSON:
            pats.append(([f"{p} is going {TO[pl]}", f"my {p} is going {TO[pl]}",
                          f"{p} will go {TO[pl]}"], f"{p} go {pl}"))
    pats.append((["where are you going", "where will you go", "where do you go"],
                 "where you go"))
    for t in WHEN:
        pats.append(([f"where are you going {t}", f"where will you go {t}"],
                     f"where you go {t}"))
    pats.append((["it is time to go", "time to go"], "time go"))
    pats.append((["you can go"], "go"))
    pats.append((["please go", "go please"], "please go"))

    # WANT is a sign: it is kept wherever it is said
    for a in ACTIVITY:
        pats.append(([f"i want to {a}", f"i really want to {a}"], f"want {a}"))
        pats.append(([f"do you want to {a}", f"you want to {a}"], f"you want {a}"))
    pats.append((["what do you want", "tell me what you want"], "what you want"))
    pats.append((["i want my phone", "i want the phone"], "want phone"))
    for p in PERSON:
        pats.append(([f"i want {p}", f"i want my {p}"], f"want {p}"))
        pats.append(([f"i want {p} to be here", f"i want my {p} to be here"],
                     f"want {p} here"))
        pats.append(([f"i want {p} to be here now", f"i want my {p} to be here now"],
                     f"want {p} here now"))

    # YESTERDAY: past-tense speech keeps its signs
    for a in ACTIVITY:
        pats.append(([f"i {PAST[a]} yesterday", f"i was {VERB[a]} yesterday"],
                     f"{a} yesterday"))
        pats.append(([f"did you {a} yesterday", f"were you {VERB[a]} yesterday"],
                     f"you {a} yesterday"))
        for p in PERSON:
            pats.append(([f"{p} {PAST[a]} yesterday", f"my {p} {PAST[a]} yesterday"],
                         f"{p} {a} yesterday"))
    for pl in TO:
        pats.append(([f"i went {TO[pl]} yesterday", f"we went {TO[pl]} yesterday"],
                     f"go {pl} yesterday"))
        pats.append(([f"did you go {TO[pl]} yesterday", f"you went {TO[pl]} yesterday"],
                     f"you go {pl} yesterday"))
    for pl in PLACE:
        at = AT[pl] if pl != "home" else "at home"
        pats.append(([f"i was {at} yesterday", f"we were {at} yesterday"],
                     f"{pl} yesterday"))
    for st in ["fine", "good"]:
        pats.append(([f"i was {st} yesterday", f"i was very {st} yesterday"],
                     f"{st} yesterday"))
    pats.append((["where were you yesterday"], "where you yesterday"))
    pats.append((["where did you go yesterday"], "where you go yesterday"))
    # where + an activity, and when
    for a in ACTIVITY:
        v = VERB[a]
        pats.append(([f"where did you {a} yesterday", f"where were you {v} yesterday"],
                     f"where you {a} yesterday"))
        pats.append(([f"where will you {a} tomorrow", f"where are you {v} tomorrow"],
                     f"where you {a} tomorrow"))
        pats.append(([f"where are you {v} today", f"where will you {a} today"],
                     f"where you {a} today"))
    pats.append((["how were you yesterday", "how were you doing yesterday"],
                 "how you yesterday"))

    # PERSON: someone not named ("people" becomes "person" in normalization)
    for s in STATE:
        pats.append(([f"the person is {s}", f"that person is {s}",
                      f"this person is {s}"], f"person {s}"))
    for a in ACTIVITY:
        pats.append(([f"the person is {VERB[a]}", f"that person is {VERB[a]}"],
                     f"person {a}"))
    for pl in PLACE:
        at = AT[pl] if pl != "home" else "at home"
        pats.append(([f"the person is {at}", f"that person is {at}"], f"person {pl}"))
    pats.append((["where is the person", "where is that person"], "where person"))
    pats.append((["how is the person", "how is that person"], "how person"))

    # THANK YOU (the sign is glossed "thank")
    pats.append((["thank you", "thank you so much", "thank you very much"], "thank"))
    pats.append((["thank you for your help", "thank you for the help",
                  "thank you so much for your help"], "thank help"))
    pats.append((["thank you for the water", "thank you for some water"],
                 "thank water"))
    for p in PERSON:
        pats.append(([f"thank you {p}", f"thank you so much {p}"], f"thank {p}"))
    pats.append((["i am fine thank you", "i am okay thank you", "fine thank you",
                  "okay thank you"], "fine thank"))

    # "okay" said as a statement is the sign FINE ("are you okay" asks HOW)
    pats.append((["okay", "it is okay", "that is okay", "it is fine", "that is fine",
                  "i am okay", "it is alright"], "fine"))

    # offering help ("i will help you" alone is "help you", above)
    for a in ACTIVITY:
        pats.append(([f"i will help you {a}", f"i can help you {a}"], f"help you {a}"))
    for a in ["learn", "study"]:
        pats.append(([f"i will help you {a} sign language",
                      f"i can help you {a} sign language"], f"help you {a} sign"))

    # PHONE
    pats.append((["where is my phone", "where is your phone", "where is the phone"],
                 "where phone"))
    pats.append((["do you have your phone", "is this your phone"], "you phone"))
    pats.append((["this is my phone", "that is my phone"], "my phone"))

    # learning sign language: LEARN / STUDY + SIGN ("language" has no sign)
    for a in ["learn", "study"]:
        v = VERB[a]
        pats.append(([f"i am {v} sign language", f"i am still {v} sign language"],
                     f"{a} sign"))
        pats.append(([f"are you {v} sign language", f"you are {v} sign language"],
                     f"you {a} sign"))
        pats.append(([f"i want to {a} sign language", f"i really want to {a} sign language"],
                     f"want {a} sign"))
        pats.append(([f"do you want to {a} sign language", f"you want to {a} sign language"],
                     f"you want {a} sign"))
        for p in PERSON:
            pats.append(([f"{p} is {v} sign language", f"my {p} is {v} sign language"],
                         f"{p} {a} sign"))
            pats.append(([f"{p} wants to {a} sign language",
                          f"my {p} wants to {a} sign language"], f"{p} want {a} sign"))
    pats.append((["i am learning to sign", "i am still learning to sign"], "learn sign"))

    # going somewhere TO DO something (spoken order is kept)
    for pl, acts in PURPOSE.items():
        for a in acts:
            goal = f"{TO[pl]} to {a}"
            pats.append(([f"i want to go {goal}", f"i really want to go {goal}"],
                         f"want go {pl} {a}"))
            pats.append(([f"do you want to go {goal}", f"you want to go {goal}"],
                         f"you want go {pl} {a}"))
            pats.append(([f"i am going {goal}", f"i will go {goal}"], f"go {pl} {a}"))
            pats.append(([f"are you going {goal}", f"will you go {goal}"],
                         f"you go {pl} {a}"))
            for p in PERSON:
                pats.append(([f"{p} is going {goal}", f"my {p} is going {goal}"],
                             f"{p} go {pl} {a}"))
                pats.append(([f"{p} wants to go {goal}", f"my {p} wants to go {goal}"],
                             f"{p} want go {pl} {a}"))

    # a person who WANTS something
    for p in PERSON:
        for a in ACTIVITY:
            pats.append(([f"{p} wants to {a}", f"my {p} wants to {a}"], f"{p} want {a}"))

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
