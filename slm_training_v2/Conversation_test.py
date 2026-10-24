"""
Conversation_test.py -- the school test, built around the seven contexts
from the deaf students' interviews: one scripted dialogue per context
(help and repair, school, whereabouts, greeting, courtesy, home and
family, feelings).

Every turn is checked in two ways:

  TOPIC    the dialogue runs with the memory kept, like a real
           conversation; the active topic after the turn must be the
           dialogue's context (the feelings dialogue starts neutral,
           because "how are you" has no trigger sign).

  OUTPUT   the same turn is run again with an empty memory and compared
           with the training data:
             MATCH    the model agrees with the dataset
             DIFFER   the model disagrees with the dataset -- a real miss
             NEW      the sentence is not in the dataset; you judge it
             RULE     decided by a rule, not the model (the repair rule)

Together the dialogues use every one of the signs; the script checks that
at the end.

A last conversation mixes contexts on purpose (the adviser's leak/switch
rule): words from another context only leak in, and the topic changes
only when another context covers at least half of a turn's context words.

Run: python Conversation_test.py
"""

import copy
import json

from slm_pipeline import REPAIR_SENTENCE, SIGNS, SLMPipeline

# (title, context the dialogue should be in, [(who, text)])
#   who = "speak": the hearing person;  "sign": the Deaf person
DIALOGUES = [
    ("Help and repair", "help and repair", [
        ("speak", "do you understand"),
        ("sign",  "not understand"),
        ("speak", "can you say that again"),
        ("sign",  "please again"),
        ("sign",  "please help"),
    ]),
    ("School and learning", "school and learning", [
        ("speak", "are you studying at school"),
        ("sign",  "teacher study tomorrow"),
        ("speak", "what time is class"),
        ("sign",  "what time class"),
        ("sign",  "why study"),
        ("sign",  "my learn school"),
    ]),
    ("Whereabouts and plans", "whereabouts and plans", [
        ("speak", "where are you now"),
        ("sign",  "school now"),
        ("sign",  "how you today"),
        ("sign",  "you here"),
        ("speak", "where is your mother"),
        ("sign",  "mother home"),
        ("sign",  "where water"),
    ]),
    ("Greeting", "greeting", [
        ("speak", "hello"),
        ("sign",  "hello nice friend"),
        ("speak", "nice to meet you"),
        ("sign",  "my deaf"),
    ]),
    ("Courtesy", "courtesy", [
        ("speak", "sorry"),
        ("sign",  "sorry"),
        ("speak", "please"),
        ("sign",  "yes please"),
    ]),
    ("Home and family", "home and family", [
        ("speak", "is your family at home"),
        ("sign",  "family home"),
        ("speak", "it is time to eat"),
        ("sign",  "eat now"),
    ]),
    ("Feelings", "feelings", [
        ("speak", "how are you"),
        ("sign",  "fine"),
        ("speak", "you are good"),
        ("sign",  "father good"),
    ]),
]
# turns whose topic is expected to differ from the dialogue's context
EXPECTED_TOPIC = {("Feelings", 0): None}

# (who, text, topic expected after the turn), memory kept throughout
MIXED = [
    ("speak", "are you studying at school", "school and learning"),
    ("sign",  "teacher study tomorrow",     "school and learning"),  # tomorrow leaks in
    ("sign",  "teacher class mother",       "school and learning"),  # mother leaks in
    ("sign",  "teacher family eat home",    "home and family"),      # 3 of 4: switch
    ("speak", "is your father at home",     "home and family"),
    ("sign",  "not understand",             "help and repair"),      # repair rule
    ("sign",  "please again",               "help and repair"),
]


def load_reference():
    """Expected answers, taken from the dataset the model was built on."""
    ref = {}
    for split in ("train", "test"):
        try:
            with open(f"dataset_v4_{split}.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    ref[(r["task"], r["input"])] = r["output"]
        except FileNotFoundError:
            print(f"[warn] dataset_v4_{split}.jsonl not found")
    return ref


def run_turn(slm, who, text):
    return slm.deaf_to_non_deaf(text) if who == "sign" else slm.non_deaf_to_deaf(text)


def main():
    slm = SLMPipeline()
    ref = load_reference()

    print("=" * 86)
    print("  CONVERSATION TEST  ·  seven contexts from the interviews")
    print(f"  {len(SIGNS)} signs   |   device: {slm.device}")
    print("=" * 86)

    counts = {"MATCH": 0, "DIFFER": 0, "NEW": 0, "RULE": 0}
    topic_ok = topic_total = 0
    differences, topic_misses, signs_used = [], [], set()

    for title, context, turns in DIALOGUES:
        print(f"\n--- {title} ---")
        slm.context.reset()
        for i, (who, text) in enumerate(turns):
            # OUTPUT check: the same turn with an empty memory, against the dataset
            kept = copy.deepcopy(slm.context)
            slm.context.reset()
            fresh = run_turn(slm, who, text).output
            slm.context = kept

            # TOPIC check: the real conversation, memory kept
            got = run_turn(slm, who, text).output
            want_topic = EXPECTED_TOPIC.get((title, i), context)
            topic_total += 1
            if slm.context.topic == want_topic:
                topic_ok += 1
                mark = "ok"
            else:
                mark = "MISS"
                topic_misses.append((title, text, want_topic, slm.context.topic))

            if who == "sign":
                signs_used.update(w for w in text.split() if w in SIGNS)
                expected = ref.get(("complete", text))
            else:
                expected = ref.get(("reduce", text.lower()))
            if fresh.rstrip("?.") == REPAIR_SENTENCE:
                verdict = "RULE"
            elif expected is None:
                verdict = "NEW"
            elif fresh.rstrip("?.").strip() == expected.strip():   # the dataset has no punctuation
                verdict = "MATCH"
            else:
                verdict = "DIFFER"
                differences.append((who, text, expected, fresh))
            counts[verdict] += 1

            label = "Deaf signs " if who == "sign" else "hearing say"
            print(f"  [{verdict:6}] {label}: {text!r:30} -> {fresh!r}")
            changed = f"   in the conversation: {got!r}" if got != fresh else ""
            print(f"           topic: {slm.context.topic or 'neutral'} ({mark}){changed}")

    print("\n--- Mixed contexts: leak or switch ---")
    slm.context.reset()
    mixed_ok = 0
    for who, text, want in MIXED:
        turn = run_turn(slm, who, text)
        note = dict(turn.steps).get("Active Context", "").split("; ")[-1]
        mark = "ok" if slm.context.topic == want else "MISS"
        mixed_ok += mark == "ok"
        if mark == "MISS":
            topic_misses.append(("Mixed contexts", text, want, slm.context.topic))
        label = "Deaf signs " if who == "sign" else "hearing say"
        print(f"  [{mark:4}] {label}: {text!r:30} -> {turn.output!r}")
        print(f"         {note}")

    total = sum(counts.values())
    print("\n" + "=" * 86)
    print("SUMMARY")
    print("=" * 86)
    print(f"  TOPIC correct                      : {topic_ok}/{topic_total}")
    print(f"  Mixed contexts (leak or switch)    : {mixed_ok}/{len(MIXED)}")
    print(f"  MATCH  (agrees with dataset)       : {counts['MATCH']}/{total}")
    print(f"  DIFFER (disagrees -- a real miss)  : {counts['DIFFER']}/{total}")
    print(f"  NEW    (not taught -- you judge)   : {counts['NEW']}/{total}")
    print(f"  RULE   (repair rule)               : {counts['RULE']}/{total}")

    if topic_misses:
        print(f"\n--- The {len(topic_misses)} topic misses ---")
        for title, text, want, got in topic_misses:
            print(f"  {title}: {text!r}  expected {want or 'neutral'}, got {got or 'neutral'}")
    if differences:
        print(f"\n--- The {len(differences)} disagreements ---")
        for who, text, exp, got in differences:
            print(f"  {who}: {text!r}")
            print(f"     expected {exp!r}")
            print(f"     got      {got!r}")

    missed = sorted(SIGNS - signs_used)
    print(f"\nSigns exercised by the Deaf side: {len(signs_used)}/{len(SIGNS)}")
    if missed:
        print(f"  not signed in these dialogues: {missed}")


if __name__ == "__main__":
    main()
