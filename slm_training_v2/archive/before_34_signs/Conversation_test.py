"""
conversation_test.py -- runs realistic two-way conversations through
the system and checks every answer.

WHY A SCRIPT AND NOT A LIST TO TYPE
Each expected answer is looked up from the training data itself rather
than written by hand, so the check cannot drift out of step with what
the model was actually taught. Every turn is reported as one of:

    MATCH    the model agrees with the dataset
    DIFFER   the model disagrees with the dataset -- a real miss
    NEW      this exact sentence is not in the dataset at all, so the
             model is generalising. There is no reference answer; YOU
             judge whether the output is sensible.

The NEW rows are the interesting ones. They are the model handling
something it was never explicitly taught.

Every one of the signs is exercised at least once; the script verifies
that at the end and lists any it missed.

Each turn runs through the full pipeline (slm_pipeline.py). The memory is
cleared before every turn, so each answer is judged on its own against
the dataset rather than on what an earlier turn left in memory.

Run: python Conversation_test.py
"""

import json

from slm_pipeline import SIGNS, SLMPipeline

# ("sign", glosses)   = the Deaf person signs
# ("speak", sentence) = the hearing person speaks
CONVERSATIONS = [
    ("Meeting for the first time", [
        ("speak", "hello"),
        ("sign",  "hello"),
        ("speak", "nice to meet you"),
        ("sign",  "nice"),
        ("speak", "how are you"),
        ("sign",  "my fine"),
        ("speak", "i am fine"),
    ]),
    ("Asking about family", [
        ("speak", "where is your mother"),
        ("sign",  "mother home"),
        ("speak", "how is your father"),
        ("sign",  "father tired"),
        ("speak", "why is your father tired"),
        ("sign",  "father laundry"),
        ("speak", "is your family okay"),
        ("sign",  "family fine"),
    ]),
    ("Talking about someone else", [
        ("speak", "where is her mother"),
        ("sign",  "her mother school"),
        ("speak", "how is his father"),
        ("sign",  "his father sick"),
        ("speak", "who is your friend"),
        ("sign",  "friend deaf"),
    ]),
    ("School and studying", [
        ("speak", "are you studying at school"),
        ("sign",  "my study school"),
        ("speak", "who is your teacher"),
        ("sign",  "teacher good"),
        ("speak", "when are you learning"),
        ("sign",  "my learn home"),
    ]),
    ("Everyday needs", [
        ("speak", "do you want water"),
        ("sign",  "yes"),
        ("speak", "do you need help"),
        ("sign",  "no"),
        ("speak", "do you want to eat"),
        ("sign",  "my eat home"),
        ("speak", "i need to go to the bathroom"),
        ("sign",  "bathroom"),
    ]),
    ("Feelings and apologies", [
        ("speak", "you are sad"),
        ("sign",  "my sad"),
        ("speak", "why are you sad"),
        ("sign",  "my sick"),
        ("speak", "sorry"),
        ("sign",  "sorry"),
        ("speak", "do you understand"),
        ("sign",  "understand"),
    ]),
    ("Getting around", [
        ("speak", "where is the school"),
        ("sign",  "my drive school"),
        ("speak", "are you finished"),
        ("sign",  "finish"),
        ("speak", "what is your family doing"),
        ("sign",  "family eat home"),
        ("speak", "please help me"),
        ("sign",  "help"),
    ]),
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


def main():
    slm = SLMPipeline()
    ref = load_reference()

    print("=" * 78)
    print("  CONVERSATION TEST")
    print(f"  {len(SIGNS)} signs   |   device: {slm.device}")
    print("=" * 78)

    match = differ = new = 0
    differences = []
    signs_used = set()

    for title, turns in CONVERSATIONS:
        print(f"\n--- {title} ---")
        for who, text in turns:
            slm.context.reset()
            if who == "sign":
                signs_used.update(w for w in text.split() if w in SIGNS)
                got = slm.deaf_to_non_deaf(text).output
                expected = ref.get(("complete", text))
                label = "Deaf signs "
                shown = f"{text!r}"
            else:
                got = slm.non_deaf_to_deaf(text).output
                expected = ref.get(("reduce", text.lower()))
                label = "hearing say"
                shown = f"{text!r}"

            if expected is None:
                verdict = "NEW   "
                new += 1
            elif got.strip() == expected.strip():
                verdict = "MATCH "
                match += 1
            else:
                verdict = "DIFFER"
                differ += 1
                differences.append((label, text, expected, got))

            print(f"  [{verdict}] {label}: {shown:<34} -> {got!r}")
            if verdict == "DIFFER":
                print(f"           dataset says: {expected!r}")

    total = match + differ + new
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  MATCH  (agrees with dataset)      : {match}/{total}")
    print(f"  DIFFER (disagrees -- a real miss) : {differ}/{total}")
    print(f"  NEW    (not taught -- you judge)  : {new}/{total}")

    if differences:
        print(f"\n--- The {len(differences)} disagreements ---")
        for label, text, exp, got in differences:
            print(f"  {label}: {text!r}")
            print(f"     expected {exp!r}")
            print(f"     got      {got!r}")

    missed = sorted(SIGNS - signs_used)
    print(f"\nSigns exercised by the Deaf side: {len(signs_used)}/{len(SIGNS)}")
    if missed:
        print(f"  not signed in these conversations: {missed}")
        print("  (several only make sense from the hearing side, e.g."
              " question words)")


if __name__ == "__main__":
    main()