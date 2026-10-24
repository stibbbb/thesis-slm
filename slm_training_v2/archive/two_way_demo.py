"""
two_way_demo.py -- compact two-way conversation: only the two screens and
the memory are shown. For every pipeline box as it runs, use full_demo.py.
Both demos run the same pipeline (slm_pipeline.py).

  1 = the DEAF person signs      -> a full sentence for the hearing person
  2 = the HEARING person speaks  -> key signs for the Deaf person's screen
  r = reset the conversation memory
  q = quit
"""

from slm_pipeline import LOOKUP_TABLE, PEOPLE, SIGNS, SLMPipeline


def box(title, text):
    w = 60
    print(f"    +{'-' * w}+")
    print(f"    | {title:<{w - 2}} |")
    print(f"    | {text:<{w - 2}} |")
    print(f"    +{'-' * w}+")


def main():
    slm = SLMPipeline()

    print("=" * 70)
    print("  TWO-WAY CONVERSATION DEMO  (with memory)")
    print(f"  {len(SIGNS)} signs   |   {len(LOOKUP_TABLE)} vocabulary words   |   device: {slm.device}")
    print("=" * 70)
    print("  1  = Deaf person signs      2  = hearing person speaks")
    print("  r  = reset memory           q  = quit")
    print("=" * 70)
    print(f"\n  Signs: {', '.join(sorted(SIGNS))}")
    print(f"\n  Recoverable subjects: {', '.join(sorted(PEOPLE))}\n")

    while True:
        try:
            choice = input("[1=sign, 2=speak, r=reset, q=quit]> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice in ("q", "quit", "exit"):
            break
        if choice == "r":
            slm.context.reset()
            print("  memory cleared\n")
            continue

        if choice == "1":
            text = input("  signs> ").strip()
            if not text:
                continue
            turn = slm.deaf_to_non_deaf(text)
            title = "HEARING PERSON'S SCREEN:"
        elif choice == "2":
            text = input("  says> ").strip()
            if not text:
                continue
            turn = slm.non_deaf_to_deaf(text)
            title = "DEAF PERSON'S SCREEN:"
        else:
            print("  type 1, 2, r, or q\n")
            continue

        print()
        box(title, turn.output or "(nothing)")
        held = slm.context.person or "(nobody yet)"
        print(f"    memory: currently talking about {held}   "
              f"| {slm.context.turns} turns so far\n")

    print("Done.")


if __name__ == "__main__":
    main()
