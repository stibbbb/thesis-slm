"""
full_demo.py -- THE live demo: test both directions with your own words,
showing every box of the adviser's pipeline as it runs (the pipeline
itself is in slm_pipeline.py).

  HEARING -> DEAF   Voice Recording -> Speech to text -> Normalization
  (ND2D)            -> Reduction -> Tokenize -> Embedding -> Contextualize
                    -> Space Restriction -> Active Context
                    -> Transcribed Words -> Display
  DEAF -> HEARING   Transcribed Words -> Tokenize -> Embedding
  (D2ND)            -> Active Context -> Sentence Completion -> Display

  1 = the Deaf person signs (type the signs)            D2ND
  2 = the hearing person speaks (type the sentence)     ND2D
  3 = the hearing person speaks into the microphone     ND2D (needs Vosk, see speech_input.py)
  w = show the words the SLM knows
  s = show / hide the pipeline steps
  r = reset the conversation memory
  q = quit

The words can go on the same line: "1 drive mother where" or
"2 where is your mom".

Run: python full_demo.py
"""

import speech_input
from slm_pipeline import LOOKUP_TABLE, PEOPLE, SIGNS, SLMPipeline, format_turn

HELP = """  1 = Deaf person signs (D2ND)       2 = hearing person types (ND2D)
  3 = hearing person speaks (ND2D)   w = known words   s = steps on/off
  r = reset memory                   q = quit
  Tip: put the words on the same line, e.g.  1 drive mother where"""


def screen(title, text):
    w = 62
    print(f"\n    +{'-' * w}+")
    print(f"    | {title:<{w - 2}} |")
    print(f"    | {text:<{w - 2}} |")
    print(f"    +{'-' * w}+\n")


def show_words():
    print(f"\n  Signs the Deaf person can sign ({len(SIGNS)}):")
    print("    " + ", ".join(sorted(SIGNS)))
    others = sorted(LOOKUP_TABLE - SIGNS)
    print(f"\n  Other words the hearing person can say ({len(others)}):")
    print("    " + ", ".join(others))
    print(f"\n  People the memory can fill in: {', '.join(sorted(PEOPLE))}\n")


def ask(label, text):
    """The words typed after the option, or ask for them."""
    return text or input(f"  {label}> ").strip()


def main():
    slm = SLMPipeline()
    mic = speech_input.is_available()
    show_steps = True

    print("=" * 72)
    print("  SLM LIVE DEMO  (adviser's pipeline, both directions)")
    print("=" * 72)
    print(f"  {len(SIGNS)} signs | {len(LOOKUP_TABLE)} words in the lookup table | device: {slm.device}")
    print(f"  microphone + speech-to-text: {'ready' if mic else 'not installed (type instead)'}\n")
    print(HELP + "\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        choice, _, text = line.partition(" ")
        choice, text = choice.lower(), text.strip()

        if choice in ("q", "quit", "exit"):
            break
        if choice in ("h", "help", "?"):
            print(HELP + "\n")
            continue
        if choice == "w":
            show_words()
            continue
        if choice == "s":
            show_steps = not show_steps
            print(f"  pipeline steps {'shown' if show_steps else 'hidden'}\n")
            continue
        if choice == "r":
            slm.context.reset()
            print("  memory cleared\n")
            continue

        if choice == "1":
            text = ask("signs", text)
            if not text:
                continue
            turn = slm.deaf_to_non_deaf(text)
            reader = "HEARING PERSON'S SCREEN"
        elif choice == "2":
            text = ask("says", text)
            if not text:
                continue
            turn = slm.non_deaf_to_deaf(text)
            reader = "DEAF PERSON'S SCREEN"
        elif choice == "3":
            if not mic:
                print("  speech-to-text is not installed yet (see speech_input.py); use 2 to type\n")
                continue
            input(f"  press Enter, then speak for {speech_input.RECORD_SECONDS:g} seconds...")
            print("  recording...")
            try:
                turn = slm.non_deaf_to_deaf()
            except speech_input.SpeechUnavailable as e:
                print(f"  {e}\n")
                continue
            reader = "DEAF PERSON'S SCREEN"
        else:
            print("  type one of the options (h for help)\n")
            continue

        if show_steps:
            print()
            print("\n".join(format_turn(turn)))
        screen(reader, turn.output or "(nothing)")
        held = slm.context.person or "(nobody yet)"
        print(f"    memory: {held}   |   {slm.context.turns} turns\n")

    print("Done.")


if __name__ == "__main__":
    main()
