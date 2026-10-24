from conversation_session import ConversationSession
from onnx_inference import map_glosses

SCREEN_WIDTH = 58


def show_screen(owner, text):
    print()
    print("  " + "+" + "-" * SCREEN_WIDTH + "+")
    label = f" {owner}'S SCREEN shows: "
    print("  |" + label.ljust(SCREEN_WIDTH) + "|")
    content = f'  "{text}"' if text else "  (nothing recognizable)"
    print("  |" + content.ljust(SCREEN_WIDTH) + "|")
    print("  " + "+" + "-" * SCREEN_WIDTH + "+")
    print()


def do_sign_turn(session, glosses):
    print(f"\n  Deaf signs: {glosses!r}")
    mapped = map_glosses(glosses)
    if mapped != glosses.lower().strip():
        print(f"    [pipeline] gloss mapping : {mapped!r}")

    sentence = session.sign_to_sentence(glosses)

    if session.last_recovery:
        print(f"    [context]  no subject in this turn -- recovered "
              f"{session.last_recovery!r} from recent conversation")

    print(f"    [pipeline] final sentence: {sentence!r}")
    show_screen("HEARING PERSON", sentence)


def do_speech_turn(session, sentence):
    print(f"\n  Hearing person speaks: {sentence!r}")
    glosses = session.speech_to_signs(sentence)
    print(f"    [pipeline] reduced to glosses: {glosses!r}")

    if session.last_confidence and session.last_confidence["low_confidence"]:
        pct = session.last_confidence["oov_density"]
        words = session.last_confidence["oov_words"]
        print(f"    [WARNING] low confidence ({pct:.0%} unfamiliar words: {words}) "
              f"-- this result may not reflect what was actually said")

    show_screen("DEAF PERSON", glosses)


def run_demo(session):
    print("\n--- Scripted demo conversation ---")
    print("(A dropped subject gets recovered mid-conversation, and it")
    print(" carries across BOTH directions. The last turn also shows")
    print(" the low-confidence warning firing on a high-OOV sentence.)")
    do_sign_turn(session, "mother drive")
    do_speech_turn(session, "Where is she going?")
    do_sign_turn(session, "home")
    do_speech_turn(session, "Is father doing the laundry?")
    do_sign_turn(session, "where")
    do_speech_turn(session, "Can you help me find my keys?")


HELP = """
Commands:
  1                sign  -- the Deaf person signs next
  2                speak -- the hearing person speaks next
  :demo            run a short scripted conversation automatically
  :reset           clear the conversation and start over
  :help            show this message
  :quit / :q       exit
"""


def main():
    print("=" * 70)
    print("  TWO-WAY CONVERSATION DEMO -- Deaf signer <-> hearing person")
    print("  Context-aware: remembers the conversation as it goes")
    print("  Flags low-confidence Reduce results from unfamiliar vocabulary")
    print("=" * 70)
    print(HELP)

    session = ConversationSession()

    while True:
        try:
            raw = input("[1=sign, 2=speak, :help]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue
        cmd = raw.lower()

        if cmd in (":quit", ":q"):
            print("Exiting.")
            break
        elif cmd == ":help":
            print(HELP)
        elif cmd == ":reset":
            session.reset()
            print("Conversation cleared.\n")
        elif cmd == ":demo":
            run_demo(session)
        elif cmd == "1":
            glosses = input("  Type the glosses being signed: ").strip()
            if glosses:
                do_sign_turn(session, glosses)
        elif cmd == "2":
            sentence = input("  Type what the hearing person says: ").strip()
            if sentence:
                do_speech_turn(session, sentence)
        else:
            print("  Not a recognized command -- type :help for the list.")


if __name__ == "__main__":
    main()