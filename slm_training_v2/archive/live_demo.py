"""
Interactive live demo for the SLM pipeline -- runs the ONNX deployment
path (onnx_inference.py), so this is the exact code that will
eventually run on the Raspberry Pi 5, not a PyTorch stand-in. Built to
be shown live to an adviser or panel while the Pi and the recognition
module are still pending.

Commands:
  :mode 1        switch to Direction 1 (signs -> sentence)   [default]
  :mode 2        switch to Direction 2 (sentence -> glosses)
  :verbose       toggle showing the intermediate pipeline steps
  :demo          run a few canned examples in the current mode
  :edge-cases    run the hardening edge-case suite live, on purpose
  :help          show this help
  :quit / :exit / :q

Anything else typed is processed as input in the current mode.

Run: python live_demo.py
"""

import time
import onnxruntime as ort

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from onnx_config import ENCODER_ONNX_PATH, DECODER_ONNX_PATH
from onnx_inference import (
    map_glosses, restructure_onnx, generate_onnx,
    sign_to_sentence_onnx, sentence_to_glosses_onnx,
)

BANNER = r"""
==============================================================
  SLM LIVE DEMO -- ASL Recognition & Two-Way Communication
  Running the ONNX deployment path (same code as the Pi 5)
==============================================================
  Mode 1 (default): signs -> sentence    (Deaf signer -> hearing person)
  Mode 2:           sentence -> glosses  (hearing person -> Deaf signer)

  Type ':help' for commands, ':quit' to exit.
==============================================================
"""

HELP = """
Commands:
  :mode 1        switch to Direction 1 (signs -> sentence)
  :mode 2        switch to Direction 2 (sentence -> glosses)
  :verbose       toggle showing intermediate pipeline steps (currently {v})
  :demo          run a few canned examples in the current mode
  :edge-cases    run the hardening edge-case suite live
  :help          show this message
  :quit / :exit / :q

Anything else you type is processed as input in the current mode.
Example (mode 1): drive mother where
Example (mode 2): Where is mother? Is she home?
"""

DEMO_EXAMPLES = {
    1: ["drive mother where", "home drive father", "papa laundry", "fine how"],
    2: ["Where is mother? Is she home?", "How are you? I am fine."],
}


def run_mode1(enc_session, dec_session, tokenizer, text, verbose):
    t0 = time.perf_counter()
    mapped = map_glosses(text)
    ordered = restructure_onnx(enc_session, tokenizer, mapped)
    if not ordered:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  (nothing recognizable in that input -- {elapsed:.1f} ms)")
        return
    sentence = generate_onnx(dec_session, tokenizer, ordered)
    elapsed = (time.perf_counter() - t0) * 1000
    if verbose:
        print(f"  gloss mapping : {mapped!r}")
        print(f"  restructured  : {ordered!r}")
    print(f"  SENTENCE      : {sentence!r}")
    print(f"  ({elapsed:.1f} ms)")


def run_mode2(dec_session, tokenizer, text, verbose):
    t0 = time.perf_counter()
    glosses = sentence_to_glosses_onnx(dec_session, tokenizer, text)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  GLOSSES       : {glosses!r}")
    print(f"  ({elapsed:.1f} ms)")


def run_demo(enc_session, dec_session, tokenizer, mode, verbose):
    print(f"\n--- Demo: mode {mode} ---")
    for example in DEMO_EXAMPLES[mode]:
        print(f"\n> {example}")
        if mode == 1:
            run_mode1(enc_session, dec_session, tokenizer, example, verbose)
        else:
            run_mode2(dec_session, tokenizer, example, verbose)


def run_edge_cases(enc_session, dec_session, tokenizer):
    print("\n--- Edge-case suite (see slm_progress_notes.md \u00a711) ---")
    cases = [
        ("empty string", ""),
        ("unknown word mixed in", "mother xyz drive"),
        ("list instead of string", ["mother", "drive", "where"]),
        ("too many words", ("mother father drive home laundry where how fine name "
                             "mother father drive home laundry where how fine name")),
    ]
    for label, case in cases:
        print(f"\n[{label}] input: {case!r}")
        try:
            result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, case)
            print(f"  -> {result!r}")
        except Exception as e:
            print(f"  -> raised {type(e).__name__}: {e}")

    print("\n[invalid type] input: None")
    try:
        sign_to_sentence_onnx(enc_session, dec_session, tokenizer, None)
    except TypeError as e:
        print(f"  -> correctly raised TypeError: {e}")


def main():
    print(BANNER)
    print("Loading models...", end=" ", flush=True)
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    enc_session = ort.InferenceSession(ENCODER_ONNX_PATH)
    dec_session = ort.InferenceSession(DECODER_ONNX_PATH)
    print("done.\n")

    mode = 1
    verbose = True

    while True:
        try:
            raw = input(f"[mode {mode}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in (":quit", ":exit", ":q"):
            print("Exiting.")
            break
        elif cmd == ":help":
            print(HELP.format(v="ON" if verbose else "OFF"))
        elif cmd == ":verbose":
            verbose = not verbose
            print(f"Verbose intermediate steps: {'ON' if verbose else 'OFF'}")
        elif cmd == ":mode" and len(parts) > 1 and parts[1] in ("1", "2"):
            mode = int(parts[1])
            label = "signs -> sentence" if mode == 1 else "sentence -> glosses"
            print(f"Switched to mode {mode} ({label}).")
        elif cmd == ":demo":
            run_demo(enc_session, dec_session, tokenizer, mode, verbose)
        elif cmd == ":edge-cases":
            run_edge_cases(enc_session, dec_session, tokenizer)
        else:
            try:
                if mode == 1:
                    run_mode1(enc_session, dec_session, tokenizer, raw, verbose)
                else:
                    run_mode2(dec_session, tokenizer, raw, verbose)
            except TypeError as e:
                print(f"  -> rejected: {e}")
            except Exception as e:
                print(f"  -> unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()