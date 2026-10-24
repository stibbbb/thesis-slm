"""
health_check.py -- Runs all three verification layers back to back and
prints one summary. Answers "is the model actually working" with real
evidence instead of a guess, in one command instead of three.

  Layer 1: robustness       -- pytest test_hardening.py (36 tests)
  Layer 2: known-case check -- fixed cases with a stable, verified answer
  Layer 3: generalization   -- the 4 held-out context-recovery cases,
                                the freshest valid held-out evidence for
                                whatever checkpoint is CURRENTLY deployed

IMPORTANT: this intentionally does NOT re-run measure_holdout.py's
original 14/14 held-out check. That check was valid for a checkpoint
that no longer exists -- the current checkpoint has since been trained
on that exact held-out data (§9/§10 of the notes). Running it today and
reporting the number would be memorization dressed up as generalization.
Layer 3 here is the correct replacement: the ONLY held-out data the
current checkpoint has genuinely never seen in this exact form.

Run: python health_check.py
"""

import json
import subprocess
import sys

import onnxruntime as ort

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from onnx_config import ENCODER_ONNX_PATH, DECODER_ONNX_PATH
from onnx_inference import sign_to_sentence_onnx, sentence_to_glosses_onnx


import re

def normalize(text):
    text = text.strip().lower()
    text = re.sub(r"[.?!,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def run_layer1():
    print("=" * 70)
    print("LAYER 1: Robustness  (pytest test_hardening.py)")
    print("=" * 70)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_hardening.py", "-q"],
        capture_output=True, text=True,
    )
    print(proc.stdout[-2000:])
    if proc.stderr.strip():
        print(proc.stderr[-500:])
    ok = proc.returncode == 0
    print(f"LAYER 1: {'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED -- see above'}\n")
    return ok


def run_layer2(enc_session, dec_session, tokenizer):
    print("=" * 70)
    print("LAYER 2: Known-case correctness  (stable, previously-verified cases)")
    print("=" * 70)

    sign_cases = [
        ("papa laundry", "father is doing laundry"),
        ("fine how", "how are you i am fine"),
    ]
    speech_cases = [
        ("Where is mother? Is she home?", "where home"),
        ("How are you? I am fine.", "how you"),
    ]
    watch_cases = [
        "drive mother where",
        "home drive father",
    ]

    all_ok = True
    for glosses, expected in sign_cases:
        result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, glosses)
        ok = normalize(result) == normalize(expected)
        all_ok &= ok
        print(f"  [{'OK' if ok else 'CHECK'}] '{glosses}' -> '{result}'"
              + ("" if ok else f"  (expected '{expected}')"))

    for sentence, expected in speech_cases:
        result = sentence_to_glosses_onnx(dec_session, tokenizer, sentence)
        ok = normalize(result) == normalize(expected)
        all_ok &= ok
        print(f"  [{'OK' if ok else 'CHECK'}] '{sentence}' -> '{result}'"
              + ("" if ok else f"  (expected '{expected}')"))

    print("\n  Watch-only (documented as unstable across retrains -- shown, not scored):")
    for glosses in watch_cases:
        result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, glosses)
        print(f"    '{glosses}' -> '{result}'")

    print(f"\nLAYER 2: {'ALL SCORED CASES MATCHED' if all_ok else 'SOME CASES DID NOT MATCH -- see above'}\n")
    return all_ok


def run_layer3(enc_session, dec_session, tokenizer):
    print("=" * 70)
    print("LAYER 3: Generalization  (held-out context-recovery cases)")
    print("=" * 70)
    try:
        with open("dataset_context_test.jsonl", "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("  dataset_context_test.jsonl not found -- skipping this layer.\n")
        return None

    all_ok = True
    for row in rows:
        result = sign_to_sentence_onnx(enc_session, dec_session, tokenizer, row["input"])
        ok = normalize(result) == normalize(row["output"])
        all_ok &= ok
        print(f"  [{'OK' if ok else 'CHECK'}] context={row['context_turn']!r} "
              f"current={row['current_turn']!r} -> '{result}'"
              + ("" if ok else f"  (expected '{row['output']}')"))

    print(f"\nLAYER 3: {'ALL HELD-OUT CASES PASSED' if all_ok else 'SOME HELD-OUT CASES FAILED -- see above'}\n")
    return all_ok


def main():
    print("SLM HEALTH CHECK -- three verification layers, one summary\n")

    layer1_ok = run_layer1()

    tokenizer = SimpleTokenizer(ALL_VOCAB)
    enc_session = ort.InferenceSession(ENCODER_ONNX_PATH)
    dec_session = ort.InferenceSession(DECODER_ONNX_PATH)

    layer2_ok = run_layer2(enc_session, dec_session, tokenizer)
    layer3_ok = run_layer3(enc_session, dec_session, tokenizer)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    def fmt(v):
        return "PASS" if v is True else ("SKIPPED" if v is None else "CHECK NEEDED")
    print(f"  Layer 1  Robustness (pytest)              : {fmt(layer1_ok)}")
    print(f"  Layer 2  Known-case correctness            : {fmt(layer2_ok)}")
    print(f"  Layer 3  Generalization (held-out context) : {fmt(layer3_ok)}")
    print()
    if layer1_ok and layer2_ok and (layer3_ok or layer3_ok is None):
        print("All layers passed. This is real evidence for the CURRENT checkpoint --")
        print("re-run this after every retrain. Older results (e.g. the original")
        print("14/14 held-out numbers from an earlier checkpoint) do not carry over")
        print("automatically to a checkpoint that has since been retrained.")
    else:
        print("One or more layers need attention -- see the details above before")
        print("trusting this checkpoint for a demo or reporting these numbers.")


if __name__ == "__main__":
    main()