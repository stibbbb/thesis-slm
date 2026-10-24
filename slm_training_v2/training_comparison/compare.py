"""
compare.py -- head-to-head comparison of the two Reduce approaches on
the SAME held-out rows.

  Approach A (current) : TinyLlama decoder GENERATES the gloss output
  Approach B (new)     : DistilBERT picker SELECTS which words to keep,
                         where they go, and whether a leading "how" is
                         needed

Both are measured on labels_test.jsonl -- phrasings neither model saw
during training. Using the identical row set is what makes the
comparison fair.

Run: python compare.py
"""

import json
import re

import torch

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from word_picker import WordPicker, picker_predict
from decoder_tinyllama import TinyLlamaForGeneration
from make_labels import normalize_word, QUESTION_MARKER  # noqa: F401

PICKER_PATH = "trained_picker.pt"
DECODER_PATH = "trained_tinyllama.pt"
TEST_PATH = "labels_test.jsonl"


def norm_text(t):
    t = t.lower().strip()
    t = re.sub(r"[.?!,]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def load_checkpoint(path, ModelClass, tokenizer, device):
    ckpt = torch.load(path, map_location=device)
    cfg = {k: v for k, v in ckpt.items()
           if k not in ("model_state_dict", "vocab_size", "variant")}
    model = ModelClass(
        vocab_size=ckpt["vocab_size"],
        pad_id=tokenizer.pad_id,
        **cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def decoder_predict(model, tokenizer, text, device):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return ""
    src = torch.tensor([ids]).to(device)
    return model.generate(src, tokenizer, max_new_tokens=16)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    print(f"Device: {device}")

    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    n_q = sum(r.get("question", 0) for r in rows)
    print(f"Held-out rows (identical for both approaches): {len(rows)}")
    print(f"  of which need a leading 'how': {n_q}\n")

    picker = load_checkpoint(PICKER_PATH, WordPicker, tokenizer, device)
    decoder = load_checkpoint(DECODER_PATH, TinyLlamaForGeneration, tokenizer, device)

    pick_correct, dec_correct = 0, 0
    q_correct = 0
    pick_misses, dec_misses = [], []
    invented = []

    for r in rows:
        expected = r["output"]
        inp = r["input"]

        pick_out, _, is_q = picker_predict(picker, tokenizer, inp, device)
        dec_out = decoder_predict(decoder, tokenizer, inp, device)

        if is_q == r.get("question", 0):
            q_correct += 1

        if norm_text(pick_out) == norm_text(expected):
            pick_correct += 1
        else:
            pick_misses.append((inp, expected, pick_out))

        if norm_text(dec_out) == norm_text(expected):
            dec_correct += 1
        else:
            dec_misses.append((inp, expected, dec_out))

        # Did the DECODER produce a word traceable to no input word?
        # The picker cannot, apart from the single "how" marker.
        avail = {QUESTION_MARKER}
        for w in inp.split():
            avail.add(w.lower())
            avail.add(normalize_word(w.lower()))
        for w in dec_out.split():
            if w.lower() not in avail:
                invented.append((inp, dec_out, w))
                break

    n = len(rows)
    print("=" * 68)
    print("RESULTS on identical held-out rows")
    print("=" * 68)
    print(f"  Approach A -- decoder GENERATES : {dec_correct}/{n}  "
          f"({100.0*dec_correct/n:.1f}%)")
    print(f"  Approach B -- picker SELECTS    : {pick_correct}/{n}  "
          f"({100.0*pick_correct/n:.1f}%)")
    print()
    print(f"  Question-flag accuracy (picker) : {q_correct}/{n}  "
          f"({100.0*q_correct/n:.1f}%)")
    print()
    print(f"  Decoder outputs containing an unsaid word: {len(invented)}/{n}")
    print(f"  Picker  outputs containing an unsaid word: 0/{n} "
          f"(impossible, apart from the 'how' marker)")
    print()

    if pick_misses:
        print(f"--- Picker misses ({len(pick_misses)}) ---")
        for inp, exp, got in pick_misses[:12]:
            print(f"    {inp!r:32s} expected {exp!r:20s} got {got!r}")
        if len(pick_misses) > 12:
            print(f"    ... and {len(pick_misses)-12} more")
        print()

    if dec_misses:
        print(f"--- Decoder misses ({len(dec_misses)}) ---")
        for inp, exp, got in dec_misses[:12]:
            print(f"    {inp!r:32s} expected {exp!r:20s} got {got!r}")
        if len(dec_misses) > 12:
            print(f"    ... and {len(dec_misses)-12} more")
        print()

    if invented:
        print("--- Examples where the decoder invented a word ---")
        for inp, out, w in invented[:6]:
            print(f"    {inp!r:32s} -> {out!r}   (invented {w!r})")
        print()


if __name__ == "__main__":
    main()