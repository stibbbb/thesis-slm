"""
try_picker.py -- LIVE DEMO of the word-level keep/drop model.

Type anything a hearing person might say and see exactly what the model
decides for EACH WORD: drop it, or keep it and place it at a position.
Also shows whether it judged the sentence to be a wellbeing question
(which adds a leading "how").

The older sentence-generating model is shown alongside for comparison.

Each sentence is marked:
    [SEEN] this exact sentence was in the picker's training data
    [NEW]  the picker has never seen this exact sentence
[NEW] results are the meaningful ones -- they show whether it learned
the rule or just memorised the example.

Run: python try_picker.py
"""

import json

import torch

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB, CONFIRMED_GESTURES
from decoder_tinyllama import TinyLlamaForGeneration
from word_picker import WordPicker, picker_predict
from compare import load_checkpoint, decoder_predict
from make_labels import normalize_word

PICKER_PATH = "trained_picker.pt"
DECODER_PATH = "trained_tinyllama.pt"

# A spread across the 40 signs: family words, possessives (the "whose
# relative" case), states, activities, needs, questions and social.
TEST_SENTENCES = [
    "where is her mother",
    "where is his mother",
    "how is his father",
    "her mother is sad",
    "my father is tired",
    "is your teacher okay",
    "why are you so tired",
    "do you want water",
    "i need to go to the bathroom",
    "are you studying at school",
    "my friend is deaf",
    "when are you eating",
    "can i help you",
    "nice to meet you",
    "do you understand",
]


def load_seen_inputs(path="labels_train.jsonl"):
    seen = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                seen.add(json.loads(line)["input"])
    except FileNotFoundError:
        print(f"[warn] {path} not found -- cannot mark SEEN/NEW")
    return seen


def show_decisions(sentence, labels, is_question):
    """Print the model's verdict for every word -- the whole point."""
    words = sentence.split()
    n = min(len(words), len(labels))
    signs = set(CONFIRMED_GESTURES)
    print("    word-by-word decision:")
    for i in range(n):
        w, lab = words[i], labels[i]
        mark = "sign" if w in signs else "    "
        if lab == 0:
            print(f"      {w:<12} {mark}  DROP")
        else:
            norm = normalize_word(w)
            extra = f"   (shown as '{norm}')" if norm != w else ""
            print(f"      {w:<12} {mark}  KEEP -> position {lab}{extra}")
    verdict = "YES -- add 'how' at the front" if is_question else "no"
    print(f"    wellbeing question? {verdict}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    seen = load_seen_inputs()

    picker = load_checkpoint(PICKER_PATH, WordPicker, tokenizer, device)
    decoder = load_checkpoint(DECODER_PATH, TinyLlamaForGeneration,
                              tokenizer, device)

    print(f"Device: {device}   |   {len(CONFIRMED_GESTURES)} signs, "
          f"{len(ALL_VOCAB)} vocabulary words")
    print("=" * 78)
    print("  WORD-LEVEL KEEP/DROP MODEL -- live demo")
    print("  The hearing person speaks; the Deaf person's screen shows signs.")
    print("=" * 78)
    print(f"\n{'':6} {'what the hearing person says':<32} {'picker':<20} "
          f"{'older model'}")
    print("-" * 78)

    for s in TEST_SENTENCES:
        tag = "SEEN" if s in seen else "NEW "
        pick, _, _ = picker_predict(picker, tokenizer, s, device)
        dec = decoder_predict(decoder, tokenizer, s, device)
        print(f"[{tag}] {s:<32} {pick:<20} {dec}")

    print("-" * 78)
    print("\nNow type your own. Blank line or 'q' to quit.\n")

    while True:
        try:
            s = input("  hearing person says> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not s or s.lower() in ("q", "quit", "exit"):
            break

        tag = "SEEN" if s in seen else "NEW"
        pick, labels, is_q = picker_predict(picker, tokenizer, s, device)
        dec = decoder_predict(decoder, tokenizer, s, device)

        print(f"\n    [{tag}]")
        show_decisions(s, labels, is_q)
        print()
        print(f"    +{'-' * 58}+")
        print(f"    | DEAF PERSON'S SCREEN:  {pick:<33}|")
        print(f"    +{'-' * 58}+")
        print(f"    (older model would show: {dec!r})")
        print()

    print("Done.")


if __name__ == "__main__":
    main()