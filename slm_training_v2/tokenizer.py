"""
Word-level tokenizer for the from-scratch SLM.

Unlike Hugging Face tokenizers (which break words into subword pieces
from a huge pre-built vocabulary), this tokenizer is intentionally
simple: every word in our vocabulary, plus a handful of special
tokens, gets exactly one ID. This works because our task's entire
vocabulary is small and fixed -- there is no need to handle
arbitrary unseen words.

Special tokens:
  <pad> : padding, used to make all sequences in a batch the same length
  <bos> : beginning of sequence (decoder uses this to start generating)
  <eos> : end of sequence (tells the decoder when to stop generating)
  <unk> : unknown word (safety fallback, should rarely/never trigger
          given our closed vocabulary)
"""

import json
import re
from vocab_pool import ALL_VOCAB

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class SimpleTokenizer:
    def __init__(self, vocab_words):
        all_tokens = SPECIAL_TOKENS + sorted(set(vocab_words))

        self.token_to_id = {tok: i for i, tok in enumerate(all_tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(all_tokens)}

        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.unk_id = self.token_to_id["<unk>"]

        self.vocab_size = len(all_tokens)

    def _clean_and_split(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split() if text else []

    def encode(self, text, add_special_tokens=True):
        words = self._clean_and_split(text)
        ids = [self.token_to_id.get(w, self.unk_id) for w in words]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, "<unk>")
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        vocab_words = [t for t in token_to_id if t not in SPECIAL_TOKENS]
        tok = cls(vocab_words)
        return tok


if __name__ == "__main__":
    tokenizer = SimpleTokenizer(ALL_VOCAB)

    print(f"Vocabulary size (including special tokens): {tokenizer.vocab_size}")
    print(f"Token to ID mapping:")
    for tok, idx in sorted(tokenizer.token_to_id.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}  {tok}")

    print("\n--- Encode/Decode test ---")
    test_strings = [
        "where mother home",
        "Where is mother? Is she home?",
        "father study tomorrow",
    ]
    for s in test_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        print(f"  Input:   {s!r}")
        print(f"  IDs:     {ids}")
        print(f"  Decoded: {decoded!r}")
        print()

    tokenizer.save("tokenizer.json")
    print("Saved tokenizer to tokenizer.json")