"""
word_picker.py -- word-level keep/drop model for the Reduce task.

THE IDEA (the adviser's requirement, implemented literally)
Instead of reading a sentence and writing a new sentence, this model
looks at EACH WORD and decides about that word:

    class 0        -> DROP this word
    class 1..N     -> KEEP it, and place it at that position

TWO OUTPUT HEADS

  1. tag_head      -- the per-word decision above.
  2. question_head -- ONE decision for the whole sentence: is this a
                      wellbeing question? If yes, "how" is placed at
                      the front of the output.

The second head exists because a few correct answers contain a word
that is in no part of the input:
      "is he fine"  ->  "how he"     ("how" appears nowhere in the input)
      "he is fine"  ->  "he fine"    (no marker needed)
Word-picking alone cannot produce "how" -- there is nothing to pick.
The flag supplies that one marker; every other word is still chosen
from what the person actually said.

WHY BIDIRECTIONAL ATTENTION
Deciding whether to drop "is" requires seeing the whole sentence --
the words before AND after it. An encoder does exactly that. A
left-to-right decoder would have to judge "is" in "what is your name"
before it had seen "name".

STRUCTURAL GUARANTEE
Apart from the single "how" marker, the model can only output words it
was given. It cannot invent content. The failure where "what are you
eating" produced "name mother" is impossible under this design.

Reuses the embedding and transformer-block code already validated for
the Restructure task, so the only genuinely new parts are the two
heads.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from make_labels import normalize_word, QUESTION_MARKER
from vocab_pool import CONFIRMED_GESTURES

from encoder_distilbert import DistilBertEmbeddings, TransformerBlock


SIGNS = set(CONFIRMED_GESTURES)

# Words that flip a sentence's meaning. Dropping them silently inverts
# what was said: "i dont know" became "know". They are mapped onto the
# sign "no" so the negation survives instead of vanishing.
NEGATIONS = {"not", "dont", "don't", "no", "never", "cannot", "cant",
             "can't", "wont", "won't", "didnt", "didn't", "isnt", "isn't"}


class WordPicker(nn.Module):
    """
    Same backbone as DistilBertForRestructure, two different heads.

    forward() returns (tag_logits, question_logits):
      tag_logits      (batch, seq_len, max_len + 1)
                      index 0 = DROP, 1..max_len = keep at that position
      question_logits (batch, 2)
                      index 1 = prepend the "how" marker
    """

    def __init__(self, vocab_size, hidden_size=64, n_heads=4, n_layers=2,
                 ffn_size=128, max_len=16, pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        self.embeddings = DistilBertEmbeddings(
            vocab_size, hidden_size, max_len, pad_id, dropout
        )
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, ffn_size, dropout)
            for _ in range(n_layers)
        ])
        # +1 for the DROP class at index 0
        self.tag_head = nn.Linear(hidden_size, max_len + 1)
        # sentence-level: 0 = normal, 1 = prepend "how"
        self.question_head = nn.Linear(hidden_size, 2)

    # The three stages below are the pipeline boxes Embedding, Contextualize
    # and Space Restriction (slm_pipeline.py). forward() runs them in order,
    # so training and the pipeline use exactly the same computation.

    def embed(self, input_ids):
        """Embedding: each word ID -> 64 numbers (word meaning + position)."""
        return self.embeddings(input_ids)

    def contextualize(self, x, padding_mask):
        """Contextualize: the transformer blocks let every word read the
        whole sentence before any keep/drop decision is made."""
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)
        return x

    def heads(self, x, padding_mask):
        """Space Restriction scores: 17 choices per word, plus the
        sentence-level "add how?" decision."""
        tag_logits = self.tag_head(x)

        # Sentence-level view: average the real (non-padding) words.
        keep = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        question_logits = self.question_head(pooled)

        return tag_logits, question_logits

    def forward(self, input_ids):
        padding_mask = (input_ids == self.pad_id)
        x = self.embed(input_ids)
        x = self.contextualize(x, padding_mask)
        return self.heads(x, padding_mask)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------
# Running the model. This lives here, next to the model itself, because
# it is the other half of the same idea: the class above produces the
# per-word verdicts, and this function turns those verdicts into the
# words shown on screen.
# --------------------------------------------------------------------

def restrict_to_signs(tag_logits, q_logits, words):
    """
    Space Restriction: turn the picker's scores into the signs shown to
    the Deaf reader. `words` must line up one-to-one with the scored
    tokens. Returns (output_words, per_word_labels, question_flag).

    Two guarantees are enforced here rather than left to the model:

    1. ONLY SIGNS SURVIVE. A word with no sign cannot be shown to the
       Deaf reader, whatever the model scored it.

    2. NO TWO WORDS SHARE A POSITION. The model scores each word
       independently, so nothing stops it assigning slot 1 to two
       different words -- which produced garbled output like
       "how no im". Hungarian assignment resolves the scores into a
       one-to-one word-to-slot match.
    """
    is_question = int(q_logits[0].argmax().item())
    n = min(len(words), tag_logits.shape[1])
    scores = tag_logits[0, :n].cpu().numpy()      # n x (max_len + 1)

    # --- decide keep vs drop, one word at a time ---
    # column 0 is the DROP score; the rest are position scores.
    keep_idx = []
    labels = [0] * n
    negated = False
    for i in range(n):
        w = normalize_word(words[i].lower())
        if words[i].lower() in NEGATIONS or w in NEGATIONS:
            negated = True
            continue
        # A word only reaches the Deaf person's screen if there is a
        # SIGN for it. Vocabulary words like "know" and "now" exist so
        # the model can learn to drop them; showing them would put a
        # word on screen the reader has no sign for.
        if w not in SIGNS:
            continue
        if scores[i, 1:].max() > scores[i, 0]:
            keep_idx.append(i)

    # --- assign the kept words to distinct positions ---
    if keep_idx:
        sub = scores[keep_idx][:, 1:len(keep_idx) + 1]
        rows, cols = linear_sum_assignment(-sub)
        for r, c in zip(rows, cols):
            labels[keep_idx[r]] = int(c) + 1

    picked = [(labels[i], normalize_word(words[i].lower()))
              for i in range(n) if labels[i] > 0]
    picked.sort(key=lambda pw: pw[0])
    out_words = [w for _, w in picked]
    # a negation that was spoken must be shown, or the meaning inverts
    if negated and "no" not in out_words:
        out_words = ["no"] + out_words
    if is_question:
        out_words = [QUESTION_MARKER] + out_words

    return out_words, labels, is_question


def picker_predict(model, tokenizer, text, device):
    """
    Returns (output_text, per_word_labels, question_flag) for one
    sentence. The words are cleaned by the tokenizer's own rules first,
    so punctuation ("mother?") can no longer knock a word out of line
    with its score, and input longer than the model's 16 positions is
    cut instead of crashing. slm_pipeline.py runs the same steps as
    separate, named pipeline stages.
    """
    words = tokenizer._clean_and_split(text)[:model.max_len]
    if not words:
        return "", [], 0
    ids = [tokenizer.token_to_id.get(w, tokenizer.unk_id) for w in words]
    x = torch.tensor([ids]).to(device)
    with torch.no_grad():
        tag_logits, q_logits = model(x)
    out_words, labels, is_question = restrict_to_signs(tag_logits, q_logits, words)
    return " ".join(out_words), labels, is_question


PICKER_CONFIG = {
    "hidden_size": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_size": 128,
    "max_len": 16,
    "dropout": 0.1,
}


if __name__ == "__main__":
    from tokenizer import SimpleTokenizer
    from vocab_pool import ALL_VOCAB

    tokenizer = SimpleTokenizer(ALL_VOCAB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = WordPicker(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        **PICKER_CONFIG,
    ).to(device)

    print("Architecture: DistilBERT-style word picker (keep/drop + question flag)")
    print(f"Config: {PICKER_CONFIG}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    sample = "is he fine"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    x = torch.tensor([ids]).to(device)
    with torch.no_grad():
        tag_logits, q_logits = model(x)
    print(f"\nSample input: {sample!r} -> token IDs {ids}")
    print(f"  tag_logits shape     : {tuple(tag_logits.shape)}  "
          f"(batch, words, choices)")
    print(f"  question_logits shape: {tuple(q_logits.shape)}  (batch, 2)")
    print("Forward pass successful.")