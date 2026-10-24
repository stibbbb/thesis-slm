"""
train_picker.py -- trains the word-level picker for the Reduce task.

Trains TWO things at once from the same sentences:
  1. the per-word keep/drop decision
  2. the sentence-level question flag ("does this need a leading 'how'?")

Both losses are added together, so one pass teaches both. Deliberately
mirrors train_distilbert.py (same seeding, optimizer, checkpoint format,
CP progress prints) so results can be compared without training-procedure
differences confounding them.

Reads:  labels_train.jsonl   (produced by make_labels.py)
Writes: trained_picker.pt

Run: python train_picker.py
"""

import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from word_picker import WordPicker, PICKER_CONFIG

SEED = 42


def set_seed(seed=SEED):
    """Pin randomness so runs are reproducible, matching train_distilbert.py."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


VARIANT = "picker"
OUTPUT_PATH = "trained_picker.pt"
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-3
DATASET_PATH = "labels_train.jsonl"


def load_labeled_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append((row["input"], row["labels"], row["question"]))
    return examples


class PickerDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.data = []
        for input_text, labels, question in examples:
            ids = tokenizer.encode(input_text, add_special_tokens=False)
            # The tokenizer is word-level, so one token per label. The
            # guard catches any silent mismatch rather than training on
            # misaligned data.
            if len(ids) == len(labels):
                self.data.append((ids, labels, question))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, labels, question = self.data[idx]
        return torch.tensor(ids), torch.tensor(labels), torch.tensor(question)


def collate_fn(batch, pad_id):
    ids_list, labels_list, q_list = zip(*batch)
    max_len = max(len(ids) for ids in ids_list)
    padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    # -100 is ignored by the loss, so padded slots train nothing
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(zip(ids_list, labels_list)):
        padded_ids[i, :len(ids)] = ids
        padded_labels[i, :len(labels)] = labels
    questions = torch.stack(q_list)
    return padded_ids, padded_labels, questions


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CP1: [{VARIANT}] device: {device}  (seed={SEED})", flush=True)

    tokenizer = SimpleTokenizer(ALL_VOCAB)
    examples = load_labeled_examples(DATASET_PATH)
    dataset = PickerDataset(examples, tokenizer)
    n_q = sum(int(q) for _, _, q in dataset.data)
    print(f"CP1b: examples: {len(dataset)}  (from {DATASET_PATH})", flush=True)
    print(f"CP1c: question-flag rows: {n_q}", flush=True)

    if len(dataset) == 0:
        raise SystemExit("No usable training rows. Run make_labels.py first.")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))

    model = WordPicker(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        **PICKER_CONFIG,
    ).to(device)
    print(f"CP2: {model.count_parameters():,} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    tag_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    q_criterion = nn.CrossEntropyLoss()

    print("CP3: starting training", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_tag, total_q, n_batches = 0.0, 0.0, 0
        for input_ids, labels, questions in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            questions = questions.to(device)

            optimizer.zero_grad()
            tag_logits, q_logits = model(input_ids)

            tag_loss = tag_criterion(
                tag_logits.reshape(-1, tag_logits.size(-1)), labels.reshape(-1)
            )
            q_loss = q_criterion(q_logits, questions)
            loss = tag_loss + q_loss

            loss.backward()
            optimizer.step()

            total_tag += tag_loss.item()
            total_q += q_loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"word loss: {total_tag/n_batches:.4f}   "
                  f"question loss: {total_q/n_batches:.4f}", flush=True)

    print("CP4: training finished", flush=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "variant": VARIANT,
        **PICKER_CONFIG,
    }, OUTPUT_PATH)
    print(f"CP5: saved to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()