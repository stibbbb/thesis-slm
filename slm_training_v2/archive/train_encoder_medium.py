"""
Training script for EncoderForRestructure.

Since we are not using Hugging Face's Trainer (that's built for
pre-trained HF models), we write the training loop manually:
  1. Load restructure examples from dataset.jsonl
  2. For each example, figure out which OUTPUT SLOT each INPUT token
     belongs in (the label the model needs to predict)
  3. Batch, forward pass, compute loss, backpropagate, update weights
  4. Repeat for several epochs, printing loss as we go
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from encoder_model_medium import EncoderForRestructure

MAX_LEN = 16
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-3
DATASET_PATH = "dataset.jsonl"
OUTPUT_PATH = "trained_encoder_medium.pt"


def load_restructure_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task"] == "restructure":
                examples.append((row["input"], row["output"]))
    return examples


def build_position_labels(input_text, output_text, tokenizer):
    """For each token in the (scrambled) input, figure out which index
    in the (correctly ordered) output it should map to.

    Example:
      input  = "drive mother"   -> tokens [drive, mother]
      output = "mother drive"   -> tokens [mother, drive]
      "drive" is input position 0, and it belongs at output position 1
      "mother"  is input position 1, and it belongs at output position 0
      labels = [1, 0]
    """
    input_tokens = tokenizer._clean_and_split(input_text)
    output_tokens = tokenizer._clean_and_split(output_text)

    # Defensive check: restructure should be a pure reordering (same
    # multiset of words on both sides). If counts don't match, skip.
    if sorted(input_tokens) != sorted(output_tokens):
        return None

    # For repeated words this is ambiguous, but our vocabulary has no
    # repeated words within a single example, so a simple lookup works.
    output_word_to_pos = {}
    for pos, word in enumerate(output_tokens):
        output_word_to_pos.setdefault(word, []).append(pos)

    labels = []
    used = {word: 0 for word in output_word_to_pos}
    for word in input_tokens:
        positions = output_word_to_pos[word]
        idx = used[word]
        labels.append(positions[idx])
        used[word] += 1

    return labels


class RestructureDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        skipped = 0
        for input_text, output_text in examples:
            ids = tokenizer.encode(input_text, add_special_tokens=False)
            labels = build_position_labels(input_text, output_text, tokenizer)

            if labels is None or len(ids) != len(labels):
                skipped += 1
                continue

            self.data.append((ids, labels))

        if skipped:
            print(f"Warning: skipped {skipped} malformed examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, labels = self.data[idx]
        return torch.tensor(ids), torch.tensor(labels)


def collate_fn(batch, pad_id):
    """Pads all sequences in a batch to the same length so they can be
    stacked into a single tensor. Labels are padded with -100, a value
    PyTorch's cross-entropy loss is told to ignore."""
    ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in ids_list)

    padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (ids, labels) in enumerate(zip(ids_list, labels_list)):
        padded_ids[i, : len(ids)] = ids
        padded_labels[i, : len(labels)] = labels

    return padded_ids, padded_labels


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CP1: using device: {device}", flush=True)

    tokenizer = SimpleTokenizer(ALL_VOCAB)
    print(f"CP2: tokenizer ready, vocab size {tokenizer.vocab_size}", flush=True)

    examples = load_restructure_examples(DATASET_PATH)
    print(f"CP3: loaded {len(examples)} restructure examples", flush=True)

    dataset = RestructureDataset(examples, tokenizer, max_len=MAX_LEN)
    print(f"CP4: dataset built, {len(dataset)} usable examples", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )

    model = EncoderForRestructure(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_len=MAX_LEN,
        pad_id=tokenizer.pad_id,
    ).to(device)
    print(f"CP5: model built, {model.count_parameters():,} trainable params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print("CP6: starting training", flush=True)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_batches = 0

        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)  # (batch, seq_len, max_len)

            # Reshape for cross-entropy: (batch * seq_len, max_len) vs (batch * seq_len,)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss: {avg_loss:.4f}", flush=True)

    print("CP7: training finished", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "max_len": MAX_LEN,
    }, OUTPUT_PATH)
    print(f"CP8: model saved to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()