"""
Training script for DecoderForGenerationTask.
Handles both COMPLETE and REDUCE tasks in one training run.
Uses teacher forcing: during training, the model sees correct
previous tokens rather than its own predictions, which makes
training more stable.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from decoder_model import DecoderForGenerationTask, DECODER_CONFIGS

# --- CHANGE THESE THREE LINES FOR EACH VARIANT ---
VARIANT = "small"
OUTPUT_PATH = "trained_decoder_small.pt"
# -------------------------------------------------

MAX_LEN = 32
BATCH_SIZE = 8
EPOCHS = 150
LEARNING_RATE = 1e-3
DATASET_PATH = "dataset.jsonl"


def load_generation_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task"] in ("complete", "reduce"):
                examples.append((row["input"], row["output"]))
    return examples


class GenerationDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_LEN):
        self.data = []
        for input_text, output_text in examples:
            src_ids = tokenizer.encode(input_text, add_special_tokens=False)
            tgt_ids = tokenizer.encode(output_text, add_special_tokens=True)
            if len(src_ids) < max_len and len(tgt_ids) < max_len:
                self.data.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch, pad_id):
    src_list, tgt_list = zip(*batch)
    src_max = max(len(s) for s in src_list)
    tgt_max = max(len(t) for t in tgt_list)

    padded_src = torch.full((len(batch), src_max), pad_id, dtype=torch.long)
    padded_tgt = torch.full((len(batch), tgt_max), pad_id, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_list, tgt_list)):
        padded_src[i, :len(src)] = src
        padded_tgt[i, :len(tgt)] = tgt

    return padded_src, padded_tgt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CP1: [{VARIANT}] using device: {device}", flush=True)

    tokenizer = SimpleTokenizer(ALL_VOCAB)
    cfg = DECODER_CONFIGS[VARIANT]

    examples = load_generation_examples(DATASET_PATH)
    print(f"CP2: loaded {len(examples)} complete+reduce examples", flush=True)

    dataset = GenerationDataset(examples, tokenizer, max_len=MAX_LEN)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )

    model = DecoderForGenerationTask(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        **cfg,
    ).to(device)
    print(f"CP3: [{VARIANT}] model built, {model.count_parameters():,} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    print(f"CP4: starting training [{VARIANT}]", flush=True)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_batches = 0

        for src_ids, tgt_ids in loader:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            # Teacher forcing:
            # Input to decoder: all target tokens EXCEPT the last one
            # Labels:           all target tokens EXCEPT the first one
            # So at each position, the model predicts the NEXT token
            tgt_input = tgt_ids[:, :-1]
            tgt_labels = tgt_ids[:, 1:]

            optimizer.zero_grad()
            logits = model(src_ids, tgt_input)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.reshape(-1),
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss: {avg_loss:.4f}", flush=True)

    print(f"CP5: training finished [{VARIANT}]", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "max_len": MAX_LEN,
        "variant": VARIANT,
        **cfg,
    }, OUTPUT_PATH)
    print(f"CP6: saved to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()