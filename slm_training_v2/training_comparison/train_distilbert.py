import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from encoder_distilbert import DistilBertForRestructure, DISTILBERT_CONFIG

SEED = 42

def set_seed(seed=SEED):
    """Pin all sources of randomness for reproducible training runs.
    Added after the out-of-vocabulary-combination instability finding
    (slm_progress_notes.md, known issue 13.12) showed retrains behaving
    differently purely from unseeded initialization, separate from any
    real data change."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

VARIANT = "distilbert"
OUTPUT_PATH = "trained_distilbert.pt"
MAX_LEN = 16
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-3
DATASET_PATH = "dataset_v4_train.jsonl"


def load_restructure_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task"] == "restructure":
                examples.append((row["input"], row["output"]))
    return examples


def build_position_labels(input_text, output_text, tokenizer):
    input_tokens = tokenizer._clean_and_split(input_text)
    output_tokens = tokenizer._clean_and_split(output_text)
    if sorted(input_tokens) != sorted(output_tokens):
        return None
    output_word_to_pos = {}
    for pos, word in enumerate(output_tokens):
        output_word_to_pos.setdefault(word, []).append(pos)
    labels = []
    used = {word: 0 for word in output_word_to_pos}
    for word in input_tokens:
        positions = output_word_to_pos[word]
        labels.append(positions[used[word]])
        used[word] += 1
    return labels


class RestructureDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.data = []
        for input_text, output_text in examples:
            ids = tokenizer.encode(input_text, add_special_tokens=False)
            labels = build_position_labels(input_text, output_text, tokenizer)
            if labels is not None and len(ids) == len(labels):
                self.data.append((ids, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, labels = self.data[idx]
        return torch.tensor(ids), torch.tensor(labels)


def collate_fn(batch, pad_id):
    ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in ids_list)
    padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(zip(ids_list, labels_list)):
        padded_ids[i, :len(ids)] = ids
        padded_labels[i, :len(labels)] = labels
    return padded_ids, padded_labels


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CP1: [{VARIANT}] device: {device}  (seed={SEED})", flush=True)
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    examples = load_restructure_examples(DATASET_PATH)
    dataset = RestructureDataset(examples, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))
    model = DistilBertForRestructure(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        **DISTILBERT_CONFIG,
    ).to(device)
    print(f"CP2: {model.count_parameters():,} params", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    print(f"CP3: starting training", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss: {avg_loss:.4f}", flush=True)
    print(f"CP4: training finished", flush=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "variant": VARIANT,
        **DISTILBERT_CONFIG,
    }, OUTPUT_PATH)
    print(f"CP5: saved to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()