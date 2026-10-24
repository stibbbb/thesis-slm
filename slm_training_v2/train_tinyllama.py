import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from decoder_tinyllama import TinyLlamaForGeneration, TINYLLAMA_CONFIG

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

VARIANT = "tinyllama"
OUTPUT_PATH = "trained_tinyllama.pt"
MAX_LEN = 32
BATCH_SIZE = 8
EPOCHS = 150
LEARNING_RATE = 1e-3
DATASET_PATH = "dataset_v4_train.jsonl"


# Complete ONLY. The word picker handles Reduce now. Training one model
# on both meant it had to guess from the input alone whether it was
# EXPANDING a gloss sequence or SHRINKING a sentence -- two opposite
# jobs -- and guessing wrong produced answers from the wrong task.
def load_generation_examples(path, tasks=("complete",)):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task"] in tasks:
                examples.append((row["input"], row["output"]))
    return examples


class GenerationDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_LEN):
        self.data = []
        for input_text, output_text in examples:
            src = tokenizer.encode(input_text, add_special_tokens=False)
            tgt = tokenizer.encode(output_text, add_special_tokens=True)
            if len(src) < max_len and len(tgt) < max_len:
                self.data.append((src, tgt))

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
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CP1: [{VARIANT}] device: {device}  (seed={SEED})", flush=True)
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    examples = load_generation_examples(DATASET_PATH)
    print(f"CP1b: examples: {len(examples)}  (from {DATASET_PATH})", flush=True)
    dataset = GenerationDataset(examples, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id))
    model = TinyLlamaForGeneration(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        **TINYLLAMA_CONFIG,
    ).to(device)
    print(f"CP2: {model.count_parameters():,} params", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    print(f"CP3: starting training", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for src_ids, tgt_ids in loader:
            src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
            tgt_input = tgt_ids[:, :-1]
            tgt_labels = tgt_ids[:, 1:]
            optimizer.zero_grad()
            logits = model(src_ids, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss: {avg_loss:.4f}", flush=True)
    print(f"CP4: training finished", flush=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "variant": VARIANT,
        **TINYLLAMA_CONFIG,
    }, OUTPUT_PATH)
    print(f"CP5: saved to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()