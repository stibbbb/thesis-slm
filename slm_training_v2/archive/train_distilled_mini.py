"""
Knowledge Distillation training script.

TEACHER: trained_decoder_medium.pt (our best Decoder variant)
STUDENTS: three progressively smaller models trained to mimic
          the teacher's output distributions, not just the raw
          dataset labels.

Why distillation works: instead of just telling the student
"the correct answer is X", we show it the teacher's full
probability distribution over all possible next words. This
gives the student much richer training signal -- it learns
not just what's correct, but HOW confident to be and which
alternatives are plausible.

Loss = alpha * CrossEntropy(student vs labels)   <- learn correct answers
     + (1-alpha) * KLDivergence(student vs teacher) <- mimic teacher
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from decoder_model import DecoderForGenerationTask, DECODER_CONFIGS

VARIANT = "distilled_mini"
OUTPUT_PATH = "trained_distilled_mini.pt"
STUDENT_CONFIG = {
    "d_model": 16, "n_heads": 2, "n_layers": 1,
    "d_ff": 32, "max_len": 32
}

TEACHER_PATH = "trained_decoder_medium.pt"
ALPHA = 0.5          # balance between learning from labels vs teacher
TEMPERATURE = 2.0    # softens teacher's probabilities, making them
                     # more informative for the student to learn from
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


def distillation_loss(student_logits, teacher_logits, labels,
                      pad_id, alpha, temperature):
    """
    Combines two losses:
    1. Hard loss: student vs correct labels (standard cross-entropy)
    2. Soft loss: student vs teacher probability distributions (KL divergence)
       Temperature > 1.0 softens both distributions, making subtle
       probability differences more visible to the student.
    """
    # Hard loss -- learn correct answers
    hard_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
    )

    # Soft loss -- mimic teacher
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(
        student_soft.reshape(-1, student_logits.size(-1)),
        teacher_soft.reshape(-1, teacher_logits.size(-1)),
        reduction="batchmean",
    ) * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CP1: [{VARIANT}] using device: {device}", flush=True)

    tokenizer = SimpleTokenizer(ALL_VOCAB)

    # Load teacher (frozen -- we never update its weights)
    teacher_ckpt = torch.load(TEACHER_PATH, map_location=device)
    teacher = DecoderForGenerationTask(
        vocab_size=teacher_ckpt["vocab_size"],
        d_model=teacher_ckpt["d_model"],
        n_heads=teacher_ckpt["n_heads"],
        n_layers=teacher_ckpt["n_layers"],
        d_ff=teacher_ckpt["d_ff"],
        max_len=teacher_ckpt["max_len"],
        pad_id=tokenizer.pad_id,
    ).to(device)
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"CP2: teacher loaded and frozen ({teacher.count_parameters():,} params)", flush=True)

    # Build student
    student = DecoderForGenerationTask(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        **STUDENT_CONFIG,
    ).to(device)
    print(f"CP3: [{VARIANT}] student built ({student.count_parameters():,} params)", flush=True)

    examples = load_generation_examples(DATASET_PATH)
    dataset = GenerationDataset(examples, tokenizer, max_len=MAX_LEN)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )
    print(f"CP4: {len(dataset)} examples loaded", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)

    print(f"CP5: starting distillation training [{VARIANT}]", flush=True)
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_batches = 0
        student.train()

        for src_ids, tgt_ids in loader:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            tgt_input = tgt_ids[:, :-1]
            tgt_labels = tgt_ids[:, 1:]

            # Get teacher predictions (no gradient needed)
            with torch.no_grad():
                teacher_logits = teacher(src_ids, tgt_input)

            # Get student predictions
            student_logits = student(src_ids, tgt_input)

            loss = distillation_loss(
                student_logits, teacher_logits, tgt_labels,
                tokenizer.pad_id, ALPHA, TEMPERATURE,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss: {avg_loss:.4f}", flush=True)

    print(f"CP6: distillation finished [{VARIANT}]", flush=True)

    torch.save({
        "model_state_dict": student.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "max_len": MAX_LEN,
        "variant": VARIANT,
        **STUDENT_CONFIG,
    }, OUTPUT_PATH)
    print(f"CP7: saved to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()