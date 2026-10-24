"""
Held-out evaluation for the FINAL selected models (DistilBERT +
TinyLlama). Measures EM, BLEU, and perplexity on both TRAIN (expect
near-perfect -- confirms the model learned) and TEST (gloss
combinations never seen in training -- the honest generalization
number for Chapter 4).

CONTENT vs STRICT exact match: the tokenizer vocabulary has no
punctuation tokens, so the decoder can never produce a period,
question mark, or capital letter. Comparing raw generated text
against the raw reference sentence therefore penalizes the model for
a capability it was never given. ContentEM normalizes (lowercase,
strip .?!,) before comparing -- the fair, honest number. StrictEM is
reported alongside for transparency, since the gap between them is
itself evidence of the documented punctuation limitation.

Run: python measure_holdout.py [--cpu]
"""

import re
import sys
import json
import math
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from encoder_distilbert import DistilBertForRestructure
from decoder_tinyllama import TinyLlamaForGeneration

ENCODER_CKPT = "trained_distilbert.pt"
DECODER_CKPT = "trained_tinyllama.pt"
TRAIN_PATH = "dataset_train.jsonl"
TEST_PATH = "dataset_test.jsonl"


def normalize(text):
    """Lowercase and strip sentence punctuation for content-level
    comparison -- the actual output space the tokenizer can produce."""
    text = text.lower().strip()
    text = re.sub(r"[.?!,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_bleu(reference, hypothesis):
    ref_tokens = normalize(reference).split()
    hyp_tokens = normalize(hypothesis).split()
    n = max(1, min(4, len(ref_tokens)))
    weights = tuple([1.0 / n] * n)
    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smoothing)


def content_match(reference, hypothesis):
    return normalize(reference) == normalize(hypothesis)


def strict_match(reference, hypothesis):
    return reference.strip() == hypothesis.strip()


def load_rows(path, task):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task"] == task:
                rows.append((row["input"], row["output"]))
    return rows


def load_model(ckpt_path, ModelClass, tokenizer, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = {k: v for k, v in ckpt.items()
           if k not in ("model_state_dict", "vocab_size", "variant")}
    model = ModelClass(vocab_size=ckpt["vocab_size"], pad_id=tokenizer.pad_id, **cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def restructure(model, tokenizer, gloss_text, device):
    input_tokens = tokenizer._clean_and_split(gloss_text)
    ids = tokenizer.encode(gloss_text, add_special_tokens=False)
    n = min(len(input_tokens), len(ids))
    if n == 0:
        return ""
    if n == 1:
        return input_tokens[0]
    input_tensor = torch.tensor([ids[:n]]).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
    score = logits[0, :n, :n].cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-score)
    ordered = [None] * n
    for word_idx, pos in zip(row_ind, col_ind):
        ordered[pos] = input_tokens[word_idx]
    return " ".join(ordered)


def encoder_perplexity(model, tokenizer, examples, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for input_text, output_text in examples:
            input_tokens = tokenizer._clean_and_split(input_text)
            output_tokens = tokenizer._clean_and_split(output_text)
            if sorted(input_tokens) != sorted(output_tokens):
                continue
            output_word_to_pos = {}
            for pos, word in enumerate(output_tokens):
                output_word_to_pos.setdefault(word, []).append(pos)
            labels = []
            used = {word: 0 for word in output_word_to_pos}
            for word in input_tokens:
                positions = output_word_to_pos[word]
                labels.append(positions[used[word]])
                used[word] += 1
            ids = tokenizer.encode(input_text, add_special_tokens=False)
            input_tensor = torch.tensor([ids]).to(device)
            label_tensor = torch.tensor([labels]).to(device)
            logits = model(input_tensor)
            loss = criterion(logits.reshape(-1, logits.size(-1)), label_tensor.reshape(-1))
            total_loss += loss.item()
            n += 1
    return math.exp(total_loss / n) if n > 0 else float("inf")


def decoder_perplexity(model, tokenizer, examples, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for input_text, output_text in examples:
            src_ids = tokenizer.encode(input_text, add_special_tokens=False)
            tgt_ids = tokenizer.encode(output_text, add_special_tokens=True)
            src = torch.tensor([src_ids]).to(device)
            tgt = torch.tensor([tgt_ids]).to(device)
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            n += 1
    return math.exp(total_loss / n) if n > 0 else float("inf")


def evaluate_encoder(model, tokenizer, examples, device, label):
    if not examples:
        print(f"  [{label}] no examples -- skipped")
        return
    em, bleu_total, misses = 0, 0.0, []
    for inp, ref in examples:
        hyp = restructure(model, tokenizer, inp, device)
        bleu_total += compute_bleu(ref, hyp)
        if content_match(ref, hyp):
            em += 1
        else:
            misses.append((inp, hyp, ref))
    ppl = encoder_perplexity(model, tokenizer, examples, device)
    print(f"  [{label}] n={len(examples)}  EM={em}/{len(examples)}  "
          f"BLEU={bleu_total/len(examples):.4f}  PPL={ppl:.4f}")
    for inp, hyp, ref in misses:
        print(f"      MISS: {inp!r} -> {hyp!r}  (expected {ref!r})")


def evaluate_decoder(model, tokenizer, examples, device, label):
    if not examples:
        print(f"  [{label}] no examples -- skipped")
        return
    content_em, strict_em, bleu_total, misses = 0, 0, 0.0, []
    for inp, ref in examples:
        ids = tokenizer.encode(inp, add_special_tokens=False)
        src = torch.tensor([ids]).to(device)
        hyp = model.generate(src, tokenizer, max_new_tokens=16)
        bleu_total += compute_bleu(ref, hyp)
        if content_match(ref, hyp):
            content_em += 1
        else:
            misses.append((inp, hyp, ref))
        if strict_match(ref, hyp):
            strict_em += 1
    ppl = decoder_perplexity(model, tokenizer, examples, device)
    print(f"  [{label}] n={len(examples)}  ContentEM={content_em}/{len(examples)}  "
          f"StrictEM={strict_em}/{len(examples)}  "
          f"BLEU={bleu_total/len(examples):.4f}  PPL={ppl:.4f}")
    for inp, hyp, ref in misses:
        print(f"      MISS (content): {inp!r} -> {hyp!r}  (expected {ref!r})")


def main():
    device = "cpu" if "--cpu" in sys.argv else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    print(f"Device: {device}")

    encoder = load_model(ENCODER_CKPT, DistilBertForRestructure, tokenizer, device)
    decoder = load_model(DECODER_CKPT, TinyLlamaForGeneration, tokenizer, device)

    train_restructure = load_rows(TRAIN_PATH, "restructure")
    test_restructure = load_rows(TEST_PATH, "restructure")
    train_gen = load_rows(TRAIN_PATH, "complete") + load_rows(TRAIN_PATH, "reduce")
    test_gen = load_rows(TEST_PATH, "complete") + load_rows(TEST_PATH, "reduce")

    print("\n=== ENCODER (DistilBERT) ===")
    evaluate_encoder(encoder, tokenizer, train_restructure, device, "TRAIN (seen)")
    evaluate_encoder(encoder, tokenizer, test_restructure, device, "TEST  (held out)")

    print("\n=== DECODER (TinyLlama) ===")
    evaluate_decoder(decoder, tokenizer, train_gen, device, "TRAIN (seen)")
    evaluate_decoder(decoder, tokenizer, test_gen, device, "TEST  (held out)")

    print("\nContentEM = fair comparison (tokenizer has no punctuation tokens).")
    print("StrictEM  = character-exact; gap vs ContentEM is punctuation-only,")
    print("            not a content error -- report both for transparency.")
    print("TEST numbers are the honest generalization result for Chapter 4.")


if __name__ == "__main__":
    main()