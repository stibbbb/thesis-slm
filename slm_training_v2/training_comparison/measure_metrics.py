"""
Metrics measurement script for the SLM thesis evaluation (v3).

Measures, for all 6 trained architectures:
  1. Exact Match (EM) - fraction of HELD-OUT cases where output == reference
  2. BLEU             - n-gram overlap, n-gram order = min(4, reference length)
  3. Perplexity       - exp(cross-entropy) on the HELD-OUT set
  4. Latency          - avg milliseconds per inference over 50 runs
  5. File Size        - .pt file size in MB

ALL metrics are now computed on dataset_v3_test.jsonl -- held-out
phrasings that NO model saw during training. All six models were
trained on dataset_v3_train.jsonl, so this test set is fair to all of
them, and the comparison measures GENERALIZATION rather than
memorization. This is a stronger comparison than the original, which
scored EM/BLEU on a handful of hardcoded cases drawn from the training
distribution.

NOTE ON EM: content-level exact match, punctuation-insensitive (the
tokenizer produces no punctuation, so comparing against punctuated
references would otherwise cause false misses).

NOTE ON BLEU: standard BLEU-4 unfairly punishes 2-3 word sentences
because they contain no 3-grams or 4-grams. We match the n-gram order
to the reference length instead, and report Exact Match alongside.

NOTE ON RESTRUCTURE SAMPLE SIZE: the held-out set contains only a
small number of restructure rows, so the encoder EM is a thin sample.
It is reported honestly as-is rather than padded.

Run on GPU:  python measure_metrics.py
Run on CPU:  python measure_metrics.py --cpu
"""

import os
import sys
import time
import json
import math
import re
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from encoder_distilbert import DistilBertForRestructure
from encoder_tinybert import TinyBertForRestructure
from encoder_mobilebert import MobileBertForRestructure
from decoder_tinyllama import TinyLlamaForGeneration
from decoder_babyllama import BabyLlamaForGeneration
from decoder_mobillama import MobiLlamaForGeneration


def _normalize(text):
    """Content-level comparison: lowercase, strip punctuation anywhere,
    collapse whitespace. Matches how the models actually emit text."""
    text = text.lower().strip()
    text = re.sub(r"[.?!,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_bleu(reference, hypothesis):
    """BLEU with n-gram order matched to reference length (max 4)."""
    ref_tokens = _normalize(reference).split()
    hyp_tokens = _normalize(hypothesis).split()
    if not ref_tokens:
        return 0.0
    n = max(1, min(4, len(ref_tokens)))
    weights = tuple([1.0 / n] * n)
    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens,
                         weights=weights, smoothing_function=smoothing)


def exact_match(reference, hypothesis):
    return _normalize(reference) == _normalize(hypothesis)


def compute_encoder_perplexity(model, tokenizer, examples, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    n = 0
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
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                label_tensor.reshape(-1)
            )
            total_loss += loss.item()
            n += 1
    avg_loss = total_loss / n if n > 0 else float("inf")
    return math.exp(avg_loss)


def compute_decoder_perplexity(model, tokenizer, examples, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for input_text, output_text in examples:
            src_ids = tokenizer.encode(input_text, add_special_tokens=False)
            tgt_ids = tokenizer.encode(output_text, add_special_tokens=True)
            src = torch.tensor([src_ids]).to(device)
            tgt = torch.tensor([tgt_ids]).to(device)
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            logits = model(src, tgt_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.reshape(-1)
            )
            total_loss += loss.item()
            n += 1
    avg_loss = total_loss / n if n > 0 else float("inf")
    return math.exp(avg_loss)


def measure_encoder_latency(model, tokenizer, test_input, device, n_runs=50):
    ids = tokenizer.encode(test_input, add_special_tokens=False)
    input_tensor = torch.tensor([ids]).to(device)
    model.eval()
    with torch.no_grad():
        model(input_tensor)  # warmup
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    return sum(times) / len(times)


def measure_decoder_latency(model, tokenizer, test_input, device, n_runs=50):
    ids = tokenizer.encode(test_input, add_special_tokens=False)
    src = torch.tensor([ids]).to(device)
    model.eval()
    model.generate(src, tokenizer, max_new_tokens=16)  # warmup
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.generate(src, tokenizer, max_new_tokens=16)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return sum(times) / len(times)


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def predict_encoder(model, tokenizer, scrambled_text, device):
    from slm_inference import restructure
    return restructure(model, tokenizer, scrambled_text, device)


def main():
    if "--cpu" in sys.argv:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    print(f"Device: {device}")
    print(f"{'='*76}")

    # All metrics measured on the HELD-OUT test set -- phrasings no
    # model saw in training. encoder_cases / decoder_cases are now
    # built FROM this data, not hardcoded, so EM/BLEU measure
    # generalization consistently with perplexity.
    restructure_examples = []
    generation_examples = []
    with open("dataset_v3_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task"] == "restructure":
                restructure_examples.append((row["input"], row["output"]))
            elif row["task"] in ("complete", "reduce"):
                generation_examples.append((row["input"], row["output"]))

    encoder_cases = restructure_examples
    decoder_cases = generation_examples

    print(f"Held-out test cases: {len(encoder_cases)} restructure, "
          f"{len(decoder_cases)} generation (complete+reduce)")

    header = (f"{'Architecture':<15} {'Params':>10} {'EM':>7} {'BLEU':>8} "
              f"{'PPL(held)':>10} {'Latency(ms)':>12} {'Size(MB)':>9}")

    # ----------------- ENCODERS -----------------
    print("\nENCODER ARCHITECTURES (Restructure Task)")
    print(header)
    print("-" * 76)

    encoder_models = [
        ("DistilBERT", "trained_distilbert.pt", DistilBertForRestructure),
        ("TinyBERT",   "trained_tinybert.pt",   TinyBertForRestructure),
        ("MobileBERT", "trained_mobilebert.pt", MobileBertForRestructure),
    ]

    encoder_misses = []
    for name, path, ModelClass in encoder_models:
        ckpt = torch.load(path, map_location=device)
        cfg = {k: v for k, v in ckpt.items()
               if k not in ("model_state_dict", "vocab_size", "variant")}
        model = ModelClass(
            vocab_size=ckpt["vocab_size"],
            pad_id=tokenizer.pad_id,
            **cfg,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        em_count = 0
        bleu_scores = []
        for inp, ref in encoder_cases:
            hyp = predict_encoder(model, tokenizer, inp, device)
            bleu_scores.append(compute_bleu(ref, hyp))
            if exact_match(ref, hyp):
                em_count += 1
            else:
                encoder_misses.append((name, inp, hyp, ref))
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        em_str = f"{em_count}/{len(encoder_cases)}"

        ppl = compute_encoder_perplexity(
            model, tokenizer, restructure_examples, device)
        lat = measure_encoder_latency(
            model, tokenizer, "drive mother where", device)
        size_mb = get_file_size_mb(path)

        print(f"{name:<15} {params:>10,} {em_str:>7} {avg_bleu:>8.4f} "
              f"{ppl:>10.4f} {lat:>12.2f} {size_mb:>9.3f}")

    if encoder_misses:
        print(f"\n  Misses ({len(encoder_misses)} of "
              f"{len(encoder_cases)*len(encoder_models)} total predictions):")
        for name, inp, hyp, ref in encoder_misses:
            print(f"    [{name}] {inp!r} -> {hyp!r}  (expected {ref!r})")

    # ----------------- DECODERS -----------------
    print(f"\n\nDECODER ARCHITECTURES (Complete + Reduce Tasks)")
    print(header)
    print("-" * 76)

    decoder_models = [
        ("TinyLlama", "trained_tinyllama.pt", TinyLlamaForGeneration),
        ("BabyLlama", "trained_babyllama.pt", BabyLlamaForGeneration),
        ("MobiLlama", "trained_mobillama.pt", MobiLlamaForGeneration),
    ]

    decoder_misses = []
    for name, path, ModelClass in decoder_models:
        ckpt = torch.load(path, map_location=device)
        cfg = {k: v for k, v in ckpt.items()
               if k not in ("model_state_dict", "vocab_size", "variant")}
        model = ModelClass(
            vocab_size=ckpt["vocab_size"],
            pad_id=tokenizer.pad_id,
            **cfg,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        em_count = 0
        bleu_scores = []
        for inp, ref in decoder_cases:
            ids = tokenizer.encode(inp, add_special_tokens=False)
            src = torch.tensor([ids]).to(device)
            hyp = model.generate(src, tokenizer, max_new_tokens=16)
            bleu_scores.append(compute_bleu(ref, hyp))
            if exact_match(ref, hyp):
                em_count += 1
            else:
                decoder_misses.append((name, inp, hyp, ref))
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        em_str = f"{em_count}/{len(decoder_cases)}"

        ppl = compute_decoder_perplexity(
            model, tokenizer, generation_examples, device)
        lat = measure_decoder_latency(
            model, tokenizer, "mother laundry", device)
        size_mb = get_file_size_mb(path)

        print(f"{name:<15} {params:>10,} {em_str:>7} {avg_bleu:>8.4f} "
              f"{ppl:>10.4f} {lat:>12.2f} {size_mb:>9.3f}")

    if decoder_misses:
        print(f"\n  Misses ({len(decoder_misses)} of "
              f"{len(decoder_cases)*len(decoder_models)} total predictions):")
        for name, inp, hyp, ref in decoder_misses:
            print(f"    [{name}] {inp!r} -> {hyp!r}  (expected {ref!r})")

    print(f"\n{'='*76}")
    print("Notes:")
    print("  EM:        content exact matches / total HELD-OUT cases")
    print("  BLEU:      n-gram order matched to reference length (max 4)")
    print("  PPL(held): perplexity on HELD-OUT phrasings no model saw in")
    print("             training -- the honest generalization signal, lower")
    print("             is better; cannot be inflated by memorization")
    print(f"  Latency:   avg of 50 runs on this device: {device} (NOT the Pi;")
    print("             indicative only until measured on target hardware)")
    print("  Size:      .pt file size in megabytes")


if __name__ == "__main__":
    main()