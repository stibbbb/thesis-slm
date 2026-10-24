"""
Test the trained EncoderForRestructure model.
Loads the saved weights and runs inference on several examples,
both seen during training and new combinations not in the dataset.
"""

import torch
from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from encoder_model_small import EncoderForRestructure

MAX_LEN = 16
MODEL_PATH = "trained_encoder_small.pt"


def predict(model, tokenizer, scrambled_text, device):
    """Takes scrambled gesture tokens, returns them in predicted order."""
    input_tokens = tokenizer._clean_and_split(scrambled_text)
    ids = tokenizer.encode(scrambled_text, add_special_tokens=False)
    input_tensor = torch.tensor([ids]).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)   # (1, seq_len, max_len)

    # For each input token, pick the output slot with the highest score
    predicted_positions = logits[0].argmax(dim=-1).tolist()

    # Reconstruct output by placing each input token at its predicted slot
    output_slots = {}
    for token, slot in zip(input_tokens, predicted_positions):
        output_slots[slot] = token

    # Sort by slot index to get the final ordered output
    ordered = [output_slots[k] for k in sorted(output_slots.keys())]
    return " ".join(ordered)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = SimpleTokenizer(ALL_VOCAB)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = EncoderForRestructure(
    vocab_size=checkpoint["vocab_size"],
    d_model=32,
    n_heads=2,
    n_layers=1,
    d_ff=64,
    max_len=checkpoint["max_len"],
    pad_id=tokenizer.pad_id,
).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {MODEL_PATH}\n")

    # --- Seen during training ---
    print("=== Examples seen during training ===")
    seen_tests = [
        ("drive mother",          "mother drive"),
        ("home where mother",     "where mother home"),
        ("fine how",            "how fine"),
        ("laundry father",        "father laundry"),
        ("home father where",     "where father home"),
    ]
    for scrambled, expected in seen_tests:
        predicted = predict(model, tokenizer, scrambled, device)
        status = "PASS" if predicted == expected else "FAIL"
        print(f"  [{status}] Input: {scrambled!r:25s} -> {predicted!r}  (expected: {expected!r})")

    print()

    # --- Unseen combinations (generalization test) ---
    print("=== Unseen combinations (generalization) ===")
    unseen_tests = [
        ("home drive father",     "father drive home"),
        ("laundry mother home",   "mother laundry home"),
        ("father how fine",       "how father fine"),
    ]
    for scrambled, expected in unseen_tests:
        predicted = predict(model, tokenizer, scrambled, device)
        status = "PASS" if predicted == expected else "FAIL"
        print(f"  [{status}] Input: {scrambled!r:25s} -> {predicted!r}  (expected: {expected!r})")


if __name__ == "__main__":
    main()