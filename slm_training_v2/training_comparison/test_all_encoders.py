"""Test all three encoder architectures on the Restructure task."""
import torch
from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from encoder_distilbert import DistilBertForRestructure
from encoder_tinybert import TinyBertForRestructure
from encoder_mobilebert import MobileBertForRestructure


def predict(model, tokenizer, scrambled_text, device):
    input_tokens = tokenizer._clean_and_split(scrambled_text)
    ids = tokenizer.encode(scrambled_text, add_special_tokens=False)
    input_tensor = torch.tensor([ids]).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
    predicted_positions = logits[0].argmax(dim=-1).tolist()
    output_slots = {}
    for token, slot in zip(input_tokens, predicted_positions):
        output_slots[slot] = token
    return " ".join(output_slots[k] for k in sorted(output_slots.keys()))


def run_tests(model, tokenizer, variant, device):
    print(f"\n=== {variant.upper()} ===")
    seen = [
        ("drive mother",        "mother drive"),
        ("home where mother",   "where mother home"),
        ("fine how",            "how fine"),
        ("laundry father",      "father laundry"),
        ("home father where",   "where father home"),
    ]
    unseen = [
        ("home drive father",   "father drive home"),
        ("laundry mother home", "mother laundry home"),
        ("father how fine",     "how father fine"),
    ]
    print("  --- Seen during training ---")
    for scrambled, expected in seen:
        pred = predict(model, tokenizer, scrambled, device)
        status = "PASS" if pred == expected else "FAIL"
        print(f"  [{status}] {scrambled!r:25s} -> {pred!r}")

    print("  --- Unseen combinations ---")
    for scrambled, expected in unseen:
        pred = predict(model, tokenizer, scrambled, device)
        status = "PASS" if pred == expected else "FAIL"
        print(f"  [{status}] {scrambled!r:25s} -> {pred!r}")
        if status == "FAIL":
            print(f"          expected: {expected!r}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = SimpleTokenizer(ALL_VOCAB)

    models = [
        ("distilbert", "trained_distilbert.pt",
         DistilBertForRestructure),
        ("tinybert",   "trained_tinybert.pt",
         TinyBertForRestructure),
        ("mobilebert", "trained_mobilebert.pt",
         MobileBertForRestructure),
    ]

    for variant, path, ModelClass in models:
        ckpt = torch.load(path, map_location=device)
        # Remove keys not needed for model init
        cfg = {k: v for k, v in ckpt.items()
               if k not in ("model_state_dict", "vocab_size", "variant")}
        model = ModelClass(
            vocab_size=ckpt["vocab_size"],
            pad_id=tokenizer.pad_id,
            **cfg,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        run_tests(model, tokenizer, variant, device)


if __name__ == "__main__":
    main()