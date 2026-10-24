"""
Test all three Decoder variants on complete and reduce tasks.
"""

import torch
from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from decoder_model import DecoderForGenerationTask

MAX_LEN = 32


def load_model(path, tokenizer, device):
    checkpoint = torch.load(path, map_location=device)
    model = DecoderForGenerationTask(
        vocab_size=checkpoint["vocab_size"],
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        d_ff=checkpoint["d_ff"],
        max_len=checkpoint["max_len"],
        pad_id=tokenizer.pad_id,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["variant"]


def run_tests(model, tokenizer, variant, device):
    print(f"\n=== Decoder {variant.upper()} ===")

    complete_tests = [
        ("mother laundry",     "mother is doing laundry."),
        ("where mother",       "Where is mother?"),
        ("how fine",         "How are you? I am fine."),
        ("father drive home",  "father is driving home."),
    ]

    reduce_tests = [
        ("Where is mother? Is she home?",          "where mother home"),
        ("father is driving.",                      "father drive"),
        ("How are you? I am fine.",               "how fine"),
        ("mother is doing laundry at home.",        "mother laundry home"),
    ]

    print("  --- COMPLETE task (gesture tokens -> full sentence) ---")
    for input_text, expected in complete_tests:
        ids = tokenizer.encode(input_text, add_special_tokens=False)
        src = torch.tensor([ids]).to(device)
        output = model.generate(src, tokenizer, max_new_tokens=16)
        status = "PASS" if output.lower() == expected.lower() else "~   "
        print(f"  [{status}] {input_text!r:25s} -> {output!r}")
        if status != "PASS":
            print(f"          expected: {expected!r}")

    print("  --- REDUCE task (full sentence -> keywords) ---")
    for input_text, expected in reduce_tests:
        ids = tokenizer.encode(input_text, add_special_tokens=False)
        src = torch.tensor([ids]).to(device)
        output = model.generate(src, tokenizer, max_new_tokens=16)
        status = "PASS" if output.lower() == expected.lower() else "~   "
        print(f"  [{status}] {input_text!r:45s} -> {output!r}")
        if status != "PASS":
            print(f"          expected: {expected!r}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = SimpleTokenizer(ALL_VOCAB)

    for path in [
        "trained_decoder_small.pt",
        "trained_decoder_medium.pt",
        "trained_decoder_large.pt",
    ]:
        try:
            model, variant = load_model(path, tokenizer, device)
            run_tests(model, tokenizer, variant, device)
        except FileNotFoundError:
            print(f"\nSkipping {path} (not found)")


if __name__ == "__main__":
    main()