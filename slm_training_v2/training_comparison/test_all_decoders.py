"""Test all three decoder architectures on Complete and Reduce tasks."""
import torch
from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from decoder_tinyllama import TinyLlamaForGeneration
from decoder_babyllama import BabyLlamaForGeneration
from decoder_mobillama import MobiLlamaForGeneration


def run_tests(model, tokenizer, variant, device):
    print(f"\n=== {variant.upper()} ===")

    complete_tests = [
        ("mother laundry",       "Mother is doing laundry."),
        ("where mother",         "Where is mother?"),
        ("how fine",             "How are you? I am fine."),
        ("father drive home",    "Father is driving home."),
    ]
    reduce_tests = [
        ("Where is mother? Is she home?",      "where mother home"),
        ("Father is driving.",                  "father drive"),
        ("How are you? I am fine.",             "how fine"),
        ("Mother is doing laundry at home.",    "mother laundry home"),
    ]

    print("  --- COMPLETE task ---")
    for input_text, expected in complete_tests:
        ids = tokenizer.encode(input_text, add_special_tokens=False)
        src = torch.tensor([ids]).to(device)
        output = model.generate(src, tokenizer, max_new_tokens=16)
        status = "PASS" if output.lower() == expected.lower() else "~   "
        print(f"  [{status}] {input_text!r:25s} -> {output!r}")
        if status != "PASS":
            print(f"          expected: {expected!r}")

    print("  --- REDUCE task ---")
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

    models = [
        ("tinyllama",  "trained_tinyllama.pt",  TinyLlamaForGeneration),
        ("babyllama",  "trained_babyllama.pt",  BabyLlamaForGeneration),
        ("mobillama",  "trained_mobillama.pt",  MobiLlamaForGeneration),
    ]

    for variant, path, ModelClass in models:
        ckpt = torch.load(path, map_location=device)
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