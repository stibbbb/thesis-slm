import torch
import numpy as np
import onnxruntime as ort

from tokenizer import SimpleTokenizer
from vocab_pool import ALL_VOCAB
from encoder_distilbert import DistilBertForRestructure, DISTILBERT_CONFIG
from decoder_tinyllama import TinyLlamaForGeneration, TINYLLAMA_CONFIG
from onnx_config import (
    ENCODER_ONNX_PATH, DECODER_ONNX_PATH,
    ENCODER_MAX_LEN, DECODER_SRC_LEN, DECODER_TGT_LEN,
    pad_ids,
)

ENCODER_CKPT = "trained_distilbert.pt"
DECODER_CKPT = "trained_tinyllama.pt"

# Sanity check: onnx_config's fixed lengths must match the actual trained
# model configs, or export would silently build a wrong-shaped graph.
assert ENCODER_MAX_LEN == DISTILBERT_CONFIG["max_len"], \
    "onnx_config.ENCODER_MAX_LEN out of sync with DISTILBERT_CONFIG"
assert DECODER_TGT_LEN == TINYLLAMA_CONFIG["max_len"], \
    "onnx_config.DECODER_TGT_LEN out of sync with TINYLLAMA_CONFIG"


def load_encoder(device="cpu"):
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    ckpt = torch.load(ENCODER_CKPT, map_location=device)
    cfg = {k: v for k, v in ckpt.items()
           if k not in ("model_state_dict", "vocab_size", "variant")}
    model = DistilBertForRestructure(
        vocab_size=ckpt["vocab_size"], pad_id=tokenizer.pad_id, **cfg
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


def load_decoder(device="cpu"):
    tokenizer = SimpleTokenizer(ALL_VOCAB)
    ckpt = torch.load(DECODER_CKPT, map_location=device)
    cfg = {k: v for k, v in ckpt.items()
           if k not in ("model_state_dict", "vocab_size", "variant")}
    model = TinyLlamaForGeneration(
        vocab_size=ckpt["vocab_size"], pad_id=tokenizer.pad_id, **cfg
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


def export_encoder():
    print("=== Exporting encoder (DistilBERT) ===")
    model, tokenizer = load_encoder()

    sample_ids = tokenizer.encode("drive mother where", add_special_tokens=False)
    padded = pad_ids(sample_ids, ENCODER_MAX_LEN, tokenizer.pad_id)
    dummy_input = torch.tensor([padded], dtype=torch.long)

    torch.onnx.export(
        model,
        dummy_input,
        ENCODER_ONNX_PATH,
        input_names=["input_ids"],
        output_names=["position_logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Saved: {ENCODER_ONNX_PATH}")
    return model, tokenizer, sample_ids, padded


def export_decoder():
    print("=== Exporting decoder (TinyLlama) ===")
    model, tokenizer = load_decoder()

    src_ids = tokenizer.encode("mother laundry", add_special_tokens=False)
    src_padded = pad_ids(src_ids, DECODER_SRC_LEN, tokenizer.pad_id)

    tgt_ids = [tokenizer.bos_id] + tokenizer.encode("mother is doing", add_special_tokens=False)
    tgt_padded = pad_ids(tgt_ids, DECODER_TGT_LEN, tokenizer.pad_id)

    dummy_src = torch.tensor([src_padded], dtype=torch.long)
    dummy_tgt = torch.tensor([tgt_padded], dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_src, dummy_tgt),
        DECODER_ONNX_PATH,
        input_names=["input_ids", "target_ids"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Saved: {DECODER_ONNX_PATH}")
    return model, tokenizer, src_ids, src_padded, tgt_padded


def check_encoder_parity(model, tokenizer, sample_ids, padded):
    print("\n=== Encoder parity check (PyTorch vs ONNX Runtime) ===")
    n = len(sample_ids)
    input_tensor = torch.tensor([padded], dtype=torch.long)
    with torch.no_grad():
        torch_logits = model(input_tensor).numpy()

    session = ort.InferenceSession(ENCODER_ONNX_PATH)
    onnx_logits = session.run(
        None, {"input_ids": np.array([padded], dtype=np.int64)}
    )[0]

    diff = np.abs(torch_logits[:, :n, :n] - onnx_logits[:, :n, :n]).max()
    print(f"  Max abs difference on real tokens: {diff:.8f}")
    status = "PASS" if diff < 1e-4 else "FAIL"
    print(f"  [{status}] (threshold 1e-4)")
    return diff


def check_decoder_parity(model, tokenizer, src_ids, src_padded, tgt_padded):
    print("\n=== Decoder parity check (PyTorch vs ONNX Runtime) ===")
    n_tgt = len([t for t in tgt_padded if t != tokenizer.pad_id])
    src_tensor = torch.tensor([src_padded], dtype=torch.long)
    tgt_tensor = torch.tensor([tgt_padded], dtype=torch.long)
    with torch.no_grad():
        torch_logits = model(src_tensor, tgt_tensor).numpy()

    session = ort.InferenceSession(DECODER_ONNX_PATH)
    onnx_logits = session.run(
        None,
        {
            "input_ids": np.array([src_padded], dtype=np.int64),
            "target_ids": np.array([tgt_padded], dtype=np.int64),
        },
    )[0]

    diff = np.abs(torch_logits[:, :n_tgt, :] - onnx_logits[:, :n_tgt, :]).max()
    print(f"  Max abs difference on real tokens: {diff:.8f}")
    status = "PASS" if diff < 1e-4 else "FAIL"
    print(f"  [{status}] (threshold 1e-4)")
    return diff


def main():
    enc_model, enc_tok, enc_sample_ids, enc_padded = export_encoder()
    dec_model, dec_tok, dec_src_ids, dec_src_padded, dec_tgt_padded = export_decoder()

    check_encoder_parity(enc_model, enc_tok, enc_sample_ids, enc_padded)
    check_decoder_parity(dec_model, dec_tok, dec_src_ids, dec_src_padded, dec_tgt_padded)

    print("\nDone. If both checks show PASS, proceed to onnx_inference.py")
    print("for the full pipeline demo backed by ONNX Runtime.")


if __name__ == "__main__":
    main()