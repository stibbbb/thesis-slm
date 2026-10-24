"""
Decoder-only architecture, built from scratch with raw PyTorch.

Tasks: COMPLETE and REDUCE
  COMPLETE: ordered gesture tokens -> full natural sentence
    e.g. "mother laundry" -> "mother is doing laundry."
  REDUCE:   full natural sentence  -> gesture keywords only
    e.g. "Where is mother? Is she home?" -> "where mother home"

A Decoder generates text one word at a time, left to right.
Given a starting prompt, it predicts the next word, appends it,
then predicts the word after that, and so on until it outputs <eos>.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DecoderForGenerationTask(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, max_len=32, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, vocab_size)

    def make_causal_mask(self, seq_len, device):
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(self, input_ids, target_ids):
        src_padding_mask = input_ids == self.pad_id
        src_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)

        tgt_seq_len = target_ids.size(1)
        causal_mask = self.make_causal_mask(tgt_seq_len, target_ids.device)
        tgt_padding_mask = target_ids == self.pad_id
        tgt_emb = self.token_embedding(target_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)

        decoded = self.decoder(
            tgt=tgt_emb,
            memory=src_emb,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        logits = self.output_head(decoded)
        return logits

    def generate(self, input_ids, tokenizer, max_new_tokens=16):
        self.eval()
        device = input_ids.device
        generated = torch.tensor([[tokenizer.bos_id]], device=device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids, generated)
                next_token = logits[0, -1, :].argmax().item()
                if next_token == tokenizer.eos_id:
                    break
                generated = torch.cat([
                    generated,
                    torch.tensor([[next_token]], device=device)
                ], dim=1)

        output_ids = generated[0, 1:].tolist()
        return tokenizer.decode(output_ids, skip_special_tokens=True)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


DECODER_CONFIGS = {
    "small":  {"d_model": 32,  "n_heads": 2, "n_layers": 1, "d_ff": 64,  "max_len": 32},
    "medium": {"d_model": 64,  "n_heads": 4, "n_layers": 2, "d_ff": 128, "max_len": 32},
    "large":  {"d_model": 128, "n_heads": 8, "n_layers": 4, "d_ff": 256, "max_len": 32},
}


if __name__ == "__main__":
    from tokenizer import SimpleTokenizer
    from vocab_pool import ALL_VOCAB

    tokenizer = SimpleTokenizer(ALL_VOCAB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    for variant, cfg in DECODER_CONFIGS.items():
        model = DecoderForGenerationTask(
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_id,
            **cfg,
        ).to(device)
        n = model.count_parameters()
        print(f"Decoder {variant:6s}: d_model={cfg['d_model']:3d}, "
              f"n_layers={cfg['n_layers']}, n_heads={cfg['n_heads']:2d}, "
              f"d_ff={cfg['d_ff']:3d}  ->  {n:>8,} trainable params")