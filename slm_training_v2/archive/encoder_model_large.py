"""
Encoder-only architecture, built from scratch with raw PyTorch (no
pre-trained weights, no Hugging Face model downloads).

Task: RESTRUCTURE
  Input:  scrambled gesture tokens, e.g. [drive, mother]
  Output: correctly ordered tokens,   e.g. [mother, drive]

Framed as a position-prediction problem: for each input token, predict
which output slot (0, 1, 2, ...) it belongs in. This suits an
Encoder's strength (understanding relationships across a full,
bidirectionally-visible sequence) better than treating it as open-ended
text generation.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Adds information about word ORDER into the embeddings, since
    self-attention on its own has no inherent sense of sequence position
    -- without this, the model couldn't tell "mother drive" from "drive mother"."""

    def __init__(self, d_model, max_len=16):
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


class EncoderForRestructure(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, max_len=16, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: for each input position, predict which OUTPUT
        # slot (0 .. max_len-1) it should move to.
        self.position_head = nn.Linear(d_model, max_len)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        padding_mask = input_ids == self.pad_id  # True where padded

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        # encoded: (batch, seq_len, d_model)

        logits = self.position_head(encoded)  # (batch, seq_len, max_len)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from tokenizer import SimpleTokenizer
    from vocab_pool import ALL_VOCAB

    tokenizer = SimpleTokenizer(ALL_VOCAB)

    model = EncoderForRestructure(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_heads=8,
    n_layers=4,
    d_ff=256,
    max_len=16,
    pad_id=tokenizer.pad_id,
)

    n_params = model.count_parameters()
    print(f"EncoderForRestructure created.")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Total trainable parameters: {n_params:,}")

    # Quick forward-pass sanity check with dummy input
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device available: {device}")
    model = model.to(device)

    sample_text = "drive mother"
    ids = tokenizer.encode(sample_text, add_special_tokens=False)
    input_tensor = torch.tensor([ids]).to(device)
    print(f"\nSample input: {sample_text!r} -> token IDs {ids}")

    with torch.no_grad():
        output = model(input_tensor)
    print(f"Output logits shape: {output.shape}  (batch=1, seq_len={len(ids)}, max_len=16)")
    print("\nForward pass successful. Model is ready for training.")