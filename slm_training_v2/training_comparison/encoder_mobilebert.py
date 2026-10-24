"""
MobileBERT-style Encoder, implemented from scratch in raw PyTorch.

Based on: "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited
Devices" (Sun et al., 2020)
Paper:  https://arxiv.org/abs/2004.02984
GitHub: https://github.com/google-research/google-research/tree/master/
        mobilebert
PyTorch: https://github.com/lonePatient/MobileBert_PyTorch

KEY ARCHITECTURAL DIFFERENCE from DistilBERT and TinyBERT:
MobileBERT uses a BOTTLENECK structure around each transformer block.

The bottleneck works like this:
  1. Input (hidden_size) -> Linear DOWN to intra_size (narrow bottleneck)
  2. Multi-Head Attention operates on the narrow intra_size representation
  3. Linear UP back to hidden_size
  4. Feed-Forward Network on hidden_size
  5. Residual connection from original hidden_size input

This narrow-wide-narrow design is the core innovation of MobileBERT --
it lets the model maintain a large inter-block hidden size (for capacity)
while keeping the per-block computation cheap (small intra_size attention).
In the original paper: inter_size=512, intra_size=128.

Additionally, MobileBERT uses multiple stacked FFN sub-layers per block
(the paper uses 4 FFN stacks per block). We use 2 FFN stacks scaled down.

Original MobileBERT:
  24 layers, inter_size=512, intra_size=128, heads=4, ffn_stacks=4 (~25M)

Scaled-down configuration for our task:
  2 layers, inter_size=64, intra_size=32, heads=4, ffn_stacks=2 (~55K params)

Task: RESTRUCTURE
  Input:  scrambled gesture tokens  e.g. "drive mother"
  Output: correctly ordered tokens  e.g. "mother drive"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileBertEmbeddings(nn.Module):
    """
    MobileBERT uses a 3-component embedding:
    word + position + token_type (unlike DistilBERT which drops token_type).
    The embedding output is projected through a linear layer (embedding
    bottleneck) before entering the transformer blocks.
    """
    def __init__(self, vocab_size, inter_size, max_len, pad_id, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, inter_size, padding_idx=pad_id
        )
        self.position_embeddings = nn.Embedding(max_len, inter_size)
        self.layer_norm = nn.LayerNorm(inter_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "position_ids",
            torch.arange(max_len).expand((1, -1))
        )

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_len]
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
        )
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BottleneckAttention(nn.Module):
    """
    MobileBERT's bottleneck attention module.

    The bottleneck structure:
      inter_size (wide) -> Linear DOWN -> intra_size (narrow)
      -> Multi-Head Self-Attention on intra_size
      -> Linear UP -> inter_size (wide)

    This is the defining structural innovation of MobileBERT.
    Q, K, V projections operate on the narrow intra_size, then output
    is projected back up to inter_size for the residual connection.
    """
    def __init__(self, inter_size, intra_size, n_heads, dropout=0.1):
        super().__init__()
        assert intra_size % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = intra_size // n_heads
        self.intra_size = intra_size

        # Bottleneck: wide -> narrow -> wide
        self.down_proj = nn.Linear(inter_size, intra_size)
        self.up_proj = nn.Linear(intra_size, inter_size)

        # Attention operates on narrow intra_size
        self.q_lin = nn.Linear(intra_size, intra_size)
        self.k_lin = nn.Linear(intra_size, intra_size)
        self.v_lin = nn.Linear(intra_size, intra_size)
        self.out_lin = nn.Linear(intra_size, intra_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, S, _ = x.shape

        # DOWN: wide inter_size -> narrow intra_size
        h = self.down_proj(x)

        # Multi-head attention on narrow representation
        q = self.q_lin(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, S, self.intra_size)
        context = self.out_lin(context)

        # UP: narrow intra_size -> wide inter_size
        return self.up_proj(context)


class MobileBertBlock(nn.Module):
    """
    One MobileBERT transformer block.

    Structure:
      1. Bottleneck Multi-Head Attention (narrow attention, wide residual)
      2. Add & Norm
      3. Multiple stacked FFN sub-layers (paper uses 4, we use 2)
      4. Add & Norm after each FFN stack

    The multiple FFN stacks per block is another MobileBERT distinction --
    this compensates for the narrower attention by doing more feed-forward
    processing at full inter_size width.
    """
    def __init__(self, inter_size, intra_size, n_heads,
                 ffn_size, n_ffn_stacks=2, dropout=0.1):
        super().__init__()
        self.attention = BottleneckAttention(
            inter_size, intra_size, n_heads, dropout
        )
        self.norm_attn = nn.LayerNorm(inter_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        # Multiple stacked FFN sublayers (MobileBERT uses 4, we use 2)
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inter_size, ffn_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_size, inter_size),
            )
            for _ in range(n_ffn_stacks)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(inter_size, eps=1e-12)
            for _ in range(n_ffn_stacks)
        ])

    def forward(self, x, key_padding_mask=None):
        # Bottleneck attention with residual
        attn_out = self.attention(x, key_padding_mask)
        x = self.norm_attn(x + self.dropout(attn_out))

        # Multiple FFN stacks with residual + norm after each
        for ffn, norm in zip(self.ffn_layers, self.ffn_norms):
            x = norm(x + self.dropout(ffn(x)))

        return x


class MobileBertForRestructure(nn.Module):
    """Full MobileBERT-style model for the Restructure task."""
    def __init__(self, vocab_size, inter_size=64, intra_size=32,
                 n_heads=4, n_layers=2, ffn_size=128,
                 n_ffn_stacks=2, max_len=16, pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        self.embeddings = MobileBertEmbeddings(
            vocab_size, inter_size, max_len, pad_id, dropout
        )
        self.layers = nn.ModuleList([
            MobileBertBlock(
                inter_size, intra_size, n_heads,
                ffn_size, n_ffn_stacks, dropout
            )
            for _ in range(n_layers)
        ])
        self.position_head = nn.Linear(inter_size, max_len)

    def forward(self, input_ids):
        padding_mask = (input_ids == self.pad_id)
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)
        logits = self.position_head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


MOBILEBERT_CONFIG = {
    "inter_size": 64,
    "intra_size": 32,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_size": 128,
    "n_ffn_stacks": 2,
    "max_len": 16,
    "dropout": 0.1,
}


if __name__ == "__main__":
    from tokenizer import SimpleTokenizer
    from vocab_pool import ALL_VOCAB

    tokenizer = SimpleTokenizer(ALL_VOCAB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = MobileBertForRestructure(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        **MOBILEBERT_CONFIG,
    ).to(device)

    print(f"Architecture: MobileBERT-style Encoder")
    print(f"Config: {MOBILEBERT_CONFIG}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    sample = "drive mother"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    x = torch.tensor([ids]).to(device)

    with torch.no_grad():
        out = model(x)
    print(f"\nSample input: {sample!r} -> token IDs {ids}")
    print(f"Output shape: {out.shape}  (batch=1, seq_len={len(ids)}, max_len=16)")
    print("Forward pass successful.")