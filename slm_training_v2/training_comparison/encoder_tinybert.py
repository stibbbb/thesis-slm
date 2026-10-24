"""
TinyBERT-style Encoder, implemented from scratch in raw PyTorch.

Based on: "TinyBERT: Distilling BERT for Natural Language Understanding"
(Jiao et al., 2019)
Paper:  https://arxiv.org/abs/1909.10351
GitHub: https://github.com/huawei-noah/Pretrained-Language-Model/tree/
        master/TinyBERT

Key architectural features from the paper (implemented here):
  - Same base structure as BERT/DistilBERT: token + position embeddings,
    stacked Transformer Encoder layers with Multi-Head Attention + FFN
  - KEY DIFFERENCE from DistilBERT: adds a linear "fitting layer" after
    each Transformer block. In the paper this projects student hidden states
    to match teacher hidden size during distillation. We include it as the
    defining structural feature of TinyBERT's design even when training
    from scratch, as it gives the model an extra learned projection at
    each layer that DistilBERT lacks.
  - Uses an embedding fitting layer too: projects word embeddings through
    a linear layer before the transformer blocks.

Original TinyBERT sizes:
  TinyBERT-4L: 4 layers, hidden=312, heads=12, intermediate=1200 (~14.5M)
  TinyBERT-6L: 6 layers, hidden=768, heads=12, intermediate=3072 (~67M)

Scaled-down configuration for our task:
  Our version: 2 layers, hidden=64, heads=4, ffn=128, fit_size=32 (~85K params)
  fit_size is the internal dimension of the fitting layers.

Task: RESTRUCTURE
  Input:  scrambled gesture tokens  e.g. "drive mother"
  Output: correctly ordered tokens  e.g. "mother drive"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyBertEmbeddings(nn.Module):
    """
    TinyBERT Embeddings: token + position, then passed through a
    linear fitting layer (the embedding fitting layer from the paper).
    This projection is unique to TinyBERT -- DistilBERT and BERT do not
    have this additional linear transformation on the embeddings.
    """
    def __init__(self, vocab_size, hidden_size, fit_size,
                 max_len, pad_id, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_id
        )
        self.position_embeddings = nn.Embedding(max_len, hidden_size)

        # TinyBERT's embedding fitting layer
        self.embedding_fit = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "position_ids",
            torch.arange(max_len).expand((1, -1))
        )

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_len]

        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)

        embeddings = word_emb + pos_emb
        # TinyBERT applies the fitting layer to the summed embeddings
        embeddings = self.embedding_fit(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """Standard Multi-Head Self-Attention (bidirectional, same as DistilBERT)."""
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.q_lin = nn.Linear(hidden_size, hidden_size)
        self.k_lin = nn.Linear(hidden_size, hidden_size)
        self.v_lin = nn.Linear(hidden_size, hidden_size)
        self.out_lin = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, S, H = x.shape
        q = self.q_lin(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, S, H)
        return self.out_lin(context)


class FFN(nn.Module):
    """Feed-Forward Network with GELU activation (same as DistilBERT)."""
    def __init__(self, hidden_size, ffn_size, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(hidden_size, ffn_size)
        self.lin2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(F.gelu(self.lin1(x))))


class TinyBertTransformerBlock(nn.Module):
    """
    One TinyBERT Transformer block.
    KEY DIFFERENCE from DistilBERT: after the FFN sublayer, applies a
    linear fitting layer to the block's output hidden states.
    In the paper this is used to project student representations to match
    the teacher's hidden size during distillation. Structurally, it gives
    each transformer block an additional learned transformation.
    """
    def __init__(self, hidden_size, n_heads, ffn_size,
                 fit_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, n_heads, dropout)
        self.ffn = FFN(hidden_size, ffn_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        # TinyBERT's hidden state fitting layer (per transformer block)
        self.hidden_fit = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, key_padding_mask=None):
        attn_out = self.attention(self.norm1(x), key_padding_mask)
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        # TinyBERT fitting layer applied after each block
        x = self.hidden_fit(x)
        return x


class TinyBertForRestructure(nn.Module):
    """Full TinyBERT-style model for the Restructure task."""
    def __init__(self, vocab_size, hidden_size=64, n_heads=4, n_layers=2,
                 ffn_size=128, fit_size=32, max_len=16,
                 pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        self.embeddings = TinyBertEmbeddings(
            vocab_size, hidden_size, fit_size, max_len, pad_id, dropout
        )
        self.layers = nn.ModuleList([
            TinyBertTransformerBlock(
                hidden_size, n_heads, ffn_size, fit_size, dropout
            )
            for _ in range(n_layers)
        ])
        self.position_head = nn.Linear(hidden_size, max_len)

    def forward(self, input_ids):
        padding_mask = (input_ids == self.pad_id)
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)
        logits = self.position_head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


TINYBERT_CONFIG = {
    "hidden_size": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_size": 128,
    "fit_size": 32,
    "max_len": 16,
    "dropout": 0.1,
}


if __name__ == "__main__":
    from tokenizer import SimpleTokenizer
    from vocab_pool import ALL_VOCAB

    tokenizer = SimpleTokenizer(ALL_VOCAB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = TinyBertForRestructure(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        **TINYBERT_CONFIG,
    ).to(device)

    print(f"Architecture: TinyBERT-style Encoder")
    print(f"Config: {TINYBERT_CONFIG}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    sample = "drive mother"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    x = torch.tensor([ids]).to(device)

    with torch.no_grad():
        out = model(x)
    print(f"\nSample input: {sample!r} -> token IDs {ids}")
    print(f"Output shape: {out.shape}  (batch=1, seq_len={len(ids)}, max_len=16)")
    print("Forward pass successful.")