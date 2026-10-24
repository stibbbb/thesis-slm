import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistilBertEmbeddings(nn.Module):
    """
    Token + Position embeddings.
    DistilBERT uses LEARNED positional embeddings (not sinusoidal like
    the original Transformer paper). This is one of the differences from
    BERT, which also adds token_type (segment) embeddings -- DistilBERT
    removes those entirely since it drops the NSP task.
    """
    def __init__(self, vocab_size, hidden_size, max_len, pad_id, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_id
        )
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        # Register position_ids as a buffer (not a trainable parameter)
        # so it moves to the correct device automatically
        self.register_buffer(
            "position_ids",
            torch.arange(max_len).expand((1, -1))
        )

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_len]

        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)

        # Sum token + position embeddings, then normalize and dropout
        embeddings = word_emb + pos_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention as in DistilBERT.
    Bidirectional -- every token attends to every other token
    (unlike causal/decoder attention which can only look left).
    This is the core reason Encoder architectures suit the restructure
    task: the model sees the whole input at once and can determine
    the correct ordering by comparing all tokens simultaneously.
    """
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

        # Project to queries, keys, values and split into heads
        q = self.q_lin(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Mask padding positions so they don't influence attention
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values, then concat heads
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, S, H)
        return self.out_lin(context)


class FFN(nn.Module):
    """
    Feed-Forward Network sub-layer.
    DistilBERT uses GELU activation (same as BERT), unlike the original
    Transformer paper which used ReLU. The intermediate size is typically
    4x the hidden size in the original -- we maintain this ratio.
    """
    def __init__(self, hidden_size, ffn_size, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(hidden_size, ffn_size)
        self.lin2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(F.gelu(self.lin1(x))))


class TransformerBlock(nn.Module):
    """
    One DistilBERT Transformer block:
      Self-Attention -> Add & Norm -> FFN -> Add & Norm
    Uses Pre-LN variant (LayerNorm before sublayer) for training stability,
    consistent with the HuggingFace DistilBERT implementation.
    """
    def __init__(self, hidden_size, n_heads, ffn_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, n_heads, dropout)
        self.ffn = FFN(hidden_size, ffn_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), key_padding_mask)
        x = x + self.dropout(attn_out)

        # FFN with residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x


class DistilBertForRestructure(nn.Module):
    """
    Full DistilBERT-style model for the Restructure task.
    Stacks N TransformerBlocks then applies a position-prediction head.
    """
    def __init__(self, vocab_size, hidden_size=64, n_heads=4, n_layers=2,
                 ffn_size=128, max_len=16, pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        self.embeddings = DistilBertEmbeddings(
            vocab_size, hidden_size, max_len, pad_id, dropout
        )
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, ffn_size, dropout)
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


# Scaled-down config that mirrors DistilBERT's structural ratios
DISTILBERT_CONFIG = {
    "hidden_size": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_size": 128,   # 2x hidden_size (original uses 4x; we use 2x to stay small)
    "max_len": 16,
    "dropout": 0.1,
}


if __name__ == "__main__":
    from tokenizer import SimpleTokenizer
    from vocab_pool import ALL_VOCAB

    tokenizer = SimpleTokenizer(ALL_VOCAB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = DistilBertForRestructure(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        **DISTILBERT_CONFIG,
    ).to(device)

    print(f"Architecture: DistilBERT-style Encoder")
    print(f"Config: {DISTILBERT_CONFIG}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    # Forward pass sanity check
    sample = "study mother"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    x = torch.tensor([ids]).to(device)

    with torch.no_grad():
        out = model(x)
    print(f"\nSample input: {sample!r} -> token IDs {ids}")
    print(f"Output shape: {out.shape}  (batch=1, seq_len={len(ids)}, max_len=16)")
    print("Forward pass successful.")