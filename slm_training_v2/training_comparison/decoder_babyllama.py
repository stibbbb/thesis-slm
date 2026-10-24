"""
BabyLlama-style Decoder, implemented from scratch in raw PyTorch.

Based on: "Baby Llama: Knowledge Distillation from an Ensemble of Teachers
Trained on a Small Dataset with No Performance Penalty"
(Timiryasov & Tastet, 2023)
Paper:  https://arxiv.org/abs/2308.02019
GitHub: https://github.com/timinar/BabyLlama

BabyLlama shares Llama's core architecture (RoPE, SwiGLU, RMSNorm, causal
attention) but has two structural differences from TinyLlama:

  1. DEEPER BUT NARROWER configuration: BabyLlama uses more layers but
     smaller hidden dimensions relative to parameter budget. The 58M
     paper model uses more layers than a same-size TinyLlama would.
     This is a deliberate design choice for low-data settings -- more
     layers means more compositional depth, which helps with limited
     training data because the model can learn hierarchical patterns
     even from few examples.

  2. PRE-NORM with additional output norm per block: BabyLlama applies
     an extra RMSNorm to the output of each block's sublayers before
     the residual addition (double normalization), making training more
     stable on small datasets where gradients can be erratic.

Original BabyLlama: ~58M parameters
Our scaled version: 3 layers, hidden=48, heads=4, ffn=96 (~75K params)
(Deeper/narrower ratio than TinyLlama which uses 2 layers, hidden=64)

Tasks: COMPLETE and REDUCE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def precompute_rope_freqs(head_dim, max_len, base=10000):
    theta = 1.0 / (base ** (
        torch.arange(0, head_dim, 2).float() / head_dim
    ))
    positions = torch.arange(max_len).float()
    freqs = torch.outer(positions, theta)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)


def apply_rope(x, freqs_cis):
    head_dim = x.shape[-1]
    x_r = x.float().reshape(*x.shape[:-1], head_dim // 2, 2)
    cos = freqs_cis[..., 0].unsqueeze(0).unsqueeze(0)
    sin = freqs_cis[..., 1].unsqueeze(0).unsqueeze(0)
    x_out = torch.stack([
        x_r[..., 0] * cos - x_r[..., 1] * sin,
        x_r[..., 0] * sin + x_r[..., 1] * cos,
    ], dim=-1).reshape(*x.shape)
    return x_out.type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        ))


class BabyLlamaCausalAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, max_len, dropout=0.1):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("freqs_cis",
            precompute_rope_freqs(self.head_dim, max_len))

    def forward(self, x, causal_mask=None, tgt_padding_mask=None):
        B, S, H = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.freqs_cis[:S])
        k = apply_rope(k, self.freqs_cis[:S])

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            scores = scores.masked_fill(
                causal_mask[:S, :S].unsqueeze(0).unsqueeze(0), float("-inf")
            )
        if tgt_padding_mask is not None:
            scores = scores.masked_fill(
                tgt_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, H)
        return self.o_proj(out)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_padding_mask=None):
        B, T, H = x.shape
        S = memory.size(1)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if src_padding_mask is not None:
            scores = scores.masked_fill(
                src_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, H)
        return self.o_proj(out)


class BabyLlamaBlock(nn.Module):
    """
    BabyLlama decoder block.

    KEY DIFFERENCE from TinyLlama: double normalization per sublayer.
    Each sublayer applies RMSNorm BOTH before AND after the sublayer
    computation (before the residual add). This extra normalization
    stabilizes training on small datasets where gradient updates can
    be noisy and erratic -- exactly our situation with 139 examples.
    This is the structural feature that distinguishes BabyLlama blocks
    from standard Llama/TinyLlama blocks.
    """
    def __init__(self, hidden_size, n_heads, ffn_size, max_len, dropout=0.1):
        super().__init__()
        # Pre-norm (applied before sublayer)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.norm3 = RMSNorm(hidden_size)
        # Post-sublayer norm (applied after sublayer, before residual add)
        # This double-norm is BabyLlama's stability contribution
        self.post_norm1 = RMSNorm(hidden_size)
        self.post_norm2 = RMSNorm(hidden_size)
        self.post_norm3 = RMSNorm(hidden_size)

        self.self_attn = BabyLlamaCausalAttention(
            hidden_size, n_heads, max_len, dropout
        )
        self.cross_attn = CrossAttention(hidden_size, n_heads, dropout)
        self.ffn = SwiGLU(hidden_size, ffn_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask=None,
                tgt_padding_mask=None, src_padding_mask=None):
        # Double-norm self-attention
        attn_out = self.self_attn(
            self.norm1(x), causal_mask=causal_mask,
            tgt_padding_mask=tgt_padding_mask
        )
        x = x + self.dropout(self.post_norm1(attn_out))

        # Double-norm cross-attention
        cross_out = self.cross_attn(
            self.norm2(x), memory, src_padding_mask=src_padding_mask
        )
        x = x + self.dropout(self.post_norm2(cross_out))

        # Double-norm FFN
        ffn_out = self.ffn(self.norm3(x))
        x = x + self.dropout(self.post_norm3(ffn_out))
        return x


class BabyLlamaForGeneration(nn.Module):
    """Full BabyLlama-style decoder for Complete and Reduce tasks."""
    def __init__(self, vocab_size, hidden_size=48, n_heads=4, n_layers=3,
                 ffn_size=96, max_len=32, pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        self.token_embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_id
        )
        self.src_embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_id
        )
        self.layers = nn.ModuleList([
            BabyLlamaBlock(hidden_size, n_heads, ffn_size, max_len, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size, bias=False)

        causal = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal)

    def forward(self, input_ids, target_ids):
        src_padding = input_ids == self.pad_id
        tgt_padding = target_ids == self.pad_id
        memory = self.src_embedding(input_ids)
        x = self.token_embedding(target_ids)
        for layer in self.layers:
            x = layer(x, memory,
                      causal_mask=self.causal_mask,
                      tgt_padding_mask=tgt_padding,
                      src_padding_mask=src_padding)
        return self.output_head(self.norm(x))

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
        return tokenizer.decode(generated[0, 1:].tolist(), skip_special_tokens=True)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


BABYLLAMA_CONFIG = {
    "hidden_size": 48,
    "n_heads": 4,
    "n_layers": 3,
    "ffn_size": 96,
    "max_len": 32,
    "dropout": 0.1,
}


if __name__ == "__main__":
    from tokenizer import SimpleTokenizer
    from vocab_pool import ALL_VOCAB

    tokenizer = SimpleTokenizer(ALL_VOCAB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = BabyLlamaForGeneration(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        **BABYLLAMA_CONFIG,
    ).to(device)

    print(f"Architecture: BabyLlama-style Decoder")
    print(f"Config: {BABYLLAMA_CONFIG}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    src = torch.tensor([tokenizer.encode(
        "mother laundry", add_special_tokens=False
    )]).to(device)
    tgt = torch.tensor([tokenizer.encode(
        "mother is", add_special_tokens=True
    )]).to(device)

    with torch.no_grad():
        out = model(src, tgt)
    print(f"Forward pass output shape: {out.shape}")
    result = model.generate(src, tokenizer)
    print(f"Untrained generation: {result!r}")
    print("Forward pass successful.")