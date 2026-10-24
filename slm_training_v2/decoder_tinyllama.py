import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Simpler than LayerNorm: no mean subtraction, just scale by 1/RMS.
    Used in all Llama-family models instead of standard LayerNorm.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def precompute_rope_freqs(head_dim, max_len, base=10000):
    """
    Precompute the rotation frequencies for RoPE.
    For each head dimension pair, compute a frequency that determines
    how fast that dimension's angle rotates with position.
    """
    theta = 1.0 / (base ** (
        torch.arange(0, head_dim, 2).float() / head_dim
    ))
    positions = torch.arange(max_len).float()
    freqs = torch.outer(positions, theta)
    # Stack cos and sin for the rotation
    freqs_cis = torch.stack([freqs.cos(), freqs.sin()], dim=-1)
    return freqs_cis


def apply_rope(x, freqs_cis):
    """
    Apply Rotary Position Embeddings to query or key tensor.
    x shape: (batch, heads, seq_len, head_dim)
    freqs_cis shape: (seq_len, head_dim//2, 2)
    """
    head_dim = x.shape[-1]
    x_r = x.float().reshape(*x.shape[:-1], head_dim // 2, 2)
    cos = freqs_cis[..., 0].unsqueeze(0).unsqueeze(0)
    sin = freqs_cis[..., 1].unsqueeze(0).unsqueeze(0)
    # Apply 2D rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    x_out = torch.stack([
        x_r[..., 0] * cos - x_r[..., 1] * sin,
        x_r[..., 0] * sin + x_r[..., 1] * cos,
    ], dim=-1).reshape(*x.shape)
    return x_out.type_as(x)


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    Instead of: Linear -> GELU -> Linear
    SwiGLU does: split(Linear(x)) -> SiLU(gate) * value -> Linear

    This gating mechanism is empirically better for language models.
    Used in all Llama-family models, contrasting with BERT's plain GELU FFN.
    """
    def __init__(self, hidden_size, ffn_size, dropout=0.1):
        super().__init__()
        # Gate and value projections (split input into two halves)
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SiLU(gate) * value, then project back down
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.down_proj(self.dropout(gate * value))


class TinyLlamaCausalAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention with RoPE.
    Causal means each position can only attend to itself and earlier
    positions -- this enables left-to-right autoregressive generation.
    """
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

        freqs = precompute_rope_freqs(self.head_dim, max_len)
        self.register_buffer("freqs_cis", freqs)

    def forward(self, x, memory=None, causal_mask=None,
                tgt_padding_mask=None, src_padding_mask=None):
        B, S, H = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, self.freqs_cis[:S])
        k = apply_rope(k, self.freqs_cis[:S])

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if causal_mask is not None:
            scores = scores.masked_fill(
                causal_mask[:S, :S].unsqueeze(0).unsqueeze(0), float("-inf")
            )
        if tgt_padding_mask is not None:
            scores = scores.masked_fill(
                tgt_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, H)
        return self.o_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention: target attends to source (encoder output / memory)."""
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % n_heads == 0
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

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if src_padding_mask is not None:
            scores = scores.masked_fill(
                src_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.o_proj(out)


class TinyLlamaBlock(nn.Module):
    """
    One TinyLlama decoder block:
      RMSNorm -> Causal Self-Attention -> residual
      RMSNorm -> Cross-Attention -> residual
      RMSNorm -> SwiGLU FFN -> residual

    Note: Llama 2 is a decoder-only model with no cross-attention.
    We add cross-attention here to handle our encoder-decoder task
    (the source gesture tokens act as the encoder memory).
    """
    def __init__(self, hidden_size, n_heads, ffn_size, max_len, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.norm3 = RMSNorm(hidden_size)

        self.self_attn = TinyLlamaCausalAttention(
            hidden_size, n_heads, max_len, dropout
        )
        self.cross_attn = CrossAttention(hidden_size, n_heads, dropout)
        self.ffn = SwiGLU(hidden_size, ffn_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask=None,
                tgt_padding_mask=None, src_padding_mask=None):
        x = x + self.dropout(self.self_attn(
            self.norm1(x), causal_mask=causal_mask,
            tgt_padding_mask=tgt_padding_mask
        ))
        x = x + self.dropout(self.cross_attn(
            self.norm2(x), memory, src_padding_mask=src_padding_mask
        ))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class TinyLlamaForGeneration(nn.Module):
    """Full TinyLlama-style decoder for the Complete task (Sentence Completion)."""
    def __init__(self, vocab_size, hidden_size=64, n_heads=4, n_layers=2,
                 ffn_size=128, max_len=32, pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.hidden_size = hidden_size

        self.token_embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_id
        )
        self.src_embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_id
        )

        self.layers = nn.ModuleList([
            TinyLlamaBlock(hidden_size, n_heads, ffn_size, max_len, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size, bias=False)

        causal = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal)

    # embed_source() is the pipeline's Embedding box for the Deaf ->
    # hearing path, and generate_from_memory() is Sentence Completion
    # (slm_pipeline.py). Splitting them lets Active Context add a
    # remembered person's vector in between. forward() and generate()
    # still run the same computation as before.

    def embed_source(self, input_ids):
        """Embedding: one 64-number vector per sign (no position added)."""
        return self.src_embedding(input_ids)

    def decode(self, memory, target_ids, src_padding_mask=None):
        """Score the next word at every position of target_ids, reading
        the embedded signs (`memory`) through cross-attention."""
        tgt_padding = target_ids == self.pad_id
        x = self.token_embedding(target_ids)

        for layer in self.layers:
            x = layer(
                x, memory,
                causal_mask=self.causal_mask,
                tgt_padding_mask=tgt_padding,
                src_padding_mask=src_padding_mask,
            )
        x = self.norm(x)
        return self.output_head(x)

    def forward(self, input_ids, target_ids):
        return self.decode(self.embed_source(input_ids), target_ids,
                           src_padding_mask=input_ids == self.pad_id)

    def generate(self, input_ids, tokenizer, max_new_tokens=16):
        self.eval()
        with torch.no_grad():
            memory = self.embed_source(input_ids)
        return self.generate_from_memory(
            memory, tokenizer, max_new_tokens,
            src_padding_mask=input_ids == self.pad_id)

    def generate_from_memory(self, memory, tokenizer, max_new_tokens=16,
                             src_padding_mask=None):
        """Write the sentence one word at a time from already-embedded signs."""
        self.eval()
        device = memory.device
        generated = torch.tensor([[tokenizer.bos_id]], device=device)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.decode(memory, generated, src_padding_mask)
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


TINYLLAMA_CONFIG = {
    "hidden_size": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_size": 128,
    "max_len": 32,
    "dropout": 0.1,
}


if __name__ == "__main__":
    from tokenizer import SimpleTokenizer
    from vocab_pool import ALL_VOCAB

    tokenizer = SimpleTokenizer(ALL_VOCAB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = TinyLlamaForGeneration(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        **TINYLLAMA_CONFIG,
    ).to(device)

    print(f"Architecture: TinyLlama-style Decoder")
    print(f"Config: {TINYLLAMA_CONFIG}")
    print(f"Trainable parameters: {model.count_parameters():,}")

    src_ids = tokenizer.encode("mother eat", add_special_tokens=False)
    tgt_ids = tokenizer.encode("mother is", add_special_tokens=True)
    src = torch.tensor([src_ids]).to(device)
    tgt = torch.tensor([tgt_ids]).to(device)

    with torch.no_grad():
        out = model(src, tgt)
    print(f"\nForward pass output shape: {out.shape}")
    print(f"Untrained generation (will be random):")
    result = model.generate(src, tokenizer)
    print(f"  Output: {result!r}")
    print("Forward pass successful.")