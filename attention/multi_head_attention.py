"""
multi_head_attention.py

Implements standard multi-head self-attention (MHA).
References:
    - "Attention is all you need" (Vaswani et al.)
    - Typical PyTorch multi-head attention

Notation:
    b = batch size
    n = sequence length
    d = model dimension
    h = number of heads (e.g., 8)
"""

import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [b, n, d_model]
        Returns: [b, n, d_model]
        """
        # Step 1: project to Q,K,V
        q = self.W_q(x)  # [b, n, d_model]
        k = self.W_k(x)  # [b, n, d_model]
        v = self.W_v(x)  # [b, n, d_model]

        # Step 2: reshape Q,K,V into multiple heads
        b, n, d_model = q.shape
        d_head = d_model // self.num_heads
        # Reshape to [b, n, h, d_head], then transpose for [b, h, n, d_head]
        q = q.view(b, n, self.num_heads, d_head).transpose(1, 2)
        k = k.view(b, n, self.num_heads, d_head).transpose(1, 2)
        v = v.view(b, n, self.num_heads, d_head).transpose(1, 2)

        # Step 3: compute attention
        # attn_scores: QK^T / sqrt(d_head) => shape = [b, h, n, n]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (d_head ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # out_per_head: [b, h, n, d_head]
        out_per_head = torch.matmul(attn_weights, v)

        # Step 4: combine heads
        # [b, h, n, d_head] -> [b, n, h, d_head] -> [b, n, d_model]
        out = out_per_head.transpose(1, 2).contiguous().view(b, n, d_model)

        # Step 5: final linear
        out = self.W_o(out)  # [b, n, d_model]
        return out


def multi_head_complexity(b, n, d, h):
    """
    Returns approximate big-O complexities for multi-head attention:

    - Arithmetic Ops: O(b * n * d^2) or more precisely O(b * n * n * d_head),
      but commonly approximated as O(b n d^2) if h is considered constant.
    - Memory: O(b * n * d), plus attention weights O(b*h*n*n).
      Typically stated as O(b n d + b h n^2).
    - Ratio: memory / ops, in big-O terms.

    For large n >> d, the n^2 factor in attention can dominate.
    For large d >> n, the d^2 factor in linear layers can dominate.

    We'll just return symbolic strings for demonstration.
    """
    # A rough measure:
    arithmetic_ops = f"O(b n d^2) with h heads"
    memory_usage = f"O(b n d + b h n^2)"
    ratio = f"O((b n d + b h n^2) / (b n d^2))"
    return arithmetic_ops, memory_usage, ratio
