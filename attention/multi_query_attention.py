"""
multi_query_attention.py

Implements multi-query self-attention (MQA).
Reference:
    - "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al.)
    - The idea: multiple heads for Q, but only 1 (or fewer) heads for K/V.
"""

import torch
import torch.nn as nn

class MultiQuerySelfAttention(nn.Module):
    def __init__(self, d_model, num_heads_q=8, num_heads_kv=1):
        super().__init__()
        self.d_model = d_model
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        # Q uses multiple heads:
        self.W_q = nn.Linear(d_model, d_model)
        # K, V use fewer heads (often 1 head)
        self.W_k = nn.Linear(d_model, d_model // self.num_heads_q * self.num_heads_kv)
        self.W_v = nn.Linear(d_model, d_model // self.num_heads_q * self.num_heads_kv)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [b, n, d_model]
        return: [b, n, d_model]
        """
        b, n, d_model = x.shape

        # Project Q with multiple heads
        q = self.W_q(x)  # [b, n, d_model]
        # Reshape for queries: [b, n, num_heads_q, d_model/num_heads_q]
        d_q_head = d_model // self.num_heads_q
        q = q.view(b, n, self.num_heads_q, d_q_head).transpose(1, 2)  # [b, num_heads_q, n, d_q_head]

        # Project K, V has one single head
        kv_dim = d_q_head * self.num_heads_kv  # total dimension for K/V
        k = self.W_k(x)  # [b, n, kv_dim]
        v = self.W_v(x)  # [b, n, kv_dim]

        # Reshape K, V to have num_heads_kv
        k = k.view(b, n, self.num_heads_kv, d_q_head).transpose(1, 2)  # [b, num_heads_kv=1, n, d_q_head]
        v = v.view(b, n, self.num_heads_kv, d_q_head).transpose(1, 2)

        # Because we have fewer heads for K/V, we might broadcast K/V across multiple Q-heads
        # In strict multi-query:
        # Q has h_q heads, K/V each have 1 head => we broadcast K/V to each Q-head
        # For demonstration, let's just do one grouping:
        if self.num_heads_kv == 1:
            # Expand k, v from shape [b, 1, n, d_q_head] to [b, num_heads_q, n, d_q_head]
            k = k.expand(b, self.num_heads_q, n, d_q_head)
            v = v.expand(b, self.num_heads_q, n, d_q_head)

        # Attention: QK^T => [b, num_heads_q, n, n]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (d_q_head ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Multiply by V => [b, num_heads_q, n, d_q_head]
        out_per_head = torch.matmul(attn_weights, v)

        # Combine heads -> [b, n, d_model]
        out = out_per_head.transpose(1, 2).contiguous().view(b, n, d_model)

        # Final linear
        out = self.W_o(out)
        return out


def multi_query_complexity(b, n, d):
    """
    Multi-query typically means we have:
      - multiple heads for Q, but 1 (or fewer) head(s) for K/V
    The big advantage is smaller overhead for K/V, especially in large contexts.

    Rough complexity:
    - Arithmetic: O(b n d^2) for Q's linear + attention, but K/V part can be smaller in practice.
    - Memory: O(b n d + b * n^2) if only 1 K,V head. Far fewer attention weights if K/V is shared.
    - The ratio similarly changes.

    We'll just return approximate symbolic forms for demonstration.
    """
    arithmetic_ops = f"O(b n d^2), but reduced K/V overhead"
    memory_usage = f"O(b n d + b n^2) since only 1 head for K/V in the extreme"
    ratio = f"O((b n d + b n^2) / (b n d^2))"
    return arithmetic_ops, memory_usage, ratio
