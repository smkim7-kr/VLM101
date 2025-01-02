"""
grouped_multi_query_attention.py

Implements "grouped multi-query" self-attention.
Idea:
  - We group the Q heads into subsets that share the same K,V heads.
  - It's in-between standard multi-head and single-head K,V.
"""

import torch
import torch.nn as nn

class GroupedMultiQuerySelfAttention(nn.Module):
    def __init__(self, d_model, num_query_heads=8, num_kv_groups=2):
        """
        Suppose we have total of 8 Q-heads, grouped into 2 groups
        => effectively 2 sets of K,V parameters, each shared by 4 Q-heads.
        """
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_groups = num_kv_groups
        self.W_q = nn.Linear(d_model, d_model)
        # K, V are sized for the kv_groups
        self.W_k = nn.Linear(d_model, d_model // self.num_query_heads * self.num_kv_groups)
        self.W_v = nn.Linear(d_model, d_model // self.num_query_heads * self.num_kv_groups)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [b, n, d_model]
        return: [b, n, d_model]
        """
        b, n, d_model = x.shape

        # Project Q
        q = self.W_q(x)  # [b, n, d_model]
        d_q_head = d_model // self.num_query_heads
        q = q.view(b, n, self.num_query_heads, d_q_head).transpose(1, 2)  # [b, num_query_heads, n, d_q_head]

        # Project K, V for 'num_kv_groups'
        # each group dimension = (d_q_head * group_size)
        k = self.W_k(x)  # [b, n, d_model // num_query_heads * num_kv_groups]
        v = self.W_v(x)

        # Reshape K, V to [b, num_kv_groups, n, d_q_head * (some group size)]
        # For simplicity, let's define each group to produce d_q_head dimension:
        # Then each group is repeated for the Q heads that belong to that group.
        k = k.view(b, n, self.num_kv_groups, d_q_head).transpose(1, 2)  # [b, num_kv_groups, n, d_q_head]
        v = v.view(b, n, self.num_kv_groups, d_q_head).transpose(1, 2)

        # We'll "assign" each Q head to one of the groups. Example: if num_query_heads=8, num_kv_groups=2,
        # heads 0..3 -> group 0, heads 4..7 -> group 1
        # We'll create a block of code that does grouping in practice:
        out_heads = []
        heads_per_group = self.num_query_heads // self.num_kv_groups
        for group_idx in range(self.num_kv_groups):
            # Q subset for that group
            start_h = group_idx * heads_per_group
            end_h = start_h + heads_per_group
            q_sub = q[:, start_h:end_h, :, :]  # [b, heads_per_group, n, d_q_head]
            
            # K,V for that group => k[:, group_idx, ...] => [b, n, d_q_head]
            k_group = k[:, group_idx, :, :].unsqueeze(1)  # [b, 1, n, d_q_head]
            v_group = v[:, group_idx, :, :].unsqueeze(1)

            # Expand along head dimension (heads_per_group)
            k_group = k_group.expand(b, heads_per_group, n, d_q_head)
            v_group = v_group.expand(b, heads_per_group, n, d_q_head)

            # Attention
            attn_scores = torch.matmul(q_sub, k_group.transpose(-1, -2)) / (d_q_head ** 0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            out_sub = torch.matmul(attn_weights, v_group)
            out_heads.append(out_sub)  # [b, heads_per_group, n, d_q_head]

        # Concat all heads
        out_all = torch.cat(out_heads, dim=1)  # [b, num_query_heads, n, d_q_head]
        out = out_all.transpose(1, 2).contiguous().view(b, n, d_model)
        out = self.W_o(out)
        return out

def grouped_multi_query_complexity(b, n, d, h, g):
    """
    For grouped multi-query:
      - h = total Q heads
      - g = number of K/V groups (g < h)
    Complexity wise, it's in-between standard MHA and single-KV MQA.

    - Arithmetic ops roughly O(b n d^2) but the K,V overhead is scaled by g rather than h.
    - Memory also in-between: O(b n d + b g n^2)

    We'll return symbolic forms for demonstration.
    """
    arithmetic_ops = f"O(b n d^2), partial reduction from h to g for K/V"
    memory_usage = f"O(b n d + b g n^2)"
    ratio = f"O((b n d + b g n^2) / (b n d^2))"
    return arithmetic_ops, memory_usage, ratio
