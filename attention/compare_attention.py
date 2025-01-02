"""
compare_attention.py

Compares:
  - Multi-Head Self-Attention
  - Multi-Query Self-Attention
  - Grouped Multi-Query Self-Attention

We import from the four separate files and run them on a toy input.
Then we print out approximate big-O complexities.
"""

import torch

from multi_head_attention import (
    MultiHeadSelfAttention,
    multi_head_complexity
)
from multi_query_attention import (
    MultiQuerySelfAttention,
    multi_query_complexity
)
from grouped_multi_query_attention import (
    GroupedMultiQuerySelfAttention,
    grouped_multi_query_complexity
)

if __name__ == "__main__":
    # Hyperparameters for demonstration
    b = 2     # batch size
    n = 16    # sequence length
    d = 32    # model dimension
    h = 8     # number of heads
    x = torch.randn(b, n, d)

    print("===== 1) Multi-Head Self-Attention =====")
    mha = MultiHeadSelfAttention(d_model=d, num_heads=h)
    out_mha = mha(x)
    ops_mha, mem_mha, ratio_mha = multi_head_complexity(b, n, d, h)
    print("Output shape:", out_mha.shape)
    print("Arithmetic Ops:", ops_mha)
    print("Memory usage:", mem_mha)
    print("Memory/Ops ratio:", ratio_mha)
    print()

    print("===== 2) Multi-Query Self-Attention =====")
    mqa = MultiQuerySelfAttention(d_model=d, num_heads_q=h, num_heads_kv=1)
    out_mqa = mqa(x)
    ops_mqa, mem_mqa, ratio_mqa = multi_query_complexity(b, n, d)
    print("Output shape:", out_mqa.shape)
    print("Arithmetic Ops:", ops_mqa)
    print("Memory usage:", mem_mqa)
    print("Memory/Ops ratio:", ratio_mqa)
    print()

    print("===== 3) Grouped Multi-Query Self-Attention =====")
    grouped_mqa = GroupedMultiQuerySelfAttention(d_model=d, num_query_heads=h, num_kv_groups=2)
    out_grouped = grouped_mqa(x)
    ops_grp, mem_grp, ratio_grp = grouped_multi_query_complexity(b, n, d, h, g=2)
    print("Output shape:", out_grouped.shape)
    print("Arithmetic Ops:", ops_grp)
    print("Memory usage:", mem_grp)
    print("Memory/Ops ratio:", ratio_grp)
    print()
