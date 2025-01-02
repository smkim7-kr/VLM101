import torch
import torch.nn as nn
import math

class SimpleSelfAttention(nn.Module):
    """
    A toy self-attention module that supports storing and using a KV-cache.

    In a real Transformer, each layer has multi-head attention and other complexities.
    Here, we show just one 'head' for demonstration.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Simple linear maps to produce Q, K, V from input
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Final linear after attention
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
               representing new tokens' embeddings.
            past_kv: Tuple (past_k, past_v) each of shape 
                     [batch_size, past_seq_len, d_model]
                     or None if there's no cache yet.

        Returns:
            out: Tensor of shape [batch_size, seq_len, d_model]
            new_kv: Tuple (k_cache, v_cache) updated to include new tokens.
        """
        batch_size, seq_len, _ = x.shape

        # Compute new K, V, Q for the *current* chunk of x
        q = self.W_q(x)  # [batch_size, seq_len, d_model]
        k = self.W_k(x)  # [batch_size, seq_len, d_model]
        v = self.W_v(x)  # [batch_size, seq_len, d_model]

        # If we have past K, V, append along the time dimension
        # so K, V become [batch_size, past_seq_len + seq_len, d_model]
        if past_kv is not None:
            past_k, past_v = past_kv
            k_cache = torch.cat([past_k, k], dim=1)
            v_cache = torch.cat([past_v, v], dim=1)
        else:
            k_cache, v_cache = k, v

        # QK^T => attention scores
        attn_scores = torch.matmul(q, k_cache.transpose(-1, -2)) 
        attn_scores = attn_scores / math.sqrt(self.d_model)  # scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Multiply by V => [batch_size, seq_len, d_model]
        out = torch.matmul(attn_weights, v_cache)
        out = self.W_o(out)

        return out, (k_cache, v_cache)

class SimpleDecoder(nn.Module):
    """
    A mini 'decoder' with n_layers of SimpleSelfAttention, each of which
    can use a KV-cache to perform incremental (autoregressive) attention.
    """
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleSelfAttention(d_model) for _ in range(n_layers)
        ])

    def forward(self, x, past_kv_list=None):
        """
        x: [batch, seq_len, d_model]
        past_kv_list: list of (past_k, past_v) for each layer, or None
        """
        if past_kv_list is None:
            past_kv_list = [None] * len(self.layers)

        new_past_kv_list = []
        out = x
        for layer, past_kv in zip(self.layers, past_kv_list):
            out, kv = layer(out, past_kv)
            new_past_kv_list.append(kv)
        return out, new_past_kv_list

def kv_cache_memory_usage(
    n_layers,
    d_model,
    seq_len,
    batch_size=1,
    precision_bytes=4
):
    """
    2 * precision_bytes * n_layers * d_model * seq_len * batch_size
    """
    return 2 * precision_bytes * n_layers * d_model * seq_len * batch_size

# -------------------------------
# Demonstration of using the KV-cache
if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size = 1
    d_model = 8
    n_layers = 2

    # Create a tiny decoder with 2 layers
    decoder = SimpleDecoder(d_model, n_layers)

    # ---------------------------
    # 1) "First pass": process an initial prompt of length = 4
    seq_len_prompt = 4
    prompt_embeddings = torch.randn(batch_size, seq_len_prompt, d_model)
    past_kv = None  # no cache yet

    # Forward pass (like finishing the "prompt")
    output, past_kv = decoder(prompt_embeddings, past_kv)

    print("After first pass (prompt of length 4):")
    for i, (k_cache, v_cache) in enumerate(past_kv):
        print(f" Layer {i} cache shapes: K={k_cache.shape}, V={v_cache.shape}")
    # K and V each are [batch_size, 4, d_model]

    # Memory usage so far for the KV cache
    memory_mb = kv_cache_memory_usage(
        n_layers=n_layers,
        d_model=d_model,
        seq_len=seq_len_prompt,  # 4 tokens in cache
        batch_size=batch_size,
        precision_bytes=4  # float32
    ) / 1e6
    print(f"Approx. KV-cache memory usage: {memory_mb:.3f} MB\n")

    # ---------------------------
    # 2) "Incremental decoding": pretend we generate 3 more tokens, one by one
    for step in range(1, 4):
        # Suppose we get a new token embedding at each step
        new_token_embed = torch.randn(batch_size, 1, d_model)
        
        # We pass it through the decoder with the existing cache
        output, past_kv = decoder(new_token_embed, past_kv)

        # Now the cache is extended by 1 token in each layer
        print(f"After generating token #{step}:")
        for i, (k_cache, v_cache) in enumerate(past_kv):
            print(f" Layer {i} cache shapes: K={k_cache.shape}, V={v_cache.shape}")
        # K and V each are [batch_size, 4 + step, d_model]

        # Memory usage after each new token is appended
        memory_mb = kv_cache_memory_usage(
            n_layers=n_layers,
            d_model=d_model,
            seq_len=seq_len_prompt + step,  
            batch_size=batch_size,
            precision_bytes=4  # float32
        ) / 1e6
        print(f"Approx. KV-cache memory usage so far: {memory_mb:.3f} MB\n")
