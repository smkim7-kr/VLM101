import torch
import math

def rotate_half(x):
    """
    Rotates half the hidden dimensions of the input tensor.
    Args:
        x (torch.Tensor): Input tensor of shape (..., dim).
    Returns:
        torch.Tensor: Tensor with rotated dimensions.
    """
    # Split the input tensor into two halves
    x1 = x[..., : x.shape[-1] // 2]  # First half
    x2 = x[..., x.shape[-1] // 2 :]  # Second half

    # Rotate the halves
    return torch.cat((-x2, x1), dim=-1)

def get_rotary_embeddings(seq_len, embed_dim, base=10000):
    """
    Generates sinusoidal rotary embeddings.
    Args:
        seq_len (int): Length of the sequence.
        dim (int): Dimension of embeddings.
        base (int): Base for the sinusoidal frequencies.
    Returns:
        cos (torch.Tensor): Cosine embeddings of shape [seq_len, dim].
        sin (torch.Tensor): Sine embeddings of shape [seq_len, dim].
    """
    # Generate position indices (shape = (seq_len, 1))
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)

    # Compute scaling factors (shape = (embed_dim // 2))
    # theta_i = 10000^(-i / (embed_dim //2))
    theta = torch.arange(0, embed_dim // 2, dtype=torch.float) / (embed_dim // 2)
    theta = base ** (-theta)  # Frequency scaling

    # Compute cos and sin embeddings (shape = (seq_len, embed_dim//2))
    angles = position * theta.unsqueeze(0)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Concatenate for full dimension (shape = (seq_len, embed_dim))
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    # shape = (1, 1, seq_len, embed_dim) 
    # where 0:embed_dim//2 and embed_dim//2:-1 are the same
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Applies rotary positional embedding to query and key tensors.
    Args:
        q (torch.Tensor): Query tensor of shape [batch_size, seq_len, dim].
        k (torch.Tensor): Key tensor of shape [batch_size, seq_len, dim].
        cos (torch.Tensor): Cosine embeddings [1, 1, seq_len, dim].
        sin (torch.Tensor): Sine embeddings [1, 1, seq_len, dim].
        position_ids (torch.Tensor): Position indices [batch_size, seq_len].
    Returns:
        torch.Tensor: Query and key tensors with rotary embeddings applied.
    """
    # Squeeze unused dimensions in cos and sin
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]

    # Index embeddings by position IDs
    cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]

    # Apply rotary positional embeddings
    # q = [q1, q2, q3, ..., q_{seq_len}]
    # ratate_helf(q) = [q
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

# Example Usage
seq_len = 10  # Sequence length
dim = 64  # Embedding dimension
batch_size = 2  # Batch size

# Generate rotary embeddings
cos, sin = get_rotary_embeddings(seq_len, dim)

# Create dummy query and key tensors
q = torch.randn(batch_size, seq_len, dim)
k = torch.randn(batch_size, seq_len, dim)

# Create dummy position IDs
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]

# Apply rotary positional embeddings
q_rotary, k_rotary = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

print("Original Q Shape:", q.shape)
print("Rotary-embedded Q Shape:", q_rotary.shape)
