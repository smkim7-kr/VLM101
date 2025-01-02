# Issue: RPE is slow, cannot implement KV-Cache
import numpy as np
import matplotlib.pyplot as plt

def get_relative_positional_encoding(seq_len, d_model):
    """
    Generate relative positional encoding for a sequence.
    
    Args:
        seq_len (int): The length of the sequence.
        d_model (int): The dimension of the embedding.

    Returns:
        np.ndarray: Relative positional encoding of shape (seq_len, seq_len, d_model).
    """
    # Create a matrix of relative positions
    # [[0, 1, 2, ...], [-1, 0, 1, ...], [-2, -1, 0, 1, ...], ...]
    relative_positions = np.arange(seq_len)[np.newaxis, :] - np.arange(seq_len)[:, np.newaxis]
    
    # Initialize positional encoding matrix
    # lookup table for all possible relative positions in the sequence (-seq_len+1 ... 0 ... seq_len-1)
    positional_encoding = np.zeros((2 * seq_len - 1, d_model))

    # Use sine and cosine for relative positional encoding
    # Same as APE, instead use relative position 
    position = np.arange(-seq_len + 1, seq_len)[:, np.newaxis]
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)
    positional_encoding[:, 0::2] = np.sin(position / div_term)  # Sine for even indices
    positional_encoding[:, 1::2] = np.cos(position / div_term)  # Cosine for odd indices
    
    # Map relative positions to the positional encoding matrix
    relative_positional_encoding = positional_encoding[relative_positions + seq_len - 1]

    return relative_positional_encoding

# Example usage
seq_len = 10  # Length of the sequence
d_model = 16  # Embedding dimension

relative_positional_encoding = get_relative_positional_encoding(seq_len, d_model)

# Visualizing relative positional encoding for the first dimension
plt.figure(figsize=(10, 6))
plt.imshow(relative_positional_encoding[:, :, 0], cmap="viridis", aspect="auto")
plt.colorbar()
plt.xlabel("Token Position")
plt.ylabel("Relative Position")
plt.title("Relative Positional Encoding (First Dimension)")
plt.show()
