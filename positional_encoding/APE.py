# Issue: position embedding between 1 and 2 are not closer than 1 and 100
import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(seq_len, d_model):
    """
    Generate the positional encoding for a sequence.
    
    Args:
        seq_len (int): The length of the sequence.
        d_model (int): The dimension of the embedding.

    Returns:
        np.ndarray: Positional encoding of shape (seq_len, d_model).
    """
    # Initialize a matrix for positional encodings
    positional_encoding = np.zeros((seq_len, d_model))

    # Create position and dimension arrays
    position = np.arange(0, seq_len)[:, np.newaxis]
    # Computational trick using log propetry: a^b = e^(b * log(a))
    # Here a = 10000 and b = 2i/d_model = np.arange(0, d_model, 2) / d_model
    # - in front of np.log(10000) since this term is denominator
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    # Apply sine to even indices (along d_model dimension)
    positional_encoding[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices (along d_model dimension)
    positional_encoding[:, 1::2] = np.cos(position * div_term)

    return positional_encoding

# Example usage
seq_len = 50  # Length of the sequence
d_model = 512  # Embedding dimension

# Dim not changed! (seq_len, d_model) -> (seq_len, d_model) 
positional_encoding = get_positional_encoding(seq_len, d_model)

print("Positional encoding for first 10 tokens and 10 embedding dim")
print(positional_encoding[:10, :10])

# Visualizing the positional encoding for the first two dimensions
plt.figure(figsize=(10, 6))
for i in range(0, 6):  # Visualize a few dimensions
    plt.plot(positional_encoding[:, i], label=f"Dim {i}")
plt.xlabel("Position")
plt.ylabel("Value")
plt.title("Positional Encoding for Different Dimensions")
plt.legend()
plt.show()
