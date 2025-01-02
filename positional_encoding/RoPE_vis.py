import numpy as np
import matplotlib.pyplot as plt

def rotary_positional_embedding_2d(position, base_freq=1e-2, magnitude=1.0):
    """
    Returns a 2D rotary positional embedding for a single position, scaled by 'magnitude'.

    Args:
        position (int) : The discrete sequence position.
        base_freq (float): A base frequency that determines how quickly the angle
                           advances with position.
        magnitude (float): A scale factor applied to the vector length.

    Returns:
        np.ndarray of shape (2,) -- (x, y) coordinates of the embedding.
    """
    
    # angle is in radian
    # base_freq tells how many rotations to complete per discrete positon step
    # larger base_freq implies larger rotation per position step
    angle = position * base_freq * 2.0 * np.pi  # angle grows with position
    x = magnitude * np.cos(angle)
    y = magnitude * np.sin(angle)
    return np.array([x, y])


def visualize_two_rotary_embeddings(num_positions=6,
                                    base_freq_1=0.1, magnitude_1=1.0,
                                    base_freq_2=0.15, magnitude_2=2.0):
    """
    Plots two sets of 2D rotary embeddings (A and B) for positions [0..num_positions-1].
    Each set has its own base frequency and magnitude, showing different rotation speeds
    AND different vector lengths in the same figure.
    """

    # Create a figure with equal aspect ratio
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Generate and plot vectors for each position
    for pos in range(num_positions):
        # Vector A: rotate with base_freq_1, length = magnitude_1
        vecA = rotary_positional_embedding_2d(pos, base_freq_1, magnitude_1)
        plt.arrow(
            0, 0, vecA[0], vecA[1],
            head_width=0.05, length_includes_head=True,
            color="blue", alpha=0.6
        )
        plt.text(
            vecA[0] * 1.05, vecA[1] * 1.05,
            f"A p={pos}",
            fontsize=8, ha='center', va='center', color='blue'
        )

        # Vector B: rotate with base_freq_2, length = magnitude_2
        vecB = rotary_positional_embedding_2d(pos, base_freq_2, magnitude_2)
        plt.arrow(
            0, 0, vecB[0], vecB[1],
            head_width=0.05, length_includes_head=True,
            color="red", alpha=0.6
        )
        plt.text(
            vecB[0] * 1.05, vecB[1] * 1.05,
            f"B p={pos}",
            fontsize=8, ha='center', va='center', color='red'
        )

    plt.title("Two Sets of 2D Rotary Positional Embeddings (Different Lengths & Angles)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    # Set axis ranges a bit larger to accommodate bigger arrows
    max_len = max(magnitude_1, magnitude_2)
    plt.xlim(-1.2*max_len, 1.2*max_len)
    plt.ylim(-1.2*max_len, 1.2*max_len)
    plt.show()


if __name__ == "__main__":
    # By increasing magnitude_2, B’s vectors appear longer than A’s.
    visualize_two_rotary_embeddings(
        num_positions=8,
        base_freq_1=0.12, magnitude_1=1.0,
        base_freq_2=0.18, magnitude_2=2.0
    )
