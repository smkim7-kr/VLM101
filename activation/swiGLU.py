import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def swish_beta(x, beta=1.0):
    """
    Swish with a tunable beta:
       swish_beta(x) = x * sigmoid(beta * x).
    """
    return x * sigmoid(beta * x)

def swiGLU_beta(x, beta=1.0):
    """
    A toy 1D 'swiGLU' style function:
       swiGLU_beta(x) = x * swish_beta(x, beta).
    In practice, you'd often do:
       x1, x2 = split(x, 2)
       return x1 * swish_beta(x2, beta)
    but for plotting, we reuse x for both.
    """
    return x * swish_beta(x, beta)

# -------------------------
# Demo: Plot for different betas
if __name__ == "__main__":
    x = np.linspace(-3, 3, 200)
    
    betas = [0.5, 1.0, 2.0, 5.0]
    plt.figure(figsize=(8, 6))
    
    for beta in betas:
        y = swiGLU_beta(x, beta)
        plt.plot(x, y, label=f"swiGLU, beta={beta}")
    
    plt.title("swiGLU(x) = x * [x * sigmoid(beta*x)] for various betas")
    plt.xlabel("x")
    plt.ylabel("swiGLU_beta(x)")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.show()
