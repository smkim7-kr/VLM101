"""
Assymmetric quantization from scratch
"""

import numpy as np

# Suppress scientific notation for better readability
np.set_printoptions(suppress=True)

def generate_params():
    """
    Generate randomly distributed parameters and ensure specific values
    are included at the beginning for debugging purposes.
    """
    params = np.random.uniform(low=-50, high=150, size=20)
    params[0] = params.max() + 1 # convert first three elements for debugging
    params[1] = params.min() - 1
    params[2] = 0
    return np.round(params, 2)

def clamp(params_q: np.array, lower_bound: int, upper_bound: int) -> np.array:
    """
    Clamp the quantized parameters to be within the specified bounds.
    """
    params_q[params_q < lower_bound] = lower_bound
    params_q[params_q > upper_bound] = upper_bound
    return params_q

def asymmetric_quantization(params: np.array, bits: int) -> tuple:
    """
    Perform asymmetric quantization on the input parameters.
    
    Args:
    - params: The input array of parameters to be quantized.
    - bits: The number of bits for quantization.

    Returns:
    - A tuple containing the quantized parameters, scale, and zero point.
    """
    alpha, beta = np.max(params), np.min(params)
    scale = (alpha - beta) / (2**bits - 1)
    zero_point = -1 * np.round(beta / scale)
    
    quantized = np.round(params / scale) + zero_point
    quantized = clamp(quantized, 0, 2**bits - 1)
    
    return quantized, scale, int(zero_point)

def dequantize(quantized: np.array, scale: float, zero_point: int) -> np.array:
    """
    Dequantize the quantized parameters to recover the original scale.
    """
    return scale * (quantized - zero_point)

def main():
    params = generate_params()
    print("Original Parameters:")
    print(params)
    
    bits = 8  # You can change this to any number of bits you want to use for quantization (8, 16, 32...)
    quantized, scale, zero_point = asymmetric_quantization(params, bits)
    assert quantized[0] == 2**bits-1 # max value of params quantized to max int value of range
    assert quantized[1] == 0 # min value of params quantized to 0
    
    print("\nQuantized Parameters:")
    print(quantized)
    
    recovered_params = dequantize(quantized, scale, zero_point)
    print("\nRecovered Parameters after Dequantization:")
    print(recovered_params)

if __name__ == "__main__":
    main()
