"""
quantization range - selecting [alpha, beta] range using percentile instead of min-max
this method is more robust to outliers
"""
import numpy as np

# Suppress scientific notation for better readability
np.set_printoptions(suppress=True)

def generate_params(size=10000, outlier_value=1000):
    """
    Generate a large array of randomly distributed parameters and introduce an outlier.
    """
    params = np.random.uniform(low=-50, high=150, size=size)
    params[-1] = outlier_value
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
    Perform asymmetric quantization using the min and max values of the parameters.
    """
    alpha = np.max(params)
    beta = np.min(params)
    scale = (alpha - beta) / (2**bits - 1)
    zero = -1 * np.round(beta / scale)
    
    quantized = np.round(params / scale + zero)
    quantized = clamp(quantized, 0, 2**bits - 1).astype(np.int32)
    
    return quantized, scale, int(zero)

def asymmetric_quantization_percentile(params: np.array, bits: int, percentile: float) -> tuple:
    """
    Perform asymmetric quantization using the given percentile of the parameters.
    """
    # this ignores extreme values (outliers) that are not within [1-percentile, percentile]
    alpha = np.percentile(params, 100 - percentile)
    beta = np.percentile(params, percentile)
    scale = (alpha - beta) / (2**bits - 1)
    zero = -1 * np.round(beta / scale)
    
    quantized = np.round(params / scale + zero)
    quantized = clamp(quantized, 0, 2**bits - 1).astype(np.int32)
    
    return quantized, scale, int(zero)

def dequantize(quantized: np.array, scale: float, zero: int) -> np.array:
    """
    Dequantize the quantized parameters to recover the original scale.
    """
    return scale * (quantized - zero)

def absolute_error(params: np.array, dequantized_params: np.array) -> np.float64: 
    """
    Calculate absolute difference between two numpy arrays
    """
    return np.sum(np.abs(params - dequantized_params))

def main():
    params = generate_params()
    print("Original Parameters:")
    print(params)

    bits = 8  # Number of bits for quantization
    percentile = 99.99  # Percentile for quantization

    # Perform standard asymmetric quantization
    quantized, scale, zero = asymmetric_quantization(params, bits)
    print("\nQuantized Parameters (Min-Max):")
    print(quantized)
    
    # Perform asymmetric quantization using percentile
    quantized_percentile, scale_percentile, zero_percentile = asymmetric_quantization_percentile(params, bits, percentile)
    print("\nQuantized Parameters (Percentile):")
    print(quantized_percentile)

    # Dequantize the results
    recovered_params = dequantize(quantized, scale, zero)
    recovered_params_percentile = dequantize(quantized_percentile, scale_percentile, zero_percentile)
    
    print("\nRecovered Parameters after Dequantization (Min-Max):")
    print(recovered_params)
    print("\nAbsolute difference between original parameters and recovered parameters (Min-Max):")
    print(absolute_error(params, recovered_params))
    
    print("\nRecovered Parameters after Dequantization (Percentile):")
    print(recovered_params_percentile)
    print("\nAbsolute difference between original parameters and recovered parameters (Percentile):")
    print(absolute_error(params, recovered_params_percentile))

if __name__ == "__main__":
    main()
