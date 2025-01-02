import numpy as np

# Suppress scientific notation for better readability
np.set_printoptions(suppress=True)

def generate_params(shape):
    """
    Generate randomly distributed parameters and ensure specific values
    are included at the beginning for debugging purposes.
    """
    params = np.random.uniform(low=-1, high=1, size=shape)
    params.flat[0] = params.max() + 0.1  # Set the first value for debugging
    if params.size > 1:
        params.flat[1] = params.min() - 0.1
    if params.size > 2:
        params.flat[2] = 0
    return np.round(params, 2)

def quantize_with_granularity(params: np.array, granularity: float) -> np.array:
    """
    Quantize the parameters with the specified granularity.
    
    Args:
    - params: The input array of parameters to be quantized.
    - granularity: The step size or granularity for quantization.

    Returns:
    - Quantized parameters.
    """
    quantized = np.round(params / granularity) * granularity
    return quantized

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def forward_pass(x: np.array, weights: list, biases: list) -> np.array:
    """
    Perform a forward pass through the network using the provided weights and biases.
    
    Args:
    - x: Input vector.
    - weights: List of weight matrices for each layer.
    - biases: List of bias vectors for each layer.

    Returns:
    - Output vector after passing through the network.
    """
    for w, b in zip(weights, biases):
        x = np.dot(x, w) + b
        x = gelu(x)  # ReLU activation
    return x

def main():
    # Define the network architecture
    input_size = 10
    hidden_layer_size = 20
    output_size = 2
    
    # Generate weights and biases for each layer
    weights_1 = generate_params((input_size, hidden_layer_size))
    weights_2 = generate_params((hidden_layer_size, output_size))
    bias_1 = generate_params((hidden_layer_size,))  # Shape should match the hidden layer size
    bias_2 = generate_params((output_size,))  # Shape should match the output layer size
    
    print("Original Weights Layer 1:")
    print(weights_1)
    print("\nOriginal Weights Layer 2:")
    print(weights_2)
    
    # Quantize each layer's weights and biases with different granularities
    granularity_1 = 0.1
    granularity_2 = 0.2
    
    quantized_weights_1 = quantize_with_granularity(weights_1, granularity_1)
    quantized_weights_2 = quantize_with_granularity(weights_2, granularity_2)
    quantized_bias_1 = quantize_with_granularity(bias_1, granularity_1)
    quantized_bias_2 = quantize_with_granularity(bias_2, granularity_2)
    
    print(f"\nQuantized Weights Layer 1 with Granularity {granularity_1}:")
    print(quantized_weights_1)
    print(f"\nQuantized Weights Layer 2 with Granularity {granularity_2}:")
    print(quantized_weights_2)
    
    print(f"\nQuantized Biases Layer 1 with Granularity {granularity_1}:")
    print(quantized_bias_1)
    print(f"\nQuantized Biases Layer 2 with Granularity {granularity_2}:")
    print(quantized_bias_2)
    
    # Perform a forward pass with original (non-quantized) weights
    x = np.random.rand(1, input_size)  # Example input
    original_output = forward_pass(x, [weights_1, weights_2], 
                                   [bias_1, bias_2])
    
    print("\nOutput after Forward Pass with Original Weights and Biases:")
    print(original_output)
    
    # Perform a forward pass with quantized weights
    quantized_output = forward_pass(x, [quantized_weights_1, quantized_weights_2], 
                                    [quantized_bias_1, quantized_bias_2])
    
    print("\nOutput after Forward Pass with Quantized Weights and Biases:")
    print(quantized_output)
    
    # Compare the results
    print("\nDifference between Original and Quantized Outputs:")
    print(original_output - quantized_output)

if __name__ == "__main__":
    main()
