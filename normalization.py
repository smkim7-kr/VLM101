import torch
import torch.nn as nn

# Example input: (batch_size=4, num_features=5)
x = torch.randn(4, 5)

# Batch Normalization
# Normalize (mean=0, std=1 along batch dim)
batch_norm = nn.BatchNorm1d(num_features=5)
batch_norm_output = batch_norm(x)

# Layer Normalization
# Normalize (mean=0, std=1 along feature dim)
layer_norm = nn.LayerNorm(normalized_shape=5)
layer_norm_output = layer_norm(x)

# RMS Normalization
# Root mean square (RMS) statistics are not dependent re-centering (mean = 0)
# Only use rescaling (var = 1)
# Requires less computions than LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms

#
rms_norm = RMSNorm(normalized_shape=5)
rms_norm_output = rms_norm(x)

# Print results for comparison
print("Input:\n", x)
print("\nBatch Normalization Output:\n", batch_norm_output)
assert torch.isclose(batch_norm_output[:, 0].sum(), torch.tensor(0.0), atol=1e-6)

print("\nLayer Normalization Output:\n", layer_norm_output)
assert torch.isclose(layer_norm_output[0, :].sum(), torch.tensor(0.0), atol=1e-6)

print("\nRMS Normalization Output:\n", rms_norm_output)
print(f"RMS row mean: {rms_norm_output[0, :].sum()} and std: {rms_norm_output[0, :].std()}")
