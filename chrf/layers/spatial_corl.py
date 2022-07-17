import torch
import torch.nn as nn


# TODOS: Implement torch based spatial correlation sample operation
def spatial_correlation_sample(
    input1, input2, kernel_size=3, padding=1, stride=1, dilation=1, patch_size=3, dilation_patch=1):
    raise ValueError


class SpatialCoRL(nn.Module):

    def __init__(self, kernel_size, padding, stride, dilation, patch_size, dilation_patch) -> None:
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.patch_size = patch_size
        self.dilation_patch = dilation_patch

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        assert query.dim == 4
        assert key.dim == 4

        # compute affinity matrix(correlation)
        return spatial_correlation_sample(
            query, key, self.kernel_size, self.padding, self.stride, self.dilation, 
            self.patch_size, self.dilation_patch)

