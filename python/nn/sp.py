"""
Custom neural network layers and utility functions
"""

import torch
import torch.nn as nn


def positive_part(x):
    """
    Returns the positive part of the input tensor x
    
    Args:
        x (torch.Tensor): Input tensor
    
    Returns:
        torch.Tensor: Positive part of x, i.e., max(x, 0)
    """
    return torch.maximum(x, torch.zeros_like(x))


class SoftplusParameterization(nn.Module):
    """
    Softplus parametrization to ensure positive weights
    Used with register_parametrization to enforce weight positivity
    """
    def forward(self, X):
        """
        Applies softplus activation to ensure positivity
        
        Args:
            X (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Softplus of X
        """
        return nn.functional.softplus(X)