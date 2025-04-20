"""
Fully Input-Convex Neural Network (FICNN) implementation
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization
from nn.sp import SoftplusParameterization


class FICNN(nn.Module):
    """
    Fully Input-Convex Neural Network (FICNN)
    
    A neural network architecture that ensures the output is a convex function of the input.
    This is achieved by constraining the weights of certain layers to be non-negative.
    
    References:
        Amos, B., Xu, L., & Kolter, J. Z. (2017). Input Convex Neural Networks.
        International Conference on Machine Learning, 146–155.
    """
    def __init__(self, in_dim=2, out_dim=1, n_layers=4, hidden_dim=8, activation_fn=nn.ReLU()):
        """
        Initialize a FICNN model
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            n_layers (int): Number of layers
            hidden_dim (int): Hidden dimension size
            activation_fn (nn.Module): Activation function
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        
        # Create weight matrices for z-path (Wz) and direct path (Wy)
        Wz = []
        Wy = []
        
        # First layer only has direct path
        Wy.append(nn.Linear(in_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            # z-path with non-negative weights
            Wz.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            register_parametrization(Wz[-1], "weight", SoftplusParameterization())
            # Direct path
            Wy.append(nn.Linear(in_dim, hidden_dim))
        
        # Output layer
        Wz.append(nn.Linear(hidden_dim, out_dim, bias=False))
        register_parametrization(Wz[-1], "weight", SoftplusParameterization())
        Wy.append(nn.Linear(in_dim, out_dim))
        
        # Register modulelists
        self.Wz = nn.ModuleList(Wz)
        self.Wy = nn.ModuleList(Wy)
    
    def forward(self, y, x=None):
        """
        Forward pass through the FICNN
        
        Args:
            y (torch.Tensor): Input tensor
            x (torch.Tensor, optional): Not used in this model, kept for API consistency
        
        Returns:
            torch.Tensor: Output tensor (sigmoid applied to squeeze to range [0,1])
        """
        # First layer
        z = self.Wy[0](y)
        
        # Hidden layers
        for (layer_y, layer_z) in zip(self.Wy[1:self.n_layers-1], self.Wz):
            z = self.activation_fn(layer_z(z) + layer_y(y))
        
        # Output layer
        z = self.Wz[-1](z) + self.Wy[-1](y)
        
        # Apply sigmoid and squeeze
        return nn.Sigmoid()(z.squeeze(-1))