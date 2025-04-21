"""
Partially Input-Convex Neural Network (PICNN) implementation
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization
from nn.sp import SoftplusParameterization, positive_part


class PICNN(nn.Module):
    """
    Partially Input-Convex Neural Network (PICNN)
    
    A neural network architecture where the output is convex with respect to
    some of its inputs (the y inputs), but not necessarily to others (the x inputs).
    
    References:
        Amos, B., Xu, L., & Kolter, J. Z. (2017). Input Convex Neural Networks.
        International Conference on Machine Learning, 146–155.
    """
    def __init__(self, x_dim=2, y_dim=2, n_layers=4, u_dim=8, z_dim=8,
                 activation_fn_u=nn.ReLU(), activation_fn_z=nn.ReLU()):
        """
        Initialize a PICNN model
        
        Args:
            x_dim (int): Dimension of non-convex input x
            y_dim (int): Dimension of convex input y
            n_layers (int): Number of layers
            u_dim (int): Hidden dimension size for u-path
            z_dim (int): Hidden dimension size for z-path
            activation_fn_u (nn.Module): Activation function for u-path
            activation_fn_z (nn.Module): Activation function for z-path
        """
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_layers = n_layers
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.activation_fn_u = activation_fn_u
        self.activation_fn_z = activation_fn_z
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Add output layer to convert from z_dim to scalar output
        self.output_layer = nn.Linear(z_dim, 1)
        
        # Initialize weight matrices
        Wbar = []   # u-path weights
        Wz = []     # z-path weights (must be non-negative)
        Wzu = []    # modulation of z by u
        Wy = []     # y-path weights (must be non-negative)
        Wyu = []    # modulation of y by u
        Wu = []     # direct u to z path
        
        # First layer
        Wbar.append(nn.Linear(x_dim, u_dim))
        Wz.append(nn.Linear(y_dim, z_dim, bias=False))
        register_parametrization(Wz[-1], "weight", SoftplusParameterization())
        Wzu.append(nn.Linear(x_dim, y_dim))
        Wy.append(nn.Linear(y_dim, z_dim, bias=False))
        Wyu.append(nn.Linear(x_dim, y_dim))
        Wu.append(nn.Linear(x_dim, z_dim))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            Wbar.append(nn.Linear(u_dim, u_dim))
            Wz.append(nn.Linear(z_dim, z_dim, bias=False))
            register_parametrization(Wz[-1], "weight", SoftplusParameterization())
            Wzu.append(nn.Linear(u_dim, z_dim))
            Wy.append(nn.Linear(y_dim, z_dim, bias=False))
            Wyu.append(nn.Linear(u_dim, y_dim))
            Wu.append(nn.Linear(u_dim, z_dim))
        
        # Output layer
        Wz.append(nn.Linear(z_dim, z_dim, bias=False))
        register_parametrization(Wz[-1], "weight", SoftplusParameterization())
        Wzu.append(nn.Linear(u_dim, z_dim))
        Wy.append(nn.Linear(y_dim, z_dim, bias=False))
        Wyu.append(nn.Linear(u_dim, y_dim))
        Wu.append(nn.Linear(u_dim, z_dim))
        
        # Register modulelists
        self.Wbar = nn.ModuleList(Wbar)
        self.Wz = nn.ModuleList(Wz)
        self.Wzu = nn.ModuleList(Wzu)
        self.Wy = nn.ModuleList(Wy)
        self.Wyu = nn.ModuleList(Wyu)
        self.Wu = nn.ModuleList(Wu)
    
    def forward(self, x, y):
        """
        Forward pass through the PICNN
        
        Args:
            x (torch.Tensor): Non-convex input tensor
            y (torch.Tensor): Convex input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        u = x
        z = y
        
        # Process through all layers except the last
        for i in range(self.n_layers - 1):
            # Update z-path
            z = self.activation_fn_z(
                self.Wz[i](z * positive_part(self.Wzu[i](u))) +
                self.Wy[i](y * self.Wyu[i](u)) +
                self.Wu[i](u)
            )
            # Update u-path
            u = self.activation_fn_u(self.Wbar[i](u))
        
        # Final layer (index i is still set from the loop)
        z = (self.Wz[i+1](z * positive_part(self.Wzu[i+1](u))) +
             self.Wy[i+1](y * self.Wyu[i+1](u)) +
             self.Wu[i+1](u))
        
        # Convert from z_dim to scalar output for each sample in the batch
        output = self.output_layer(z).squeeze(-1)
        
        return output