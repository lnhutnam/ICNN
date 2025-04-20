"""
Convolutional Partially Input-Convex Neural Network implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from nn.sp import SoftplusParameterization, positive_part


class PICNN_Conv2d(nn.Module):
    """
    Convolutional Partially Input-Convex Neural Network
    
    A convolutional variant of PICNN where convolution operations are used
    instead of linear layers for image processing tasks.
    """
    def __init__(
            self,
            nr_channels,        # tuple of channels
            kernel_sizes,       # tuple of kernel sizes
            strides,            # tuple of strides
            in_channels=1,
            activation_fn=nn.ReLU(),
    ):
        """
        Initialize a PICNN_Conv2d model
        
        Args:
            nr_channels (tuple): Number of channels for each layer
            kernel_sizes (tuple): Kernel sizes for each layer
            strides (tuple): Strides for each layer
            in_channels (int): Number of input channels
            activation_fn (nn.Module): Activation function
        """
        super().__init__()
        self.nr_channels = nr_channels
        self.kernel_size = kernel_sizes
        self.strides = strides
        self.activation_fn = activation_fn
        self.nr_layers = len(nr_channels)
        
        # Batch normalization layers
        bn = [nn.BatchNorm2d(num_features=nr_ch) for nr_ch in nr_channels]
        self.bn = nn.ModuleList(bn)
        
        # Initialize weight matrices
        Wbar = []   # u-path weights
        Wz = []     # z-path weights (must be non-negative)
        Wzu = []    # modulation of z by u
        Wy = []     # y-path weights
        Wyu = []    # modulation of y by u
        Wu = []     # direct u to z path
        
        # First layer
        layer = 0
        Wbar.append(nn.Conv2d(in_channels, nr_channels[layer], kernel_sizes[layer], stride=strides[layer]))
        Wzu.append(nn.Conv2d(in_channels, in_channels, 3, padding="same"))
        Wz.append(nn.Conv2d(in_channels, nr_channels[layer], kernel_sizes[layer], 
                           stride=strides[layer], bias=False))
        Wyu.append(nn.Conv2d(in_channels, 1, 3, padding="same"))
        Wy.append(nn.Conv2d(1, nr_channels[layer], kernel_sizes[layer], stride=strides[layer]))
        Wu.append(nn.Conv2d(in_channels, nr_channels[layer], kernel_sizes[layer], stride=strides[layer]))
        layer += 1
        
        # Apply softplus parametrization to ensure non-negativity
        register_parametrization(Wz[-1], "weight", SoftplusParameterization())
        
        # Remaining layers
        while layer < self.nr_layers:
            Wbar.append(nn.Conv2d(nr_channels[layer-1], nr_channels[layer], kernel_sizes[layer], 
                                stride=strides[layer]))
            Wzu.append(nn.Conv2d(nr_channels[layer-1], nr_channels[layer-1], 3, padding="same"))
            Wz.append(nn.Conv2d(nr_channels[layer-1], nr_channels[layer], kernel_sizes[layer], 
                               stride=strides[layer], bias=False))
            Wyu.append(nn.Conv2d(nr_channels[layer-1], 1, 3, padding="same"))
            Wy.append(nn.Conv2d(1, nr_channels[layer], kernel_sizes[layer], stride=strides[layer]))
            Wu.append(nn.Conv2d(nr_channels[layer-1], nr_channels[layer], kernel_sizes[layer], 
                               stride=strides[layer]))
            
            register_parametrization(Wz[-1], "weight", SoftplusParameterization())
            layer += 1
        
        # Register modulelists
        self.Wbar = nn.ModuleList(Wbar)
        self.Wz = nn.ModuleList(Wz)
        self.Wzu = nn.ModuleList(Wzu)
        self.Wy = nn.ModuleList(Wy)
        self.Wyu = nn.ModuleList(Wyu)
        self.Wu = nn.ModuleList(Wu)
        
        # Fully connected layers for final output
        self.fc_uz = nn.Linear(768, 768)
        self.fc_z = nn.Linear(768, 1, bias=False)
        self.fc_u = nn.Linear(768, 1)
        
        register_parametrization(self.fc_z, "weight", SoftplusParameterization())
    
    def forward(self, x, y):
        """
        Forward pass through the convolutional PICNN
        
        Args:
            x (torch.Tensor): Non-convex input tensor (image)
            y (torch.Tensor): Convex input tensor (image)
        
        Returns:
            torch.Tensor: Output tensor
        """
        u = x
        z = y
        
        # Process through convolutional layers
        for i in range(self.nr_layers):
            z = self.activation_fn(
                self.Wz[i](z * positive_part(self.Wzu[i](u))) +
                self.Wy[i](F.interpolate(y, size=self.Wyu[i](u).shape[2:]) * self.Wyu[i](u)) + 
                self.Wu[i](u)
            )
            u = self.bn[i](self.activation_fn(self.Wbar[i](u)))
        
        # Flatten for fully connected layers
        u = u.view(-1, 768)
        z = z.view(-1, 768)
        
        # Final output
        z = self.activation_fn(self.fc_z(positive_part(self.fc_uz(u)) * z) + self.fc_u(u))
        
        return z