"""
Configuration file for hyperparameters
"""

# FICNN model configuration
FICNN_CONFIG = {
    'in_dim': 2,
    'out_dim': 1,
    'n_layers': 4,
    'hidden_dim': 8,
}

# PICNN model configuration
PICNN_CONFIG = {
    'x_dim': 2,
    'y_dim': 2,
    'n_layers': 4,
    'u_dim': 8,
    'z_dim': 8,
}

# PICNN_Conv2d model configuration
PICNN_CONV2D_CONFIG = {
    'nr_channels': (32, 64, 128, 256),
    'kernel_sizes': (3, 3, 3, 3),
    'strides': (2, 2, 2, 2),
    'in_channels': 1,
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 500,
    'batch_size': 200,
    'lr': 0.001,
}