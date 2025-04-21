"""
Configuration for PICNN_Conv2d model and training
"""

# PICNN_Conv2d model configuration for image datasets
PICNN_CONV2D_CONFIG = {
    # Network architecture
    'nr_channels': (32, 64, 128, 256),  # Number of channels per layer
    'kernel_sizes': (3, 3, 3, 3),       # Kernel size per layer
    'strides': (2, 2, 2, 2),            # Stride per layer
    'in_channels': 1,                   # Input channels (e.g., 1 for grayscale)
    
    # Image dimensions
    'image_size': 32,                   # Input image size (assumed square)
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 50,                       # Number of training epochs
    'batch_size': 64,                   # Batch size
    'lr': 0.0001,                       # Learning rate
    'weight_decay': 1e-5,               # Weight decay for regularization
    'lr_scheduler_step': 10,            # Learning rate scheduler step size
    'lr_scheduler_gamma': 0.5,          # Learning rate decay factor
}

# Data configuration
DATA_CONFIG = {
    'train_val_split': 0.8,             # Train/validation split ratio
    'normalize': True,                  # Whether to normalize the data
    'augment': True,                    # Whether to use data augmentation
}

# Logging and checkpoint configuration
LOG_CONFIG = {
    'log_interval': 10,                 # Log interval (batches)
    'eval_interval': 1,                 # Evaluation interval (epochs)
    'save_interval': 5,                 # Model save interval (epochs)
    'checkpoint_dir': 'checkpoints',    # Directory for model checkpoints
}