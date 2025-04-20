"""
Utility functions for PyTorch operations
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def create_dataloader(x, y, batch_size, split_ratio=0.8, shuffle=True):
    """
    Create DataLoader objects for training and validation
    
    Args:
        x (torch.Tensor): Input features
        y (torch.Tensor): Target values
        batch_size (int): Batch size
        split_ratio (float): Ratio of data to use for training (default: 0.8)
        shuffle (bool): Whether to shuffle the data (default: True)
    
    Returns:
        tuple: (train_loader, val_loader) DataLoader objects
    """
    dataset = TensorDataset(x, y)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(dataset_size * split_ratio)
    val_size = dataset_size - train_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader


def generate_synthetic_data(num_samples, dim, function_to_approximate):
    """
    Generate synthetic data for training
    
    Args:
        num_samples (int): Number of samples to generate
        dim (int): Dimension of input features
        function_to_approximate (callable): Function to generate target values
    
    Returns:
        tuple: (features, targets) as torch.Tensors
    """
    # Generate random features in [-1, 1]
    features = torch.rand((num_samples, dim)) * 2 - 1
    
    # Apply the function to get targets
    targets = function_to_approximate(features)
    
    return features, targets