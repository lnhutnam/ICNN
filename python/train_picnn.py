"""
Script to train and evaluate a PICNN model with GPU support
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models import PICNN
from utils.data_utils import generate_synthetic_data, create_dataloader
from cfg.config import PICNN_CONFIG, TRAINING_CONFIG


def get_device():
    """
    Determine the available device (GPU or CPU)
    
    Returns:
        torch.device: Device to use for computations
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def f_joint(x, y):
    """
    Function to approximate: joint function of x and y
    x is non-convex input, y is convex input
    
    Args:
        x (torch.Tensor): Non-convex input tensor
        y (torch.Tensor): Convex input tensor
    
    Returns:
        torch.Tensor: Output tensor
    """
    # The output is convex in y but not necessarily in x
    return torch.sum(y * y, dim=-1) + torch.sin(3 * torch.sum(x, dim=-1))


def prepare_data(num_samples, x_dim, y_dim, device):
    """
    Prepare synthetic data for PICNN
    
    Args:
        num_samples (int): Number of samples
        x_dim (int): Dimension of x (non-convex input)
        y_dim (int): Dimension of y (convex input)
        device (torch.device): Device to place tensors on
    
    Returns:
        tuple: (x_data, y_data, targets)
    """
    # Generate random x and y data in [-1, 1]
    x_data = torch.rand((num_samples, x_dim), device=device) * 2 - 1
    y_data = torch.rand((num_samples, y_dim), device=device) * 2 - 1
    
    # Calculate targets
    targets = f_joint(x_data, y_data)
    
    return x_data, y_data, targets


def train_picnn(model, x_data, y_data, targets, num_epochs, batch_size, learning_rate, device):
    """
    Train the PICNN model
    
    Args:
        model (PICNN): Model to train
        x_data (torch.Tensor): Non-convex input data
        y_data (torch.Tensor): Convex input data
        targets (torch.Tensor): Target values
        num_epochs (int): Number of epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        device (torch.device): Device to use for training
    
    Returns:
        list: Training losses
    """
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(x_data, y_data, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move model to device
    model.to(device)
    
    # Training loop
    losses = []
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch, target_batch in dataloader:
            # Move batches to device if not already there
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Forward pass
            output = model(x_batch, y_batch)
            
            # Handle output shape
            if output.dim() > 1:
                # If output has more than 1 dimension, reduce to match target
                output = output.squeeze()  # Remove all dimensions of size 1
            
            # If still multiple dimensions, take mean
            if output.dim() > 1:
                output = output.mean(dim=-1)
            
            loss = loss_fn(output, target_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
        
        # Average loss for the epoch
        epoch_loss /= len(dataset)
        losses.append(epoch_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    return losses


def visualize_picnn_results(model, x_fixed, device):
    """
    Visualize the output of PICNN for a fixed x and varying y
    
    Args:
        model (PICNN): Trained model
        x_fixed (torch.Tensor): Fixed non-convex input
        device (torch.device): Device to use for computation
    """
    # Create a grid of y values
    y1 = np.linspace(-1, 1, 50)
    y2 = np.linspace(-1, 1, 50)
    y_grid = np.meshgrid(y1, y2)
    
    # Create the coordinate matrices
    Y1, Y2 = np.meshgrid(y1, y2)
    
    # Initialize output grid
    Z = np.zeros_like(Y1)
    
    # Evaluate model at each point in the grid
    model.eval()
    with torch.no_grad():
        for i in range(len(y1)):
            for j in range(len(y2)):
                y = torch.tensor([[y1[i], y2[j]]], dtype=torch.float32, device=device)
                x = x_fixed.to(device).repeat(y.shape[0], 1)
                Z[j, i] = model(x, y).cpu().item()
    
    # Plot the results
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(Y1, Y2, Z, cmap=plt.cm.viridis, linewidth=0)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_xlabel('y1')
    ax1.set_ylabel('y2')
    ax1.set_zlabel('f(x, y)')
    ax1.set_title('PICNN Output (3D)')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(Y1, Y2, Z, 50, cmap=plt.cm.viridis)
    fig.colorbar(contour, ax=ax2)
    ax2.set_xlabel('y1')
    ax2.set_ylabel('y2')
    ax2.set_title('PICNN Output (Contour)')
    
    plt.tight_layout()
    plt.savefig('./figures/picnn_visualization.png')
    plt.show()


def main():
    """Main function to run PICNN experiment"""
    # Determine device
    device = get_device()
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)  # For CUDA
    
    # Configuration
    x_dim = PICNN_CONFIG['x_dim']
    y_dim = PICNN_CONFIG['y_dim']
    n_layers = PICNN_CONFIG['n_layers']
    u_dim = PICNN_CONFIG['u_dim']
    z_dim = PICNN_CONFIG['z_dim']
    num_epochs = TRAINING_CONFIG['epochs']
    batch_size = TRAINING_CONFIG['batch_size']
    learning_rate = TRAINING_CONFIG['lr']
    
    # Prepare data
    print("Preparing data...")
    nr_samples = 10000
    x_data, y_data, targets = prepare_data(nr_samples, x_dim, y_dim, device)
    
    # Initialize model
    print("Initializing PICNN model...")
    model = PICNN(
        x_dim=x_dim,
        y_dim=y_dim,
        n_layers=n_layers,
        u_dim=u_dim,
        z_dim=z_dim
    )
    
    # Train model
    print("Training PICNN model...")
    losses = train_picnn(model, x_data, y_data, targets, num_epochs, batch_size, learning_rate, device)
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('PICNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('./figures/picnn_loss.png')
    
    # Visualize results for a fixed x
    print("Visualizing results...")
    x_fixed = torch.tensor([[0.5, -0.5]], dtype=torch.float32, device=device)  # Fixed value for x
    visualize_picnn_results(model, x_fixed, device)
    
    # Save model
    torch.save(model.state_dict(), './checkpoints/picnn_model.pth')
    print("Model saved to 'picnn_model.pth'")


if __name__ == "__main__":
    main()