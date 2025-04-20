"""
Main script to train and test FICNN model on synthetic data
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from models import FICNN
from utils.torch_utils import generate_synthetic_data, create_dataloader
from cfg.config import FICNN_CONFIG, TRAINING_CONFIG


def f(x):
    """
    Function to approximate: sum of squares
    
    Args:
        x (torch.Tensor): Input tensor
    
    Returns:
        torch.Tensor: Output tensor
    """
    return torch.sum(x * x, dim=-1)


def train_model(model, train_loader, num_epochs, learning_rate):
    """
    Train the model
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        num_epochs (int): Number of epochs to train for
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        list: Training losses per epoch
    """
    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track training loss
    training_losses = []
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_avg_loss = 0
        for data, target in train_loader:
            # Forward pass
            predicted = model(data)
            loss = loss_fn(predicted, target)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_avg_loss += loss.item() * data.size(0) / len(train_loader.dataset)
        
        training_losses.append(epoch_avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_avg_loss:.4f}')
    
    return training_losses


def visualize_approximation(model, target_function):
    """
    Visualize the approximation error of the model vs the target function
    
    Args:
        model (nn.Module): Trained model
        target_function (callable): Target function to compare against
    """
    # Create a grid for evaluation
    xg = np.linspace(-1, 1, 100)
    yg = np.linspace(-1, 1, 100)
    X_grid = np.dstack(np.meshgrid(xg, yg))
    X_tensor = torch.tensor(X_grid, dtype=torch.float32)
    
    # Evaluate model and target function
    with torch.no_grad():
        model_output = model(X_tensor)
    target_output = target_function(X_tensor)
    
    # Calculate absolute error
    error = np.abs(model_output.numpy() - target_output.numpy())
    
    # Create plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot model output
    xgrid, ygrid = np.meshgrid(xg, yg)
    contour1 = ax[0].contourf(xgrid, ygrid, model_output.numpy(), 100, cmap='viridis')
    ax[0].set_title('Model Output')
    fig.colorbar(contour1, ax=ax[0])
    
    # Plot target function
    contour2 = ax[1].contourf(xgrid, ygrid, target_output.numpy(), 100, cmap='viridis')
    ax[1].set_title('Target Function')
    fig.colorbar(contour2, ax=ax[1])
    
    # Plot error
    contour3 = ax[2].contourf(xgrid, ygrid, error, 100, cmap='hot')
    ax[2].set_title('Absolute Error')
    fig.colorbar(contour3, ax=ax[2])
    
    plt.tight_layout()
    plt.savefig('approximation_results.png')
    plt.show()


def main():
    """Main function to run the experiment"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    nr_samples = 10000
    dim = 2
    X, y = generate_synthetic_data(nr_samples, dim, f)
    
    # Create data loaders
    print("Creating data loaders...")
    batch_size = TRAINING_CONFIG['batch_size']
    train_loader, val_loader = create_dataloader(X, y, batch_size)
    
    # Initialize model
    print("Initializing model...")
    model = FICNN(
        in_dim=FICNN_CONFIG['in_dim'],
        out_dim=FICNN_CONFIG['out_dim'],
        n_layers=FICNN_CONFIG['n_layers'],
        hidden_dim=FICNN_CONFIG['hidden_dim']
    )
    
    # Train model
    print("Training model...")
    epochs = TRAINING_CONFIG['epochs']
    lr = TRAINING_CONFIG['lr']
    training_losses = train_model(model, train_loader, epochs, lr)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    
    # Visualize results
    print("Visualizing results...")
    visualize_approximation(model, f)
    
    # Save the model
    torch.save(model.state_dict(), 'ficnn_model.pth')
    print("Model saved to 'ficnn_model.pth'")


if __name__ == "__main__":
    main()