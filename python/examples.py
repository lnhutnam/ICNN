"""
Example script demonstrating the usage of all three DC neural network models:
- FICNN (Fully Input-Convex Neural Network)
- PICNN (Partially Input-Convex Neural Network)
- PICNN_Conv2d (Convolutional Partially Input-Convex Neural Network)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from models import FICNN, PICNN, PICNN_Conv2d
from cfg.config import FICNN_CONFIG, PICNN_CONFIG
from cfg.picnn_conv2d_config import PICNN_CONV2D_CONFIG


def demonstrate_ficnn():
    """
    Demonstrate FICNN functionality by fitting a simple convex function
    """
    print("\n" + "="*50)
    print("FICNN Demonstration")
    print("="*50)
    
    # Define a simple convex function: f(x) = x^2
    def f(x):
        return torch.sum(x * x, dim=-1)
    
    # Generate some data
    nr_samples = 1000
    dim = 2
    X = torch.rand((nr_samples, dim)) * 2 - 1  # Uniform in [-1, 1]
    y = f(X)
    
    # Create a FICNN model
    model = FICNN(
        in_dim=FICNN_CONFIG['in_dim'],
        out_dim=FICNN_CONFIG['out_dim'],
        n_layers=FICNN_CONFIG['n_layers'],
        hidden_dim=FICNN_CONFIG['hidden_dim']
    )
    
    # Train the model for a few iterations
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Create dataloader
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    print("Training FICNN model...")
    for epoch in range(100):  # Just a few epochs for demonstration
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            # Forward pass
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
        
        epoch_loss /= len(dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {epoch_loss:.6f}")
    
    # Visualize the learned function
    print("Visualizing learned function...")
    with torch.no_grad():
        # Create a grid of points
        x1 = np.linspace(-1, 1, 50)
        x2 = np.linspace(-1, 1, 50)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Reshape for prediction
        X_grid = np.stack((X1.flatten(), X2.flatten()), axis=1)
        X_tensor = torch.tensor(X_grid, dtype=torch.float32)
        
        # Get predictions
        y_pred = model(X_tensor).detach().numpy()
        
        # Reshape for plotting
        Y_pred = y_pred.reshape(X1.shape)
        
        # Calculate true values
        Y_true = f(X_tensor).numpy().reshape(X1.shape)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
        
        # Plot learned function
        surf1 = ax1.plot_surface(X1, X2, Y_pred, cmap='viridis', alpha=0.8)
        ax1.set_title('FICNN Learned Function')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('f(x)')
        
        # Plot true function
        surf2 = ax2.plot_surface(X1, X2, Y_true, cmap='plasma', alpha=0.8)
        ax2.set_title('True Function: f(x) = x²')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('f(x)')
        
        plt.tight_layout()
        plt.savefig('./figures/ficnn_example.png')
        plt.show()
    
    print("FICNN demonstration completed.")


def demonstrate_picnn():
    """
    Demonstrate PICNN functionality with a function that's convex in y but not in x
    """
    print("\n" + "="*50)
    print("PICNN Demonstration")
    print("="*50)
    
    # Define a function that's convex in y but not necessarily in x
    def f(x, y):
        # x component (non-convex): sin(x)
        x_part = torch.sin(torch.sum(x, dim=-1))
        # y component (convex): y^2
        y_part = torch.sum(y * y, dim=-1)
        return x_part + y_part
    
    # Generate some data
    nr_samples = 1000
    x_dim = PICNN_CONFIG['x_dim']
    y_dim = PICNN_CONFIG['y_dim']
    
    X = torch.rand((nr_samples, x_dim)) * 2 - 1  # Uniform in [-1, 1]
    Y = torch.rand((nr_samples, y_dim)) * 2 - 1  # Uniform in [-1, 1]
    targets = f(X, Y)
    
    # Create a PICNN model
    model = PICNN(
        x_dim=x_dim,
        y_dim=y_dim,
        n_layers=PICNN_CONFIG['n_layers'],
        u_dim=PICNN_CONFIG['u_dim'],
        z_dim=PICNN_CONFIG['z_dim']
    )
    
    # Train the model for a few iterations
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Create dataloader
    dataset = torch.utils.data.TensorDataset(X, Y, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    print("Training PICNN model...")
    for epoch in range(100):  # Just a few epochs for demonstration
        epoch_loss = 0.0
        for x_batch, y_batch, target_batch in dataloader:
            # Forward pass
            output = model(x_batch, y_batch).squeeze()
            loss = loss_fn(output, target_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
        
        epoch_loss /= len(dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {epoch_loss:.6f}")
    
    # Visualize the learned function for a fixed x
    print("Visualizing learned function...")
    with torch.no_grad():
        # Fix x and vary y
        x_fixed = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        
        # Create a grid of y values
        y1 = np.linspace(-1, 1, 50)
        y2 = np.linspace(-1, 1, 50)
        Y1, Y2 = np.meshgrid(y1, y2)
        
        # Reshape for prediction
        Y_grid = np.stack((Y1.flatten(), Y2.flatten()), axis=1)
        Y_tensor = torch.tensor(Y_grid, dtype=torch.float32)
        
        # Repeat x_fixed for each y point
        X_repeated = x_fixed.repeat(Y_tensor.shape[0], 1)
        
        # Get predictions
        Z_pred = model(X_repeated, Y_tensor).detach().numpy()
        
        # Reshape for plotting
        Z_pred = Z_pred.reshape(Y1.shape)
        
        # Calculate true values
        Z_true = f(X_repeated, Y_tensor).numpy().reshape(Y1.shape)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot learned function
        contour1 = ax1.contourf(Y1, Y2, Z_pred, 50, cmap='viridis')
        ax1.set_title(f'PICNN Output (Fixed x = {x_fixed.numpy().flatten()})')
        ax1.set_xlabel('y1')
        ax1.set_ylabel('y2')
        fig.colorbar(contour1, ax=ax1)
        
        # Plot true function
        contour2 = ax2.contourf(Y1, Y2, Z_true, 50, cmap='plasma')
        ax2.set_title(f'True Function (Fixed x = {x_fixed.numpy().flatten()})')
        ax2.set_xlabel('y1')
        ax2.set_ylabel('y2')
        fig.colorbar(contour2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig('./figures/picnn_example.png')
        plt.show()
        
        # 3D visualization
        fig = plt.figure(figsize=(15, 6))
        
        # Learned function
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(Y1, Y2, Z_pred, cmap='viridis', alpha=0.8)
        ax1.set_title(f'PICNN Output (Fixed x = {x_fixed.numpy().flatten()})')
        ax1.set_xlabel('y1')
        ax1.set_ylabel('y2')
        ax1.set_zlabel('f(x,y)')
        
        # True function
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(Y1, Y2, Z_true, cmap='plasma', alpha=0.8)
        ax2.set_title(f'True Function (Fixed x = {x_fixed.numpy().flatten()})')
        ax2.set_xlabel('y1')
        ax2.set_ylabel('y2')
        ax2.set_zlabel('f(x,y)')
        
        plt.tight_layout()
        plt.savefig('./figures/picnn_example_3d.png')
        plt.show()
    
    print("PICNN demonstration completed.")


def demonstrate_picnn_conv2d():
    """
    Demonstrate PICNN_Conv2d functionality with synthetic image data
    """
    print("\n" + "="*50)
    print("PICNN_Conv2d Demonstration")
    print("="*50)
    
    # Configuration
    image_size = 32
    in_channels = 1
    
    # Create a simple PICNN_Conv2d model with reduced complexity for demonstration
    model = PICNN_Conv2d(
        nr_channels=(16, 32, 64),  # Reduced channels
        kernel_sizes=(3, 3, 3),    # Smaller kernel size
        strides=(2, 2, 2),         # Same strides
        in_channels=in_channels,
    )
    
    # Generate synthetic image data
    print("Generating synthetic image data...")
    
    # Create a simple x input (random noise)
    x_image = torch.randn(1, in_channels, image_size, image_size)
    
    # Create a simple y input (gradient pattern)
    y_image = torch.zeros(1, in_channels, image_size, image_size)
    for h in range(image_size):
        for w in range(image_size):
            # Create a radial gradient
            dist = np.sqrt((h - image_size/2)**2 + (w - image_size/2)**2)
            y_image[0, 0, h, w] = torch.sigmoid(torch.tensor(10 * (1 - dist / (image_size/2))))
    
    # Forward pass through untrained model
    print("Running forward pass...")
    with torch.no_grad():
        output = model(x_image, y_image)
        print(f"Model output: {output.item()}")
    
    # Visualize inputs
    print("Visualizing inputs...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot x input
    ax1.imshow(x_image[0, 0].numpy(), cmap='gray')
    ax1.set_title('X Input (Non-convex)')
    ax1.axis('off')
    
    # Plot y input
    ax2.imshow(y_image[0, 0].numpy(), cmap='gray')
    ax2.set_title('Y Input (Convex)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('./figures/picnn_conv2d_example.png')
    plt.show()
    
    print("PICNN_Conv2d demonstration completed.")
    print("Note: For meaningful results, the model would need to be trained.")


def main():
    """
    Main function to run all demonstrations
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Demonstrating DC Neural Network models...")
    
    # Run demonstrations
    demonstrate_ficnn()
    demonstrate_picnn()
    demonstrate_picnn_conv2d()
    
    print("\nAll demonstrations completed.")


if __name__ == "__main__":
    main()