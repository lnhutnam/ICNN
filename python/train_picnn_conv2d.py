"""
Script to train and test the PICNN_Conv2d model with GPU support
This example uses synthetic image data
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

from models import PICNN_Conv2d
from cfg.picnn_conv2d_config import PICNN_CONV2D_CONFIG, TRAINING_CONFIG, DATA_CONFIG, LOG_CONFIG


def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_synthetic_image_data(num_samples, image_size, in_channels):
    """
    Generate synthetic image data for training
    
    Args:
        num_samples (int): Number of samples to generate
        image_size (int): Size of the images (square)
        in_channels (int): Number of input channels
    
    Returns:
        tuple: (x_images, y_images, targets)
    """
    # Generate x images (non-convex input)
    x_images = torch.randn(num_samples, in_channels, image_size, image_size)
    
    # Generate y images (convex input) - simpler pattern
    y_images = torch.zeros(num_samples, in_channels, image_size, image_size)
    for i in range(num_samples):
        # Create simple patterns (e.g., gradients or circles)
        for c in range(in_channels):
            # Create a radial gradient
            center_x, center_y = image_size // 2, image_size // 2
            for h in range(image_size):
                for w in range(image_size):
                    dist = np.sqrt((h - center_x)**2 + (w - center_y)**2)
                    y_images[i, c, h, w] = torch.sigmoid(torch.tensor(10 * (1 - dist / (image_size // 2))))
    
    # Generate target values (some function of x and y)
    # Here we use a simple function: mean of x * mean of y + noise
    targets = torch.mean(x_images, dim=(1, 2, 3)) * torch.mean(y_images, dim=(1, 2, 3))
    targets = targets + 0.1 * torch.randn_like(targets)
    
    return x_images, y_images, targets


def prepare_dataloaders(x_images, y_images, targets, batch_size, split_ratio):
    """
    Prepare train and validation dataloaders
    
    Args:
        x_images (torch.Tensor): x input images
        y_images (torch.Tensor): y input images 
        targets (torch.Tensor): target values
        batch_size (int): batch size
        split_ratio (float): train/val split ratio
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create dataset
    dataset = TensorDataset(x_images, y_images, targets)
    
    # Split into train and validation
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer
        criterion (Loss): Loss function
        device (torch.device): Device to use
    
    Returns:
        float: Average training loss
    """
    model.train()
    train_loss = 0.0
    
    for batch_idx, (x_batch, y_batch, targets) in enumerate(train_loader):
        # Move data to device
        x_batch, y_batch, targets = x_batch.to(device), y_batch.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(x_batch, y_batch).squeeze()
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        train_loss += loss.item() * x_batch.size(0)
        
        # Log progress
        if (batch_idx + 1) % LOG_CONFIG['log_interval'] == 0:
            print(f'Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Calculate average loss
    train_loss /= len(train_loader.dataset)
    
    return train_loss


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion (Loss): Loss function
        device (torch.device): Device to use
    
    Returns:
        float: Validation loss
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for x_batch, y_batch, targets in val_loader:
            # Move data to device
            x_batch, y_batch, targets = x_batch.to(device), y_batch.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(x_batch, y_batch).squeeze()
            loss = criterion(outputs, targets)
            
            # Update metrics
            val_loss += loss.item() * x_batch.size(0)
    
    # Calculate average loss
    val_loss /= len(val_loader.dataset)
    
    return val_loss


def train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, device, scheduler=None):
    """
    Train the model
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs
        optimizer (Optimizer): Optimizer
        criterion (Loss): Loss function
        device (torch.device): Device to use
        scheduler (LRScheduler, optional): Learning rate scheduler
    
    Returns:
        dict: Training history
    """
    # Initialize metrics tracking
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Create checkpoints directory
    create_directory(LOG_CONFIG['checkpoint_dir'])
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        if (epoch + 1) % LOG_CONFIG['eval_interval'] == 0:
            val_loss = validate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % LOG_CONFIG['save_interval'] == 0:
            checkpoint_path = os.path.join(
                LOG_CONFIG['checkpoint_dir'], 
                f'picnn_conv2d_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': history['val_loss'][-1] if len(history['val_loss']) > 0 else None
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
    
    return history


def visualize_training_history(history):
    """
    Visualize training history
    
    Args:
        history (dict): Training history
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    
    # Plot validation loss if available
    if len(history['val_loss']) > 0:
        # Adjust x-axis for validation loss (it might be recorded less frequently)
        val_epochs = [i * LOG_CONFIG['eval_interval'] for i in range(len(history['val_loss']))]
        plt.plot(val_epochs, history['val_loss'], label='Val Loss')
    
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figures/picnn_conv2d_training_history.png')
    plt.close()  # Close the plot to prevent display in environments without GUI


def visualize_model_output(model, x_image, y_image, device):
    """
    Visualize model input and output
    
    Args:
        model (nn.Module): Trained model
        x_image (torch.Tensor): Input x image
        y_image (torch.Tensor): Input y image
        device (torch.device): Device to use
    """
    # Move data to device
    x_image = x_image.unsqueeze(0).to(device)
    y_image = y_image.unsqueeze(0).to(device)
    
    # Get model output
    model.eval()
    with torch.no_grad():
        output = model(x_image, y_image).item()
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot x input (move back to CPU for visualization)
    axs[0].imshow(x_image.squeeze().cpu().numpy(), cmap='gray')
    axs[0].set_title('Input X (Non-convex)')
    axs[0].axis('off')
    
    # Plot y input (move back to CPU for visualization)
    axs[1].imshow(y_image.squeeze().cpu().numpy(), cmap='gray')
    axs[1].set_title('Input Y (Convex)')
    axs[1].axis('off')
    
    plt.suptitle(f'Model Output: {output:.4f}')
    plt.tight_layout()
    plt.savefig('./figures/picnn_conv2d_visualization.png')
    plt.close()  # Close the plot to prevent display in environments without GUI


def main():
    """Main function"""
    # Set up GPU/CUDA device with fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Check and print CUDA details if using GPU
    if torch.cuda.is_available():
        print(f'CUDA Device Name: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Device Count: {torch.cuda.device_count()}')
        print(f'Current CUDA Device: {torch.cuda.current_device()}')
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Ensure reproducibility on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Configuration
    num_samples = 1000
    image_size = PICNN_CONV2D_CONFIG['image_size']
    in_channels = PICNN_CONV2D_CONFIG['in_channels']
    batch_size = TRAINING_CONFIG['batch_size']
    num_epochs = TRAINING_CONFIG['epochs']
    lr = TRAINING_CONFIG['lr']
    weight_decay = TRAINING_CONFIG['weight_decay']
    split_ratio = DATA_CONFIG['train_val_split']
    
    # Generate synthetic data
    print("Generating synthetic image data...")
    x_images, y_images, targets = generate_synthetic_image_data(
        num_samples, image_size, in_channels
    )
    
    # Prepare dataloaders
    print("Preparing dataloaders...")
    train_loader, val_loader = prepare_dataloaders(
        x_images, y_images, targets, batch_size, split_ratio
    )
    
    # Initialize model
    print("Initializing PICNN_Conv2d model...")
    model = PICNN_Conv2d(
        nr_channels=PICNN_CONV2D_CONFIG['nr_channels'],
        kernel_sizes=PICNN_CONV2D_CONFIG['kernel_sizes'],
        strides=PICNN_CONV2D_CONFIG['strides'],
        in_channels=PICNN_CONV2D_CONFIG['in_channels']
    ).to(device)
    
    # Wrap model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=TRAINING_CONFIG['lr_scheduler_step'],
        gamma=TRAINING_CONFIG['lr_scheduler_gamma']
    )
    
    # Train model
    print("Training model...")
    history = train_model(
        model, train_loader, val_loader, num_epochs,
        optimizer, criterion, device, scheduler
    )
    
    # Visualize training history
    visualize_training_history(history)
    
    # Visualize model output
    print("Visualizing model output...")
    # Get a sample from validation set and move to device
    x_sample, y_sample, _ = next(iter(val_loader))
    visualize_model_output(model, x_sample[0], y_sample[0], device)
    
    # Save final model
    # If using DataParallel, save the state dict of the underlying model
    final_path = os.path.join(LOG_CONFIG['checkpoint_dir'], 'picnn_conv2d_final.pth')
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), final_path)
    else:
        torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()