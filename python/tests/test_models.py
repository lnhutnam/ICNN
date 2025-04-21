"""
Unit tests for DC neural network models
"""

import unittest
import torch
import sys
import os

# Add the parent directory to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import FICNN, PICNN, PICNN_Conv2d
from nn.sp import positive_part, SoftplusParameterization


class TestLayers(unittest.TestCase):
    """Test custom neural network layers"""
    
    def test_positive_part(self):
        """Test the positive_part function"""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        result = positive_part(x)
        self.assertTrue(torch.allclose(result, expected))
    
    def test_softplus_parameterization(self):
        """Test the SoftplusParameterization layer"""
        layer = SoftplusParameterization()
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = layer(x)
        
        # All outputs should be positive
        self.assertTrue((result > 0).all())
        
        # Output should increase monotonically
        self.assertTrue(torch.all(result[1:] > result[:-1]))


class TestFICNN(unittest.TestCase):
    """Test FICNN model"""
    
    def setUp(self):
        """Set up the test"""
        self.in_dim = 2
        self.out_dim = 1
        self.n_layers = 3
        self.hidden_dim = 4
        self.model = FICNN(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        # Check dimensions
        self.assertEqual(len(self.model.Wz), self.n_layers - 1)
        self.assertEqual(len(self.model.Wy), self.n_layers)
        
        # Check first layer
        self.assertEqual(self.model.Wy[0].in_features, self.in_dim)
        self.assertEqual(self.model.Wy[0].out_features, self.hidden_dim)
        
        # Check z weights are non-negative
        for layer_z in self.model.Wz:
            # Get the actual weight (through the parametrization)
            weight = layer_z.weight
            self.assertTrue((weight >= 0).all())
    
    def test_forward_pass(self):
        """Test the forward pass"""
        batch_size = 5
        x = torch.randn(batch_size, self.in_dim)
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size,))
        
        # Check output range (should be in [0, 1] due to sigmoid)
        self.assertTrue(((output >= 0) & (output <= 1)).all())


class TestPICNN(unittest.TestCase):
    """Test PICNN model"""
    
    def setUp(self):
        """Set up the test"""
        self.x_dim = 2
        self.y_dim = 2
        self.n_layers = 3
        self.u_dim = 4
        self.z_dim = 4
        self.model = PICNN(
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            n_layers=self.n_layers,
            u_dim=self.u_dim,
            z_dim=self.z_dim
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        # Check dimensions
        self.assertEqual(len(self.model.Wz), self.n_layers)
        self.assertEqual(len(self.model.Wy), self.n_layers)
        self.assertEqual(len(self.model.Wbar), self.n_layers - 1)
        
        # Check first layer dimensions
        self.assertEqual(self.model.Wbar[0].in_features, self.x_dim)
        self.assertEqual(self.model.Wbar[0].out_features, self.u_dim)
        self.assertEqual(self.model.Wz[0].in_features, self.y_dim)
        self.assertEqual(self.model.Wz[0].out_features, self.z_dim)
        
        # Check z weights are non-negative
        for layer_z in self.model.Wz:
            # Get the actual weight (through the parametrization)
            weight = layer_z.weight
            self.assertTrue((weight >= 0).all())
    
    def test_forward_pass(self):
        """Test the forward pass"""
        batch_size = 5
        x = torch.randn(batch_size, self.x_dim)
        y = torch.randn(batch_size, self.y_dim)
        output = self.model(x, y)
        
        # Check output shape - now should be [batch_size] due to the output_layer
        self.assertEqual(output.shape, (batch_size,))


class TestPICNNConv2d(unittest.TestCase):
    """Test PICNN_Conv2d model"""
    
    def setUp(self):
        """Set up the test"""
        self.nr_channels = (16, 32, 64)
        self.kernel_sizes = (3, 3, 3)
        self.strides = (2, 2, 2)
        self.in_channels = 1
        self.model = PICNN_Conv2d(
            nr_channels=self.nr_channels,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            in_channels=self.in_channels,
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        # Check dimensions
        self.assertEqual(len(self.model.Wz), len(self.nr_channels))
        self.assertEqual(len(self.model.Wy), len(self.nr_channels))
        self.assertEqual(len(self.model.Wbar), len(self.nr_channels))
        
        # Check first layer
        self.assertEqual(self.model.Wbar[0].in_channels, self.in_channels)
        self.assertEqual(self.model.Wbar[0].out_channels, self.nr_channels[0])
        
        # Check z weights are non-negative
        for layer_z in self.model.Wz:
            # Get the actual weight (through the parametrization)
            weight = layer_z.weight
            self.assertTrue((weight >= 0).all())
    
    def test_forward_pass(self):
        """Test the forward pass"""
        batch_size = 2
        image_size = 32
        x = torch.randn(batch_size, self.in_channels, image_size, image_size)
        y = torch.randn(batch_size, self.in_channels, image_size, image_size)
        
        try:
            output = self.model(x, y)
            # If we got here, the forward pass completed without errors
            self.assertEqual(output.shape, (batch_size, 1))
        except Exception as e:
            self.fail(f"Forward pass failed with error: {e}")


if __name__ == '__main__':
    unittest.main()