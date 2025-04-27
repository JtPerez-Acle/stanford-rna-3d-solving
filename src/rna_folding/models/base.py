"""
Base classes for RNA 3D structure prediction models.

This module provides the foundation for building groundbreaking RNA 3D structure
prediction models, including base classes for model configuration and model implementation.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for RNA 3D structure prediction models."""
    
    # Model architecture
    embedding_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    
    # Model type and specific parameters
    model_type: str = "base"  # Options: "base", "transformer", "gnn", "multi_scale"
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path):
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class RNAModel(nn.Module):
    """
    Base class for RNA 3D structure prediction models.
    
    This class provides common functionality for all RNA 3D structure prediction models,
    including device management, saving/loading, and a common interface.
    """
    
    def __init__(self, config):
        """
        Initialize the RNA model.
        
        Args:
            config (ModelConfig): Model configuration.
        """
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input data.
            
        Returns:
            Model output.
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def to(self, device):
        """
        Move model to specified device.
        
        Args:
            device: Device to move model to.
            
        Returns:
            self: The model instance.
        """
        self.device = torch.device(device)
        return super().to(device)
    
    def save(self, path):
        """
        Save model to file.
        
        Args:
            path (str or Path): Path to save model to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state and config
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict()
        }, path)
    
    @classmethod
    def load(cls, path):
        """
        Load model from file.
        
        Args:
            path (str or Path): Path to load model from.
            
        Returns:
            RNAModel: Loaded model.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create model with saved config
        config = ModelConfig.from_dict(checkpoint['config'])
        model = cls(config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def predict(self, sequence, num_predictions=5):
        """
        Generate 3D structure predictions for an RNA sequence.
        
        Args:
            sequence (str): RNA sequence.
            num_predictions (int): Number of predictions to generate.
            
        Returns:
            list: List of predicted 3D structures.
        """
        raise NotImplementedError("Subclasses must implement predict method")
