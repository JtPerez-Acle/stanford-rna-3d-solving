"""
Tests for the base model classes.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from rna_folding.models.base import RNAModel, ModelConfig


def test_model_config_initialization():
    """Test that ModelConfig can be initialized with default values."""
    config = ModelConfig()
    assert config.embedding_dim > 0
    assert config.num_layers > 0
    assert config.dropout >= 0 and config.dropout < 1


def test_model_config_custom_values():
    """Test that ModelConfig can be initialized with custom values."""
    config = ModelConfig(
        embedding_dim=128,
        num_layers=4,
        dropout=0.2
    )
    assert config.embedding_dim == 128
    assert config.num_layers == 4
    assert config.dropout == 0.2


def test_rna_model_initialization():
    """Test that RNAModel can be initialized with a config."""
    config = ModelConfig()
    model = RNAModel(config)
    assert model.config == config
    assert hasattr(model, 'device')


def test_rna_model_to_device():
    """Test that RNAModel can be moved to a specific device."""
    config = ModelConfig()
    model = RNAModel(config)
    
    # Move to CPU (should work on all systems)
    model.to('cpu')
    assert model.device == torch.device('cpu')


def test_rna_model_save_load(tmp_path):
    """Test that RNAModel can be saved and loaded."""
    config = ModelConfig()
    model = RNAModel(config)
    
    # Save the model
    save_path = tmp_path / "test_model.pt"
    model.save(save_path)
    
    # Check that the file exists
    assert save_path.exists()
    
    # Load the model
    loaded_model = RNAModel.load(save_path)
    
    # Check that the config is the same
    assert loaded_model.config.embedding_dim == config.embedding_dim
    assert loaded_model.config.num_layers == config.num_layers
    assert loaded_model.config.dropout == config.dropout


def test_rna_model_forward_not_implemented():
    """Test that RNAModel's forward method raises NotImplementedError."""
    config = ModelConfig()
    model = RNAModel(config)
    
    with pytest.raises(NotImplementedError):
        model.forward(torch.randn(1, 10))
