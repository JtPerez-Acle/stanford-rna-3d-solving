"""
Tests for the multi-scale equivariant model.
"""

import pytest
import torch
import numpy as np

from rna_folding.models.multi_scale import (
    MultiScaleModelConfig,
    NucleotideEncoder,
    MotifEncoder,
    GlobalEncoder,
    StructureDecoder,
    MultiScaleRNA
)


def test_multi_scale_config():
    """Test MultiScaleModelConfig initialization."""
    config = MultiScaleModelConfig()
    
    assert config.model_type == "multi_scale"
    assert config.nucleotide_features > 0
    assert config.motif_features > 0
    assert config.global_features > 0
    assert config.num_layers_per_scale > 0


def test_nucleotide_encoder():
    """Test NucleotideEncoder forward pass."""
    config = MultiScaleModelConfig(
        nucleotide_features=32,
        num_layers_per_scale=2
    )
    
    encoder = NucleotideEncoder(config)
    
    # Create sample input (batch_size=2, seq_len=10, num_nucleotides=6)
    x = torch.zeros(2, 10, 6)
    for i in range(2):
        for j in range(10):
            x[i, j, j % 4] = 1.0  # One-hot encoding
    
    # Forward pass
    output = encoder(x)
    
    # Check output shape
    assert output.shape == (2, 10, config.nucleotide_features)


def test_motif_encoder():
    """Test MotifEncoder forward pass."""
    config = MultiScaleModelConfig(
        nucleotide_features=32,
        motif_features=64,
        num_layers_per_scale=2
    )
    
    encoder = MotifEncoder(config)
    
    # Create sample input (batch_size=2, seq_len=10, nucleotide_features=32)
    x = torch.randn(2, 10, config.nucleotide_features)
    
    # Forward pass
    output = encoder(x)
    
    # Check output shape
    assert output.shape == (2, 10, config.motif_features)


def test_global_encoder():
    """Test GlobalEncoder forward pass."""
    config = MultiScaleModelConfig(
        motif_features=64,
        global_features=128,
        num_layers_per_scale=2
    )
    
    encoder = GlobalEncoder(config)
    
    # Create sample input (batch_size=2, seq_len=10, motif_features=64)
    x = torch.randn(2, 10, config.motif_features)
    
    # Forward pass
    output = encoder(x)
    
    # Check output shape
    assert output.shape == (2, 10, config.global_features)


def test_structure_decoder():
    """Test StructureDecoder forward pass."""
    config = MultiScaleModelConfig(
        nucleotide_features=32,
        motif_features=64,
        global_features=128
    )
    
    decoder = StructureDecoder(config)
    
    # Create sample inputs
    nucleotide_features = torch.randn(2, 10, config.nucleotide_features)
    motif_features = torch.randn(2, 10, config.motif_features)
    global_features = torch.randn(2, 10, config.global_features)
    
    # Forward pass
    coords, uncertainty = decoder(nucleotide_features, motif_features, global_features)
    
    # Check output shapes
    assert coords.shape == (2, 10, 3)  # 3D coordinates
    assert uncertainty.shape == (2, 10, 3)  # Uncertainty for each coordinate


def test_multi_scale_rna_model():
    """Test MultiScaleRNA model forward pass."""
    config = MultiScaleModelConfig(
        nucleotide_features=32,
        motif_features=64,
        global_features=128,
        num_layers_per_scale=2
    )
    
    model = MultiScaleRNA(config)
    
    # Create sample input (batch_size=2, seq_len=10, num_nucleotides=6)
    x = torch.zeros(2, 10, 6)
    for i in range(2):
        for j in range(10):
            x[i, j, j % 4] = 1.0  # One-hot encoding
    
    # Forward pass
    coords, uncertainty = model(x)
    
    # Check output shapes
    assert coords.shape == (2, 10, 3)  # 3D coordinates
    assert uncertainty.shape == (2, 10, 3)  # Uncertainty for each coordinate


def test_multi_scale_rna_predict():
    """Test MultiScaleRNA predict method."""
    config = MultiScaleModelConfig(
        nucleotide_features=32,
        motif_features=64,
        global_features=128,
        num_layers_per_scale=2
    )
    
    model = MultiScaleRNA(config)
    
    # Create sample sequence
    sequence = "GGGAAACCC"
    
    # Generate predictions
    predictions = model.predict(sequence, num_predictions=3)
    
    # Check output
    assert len(predictions) == 3
    assert all(isinstance(pred, np.ndarray) for pred in predictions)
    assert all(pred.shape == (len(sequence), 3) for pred in predictions)
