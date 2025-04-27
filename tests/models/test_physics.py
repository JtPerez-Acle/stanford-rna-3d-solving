"""
Tests for the physics-informed neural network layers.
"""

import pytest
import torch
import numpy as np

from rna_folding.models.physics import PhysicsConstraints, PhysicsInformedLoss


def test_physics_constraints_initialization():
    """Test initialization of PhysicsConstraints."""
    constraints = PhysicsConstraints()
    
    assert hasattr(constraints, 'c1_c1_min_distance')
    assert hasattr(constraints, 'c1_c1_max_distance')
    assert hasattr(constraints, 'c1_c1_mean_distance')
    assert hasattr(constraints, 'base_pair_distances')


def test_bond_length_energy():
    """Test bond length energy calculation."""
    constraints = PhysicsConstraints()
    
    # Create sample coordinates
    coords = torch.tensor([
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],  # Distance = 5.0 (equal to mean)
            [10.0, 0.0, 0.0]  # Distance = 5.0 (equal to mean)
        ]
    ], dtype=torch.float32)
    
    # Calculate energy
    energy = constraints.bond_length_energy(coords)
    
    # Energy should be close to 0 since distances are equal to mean
    assert energy.item() < 1e-5


def test_steric_clash_energy():
    """Test steric clash energy calculation."""
    constraints = PhysicsConstraints()
    
    # Create sample coordinates with no clashes
    coords_no_clash = torch.tensor([
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 5.0, 0.0]
        ]
    ], dtype=torch.float32)
    
    # Create sample coordinates with clashes
    coords_with_clash = torch.tensor([
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]  # Close to first atom (clash)
        ]
    ], dtype=torch.float32)
    
    # Calculate energies
    energy_no_clash = constraints.steric_clash_energy(coords_no_clash)
    energy_with_clash = constraints.steric_clash_energy(coords_with_clash)
    
    # Energy with clash should be higher
    assert energy_with_clash.item() > energy_no_clash.item()


def test_base_pairing_energy():
    """Test base pairing energy calculation."""
    constraints = PhysicsConstraints()
    
    # Create sample coordinates and sequence
    coords = torch.tensor([
        [
            [0.0, 0.0, 0.0],  # A
            [5.0, 0.0, 0.0],  # C
            [10.0, 0.0, 0.0], # G
            [15.0, 0.0, 0.0], # U
            [20.0, 0.0, 0.0]  # G
        ]
    ], dtype=torch.float32)
    
    sequence = ["ACGUG"]
    
    # Calculate energy
    energy = constraints.base_pairing_energy(coords, sequence)
    
    # Energy should be a non-negative value
    assert energy.item() >= 0.0


def test_physics_constraints_forward():
    """Test forward method of PhysicsConstraints."""
    constraints = PhysicsConstraints()
    
    # Create sample coordinates and sequence
    coords = torch.tensor([
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0]
        ]
    ], dtype=torch.float32)
    
    sequence = ["ACGU"]
    
    # Calculate energy terms
    energy_terms = constraints(coords, sequence)
    
    # Check that all energy terms are present
    assert 'bond_energy' in energy_terms
    assert 'steric_energy' in energy_terms
    assert 'base_pair_energy' in energy_terms
    assert 'total_energy' in energy_terms
    
    # Check that all energy terms are non-negative
    for term, value in energy_terms.items():
        assert value.item() >= 0.0


def test_physics_informed_loss():
    """Test PhysicsInformedLoss calculation."""
    loss_fn = PhysicsInformedLoss(prediction_weight=1.0, physics_weight=0.1)
    
    # Create sample predicted and true coordinates
    predicted_coords = torch.tensor([
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0]
        ]
    ], dtype=torch.float32)
    
    true_coords = torch.tensor([
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0]
        ]
    ], dtype=torch.float32)
    
    sequence = ["ACGU"]
    
    # Calculate loss
    total_loss, loss_components = loss_fn(predicted_coords, true_coords, sequence)
    
    # Check that all loss components are present
    assert 'prediction_loss' in loss_components
    assert 'physics_loss' in loss_components
    assert 'bond_energy' in loss_components
    assert 'steric_energy' in loss_components
    assert 'base_pair_energy' in loss_components
    assert 'total_loss' in loss_components
    
    # Prediction loss should be close to 0 since coordinates are identical
    assert loss_components['prediction_loss'].item() < 1e-5
    
    # Total loss should be equal to weighted sum of components
    expected_total = (
        loss_fn.prediction_weight * loss_components['prediction_loss'] +
        loss_fn.physics_weight * loss_components['physics_loss']
    )
    assert torch.isclose(total_loss, expected_total)
