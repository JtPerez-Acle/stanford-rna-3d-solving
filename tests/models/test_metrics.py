"""
Tests for the metrics module.
"""

import pytest
import numpy as np
import torch

from rna_folding.models.metrics import rmsd, tm_score, calculate_all_metrics


def test_rmsd_identical_structures():
    """Test RMSD calculation with identical structures."""
    coords = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    # RMSD of identical structures should be 0
    assert rmsd(coords, coords) == 0.0


def test_rmsd_different_structures():
    """Test RMSD calculation with different structures."""
    coords1 = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    coords2 = np.array([
        [2.0, 3.0, 4.0],  # Shifted by (1,1,1)
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0]
    ])
    
    # RMSD should be sqrt(3) = 1.732...
    assert np.isclose(rmsd(coords1, coords2), np.sqrt(3))


def test_rmsd_with_torch_tensors():
    """Test RMSD calculation with torch tensors."""
    coords1 = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    coords2 = torch.tensor([
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0]
    ])
    
    # RMSD should be sqrt(3) = 1.732...
    assert np.isclose(rmsd(coords1, coords2), np.sqrt(3))


def test_tm_score_identical_structures():
    """Test TM-score calculation with identical structures."""
    coords = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ])
    
    # TM-score of identical structures should be 1.0
    assert np.isclose(tm_score(coords, coords), 1.0)


def test_tm_score_with_sequence_length():
    """Test TM-score calculation with specified sequence length."""
    coords = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    # TM-score with sequence_length=10 should be different from default
    tm1 = tm_score(coords, coords)
    tm2 = tm_score(coords, coords, sequence_length=10)
    
    assert tm1 != tm2


def test_calculate_all_metrics():
    """Test calculation of all metrics."""
    coords1 = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    coords2 = np.array([
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0]
    ])
    
    metrics = calculate_all_metrics(coords1, coords2, sequence="AAA")
    
    assert 'rmsd' in metrics
    assert 'tm_score' in metrics
    assert np.isclose(metrics['rmsd'], np.sqrt(3))
