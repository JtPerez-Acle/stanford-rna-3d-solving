"""
Evaluation metrics for RNA 3D structure prediction.

This module provides functions for evaluating the quality of predicted RNA 3D structures
using metrics such as RMSD (Root Mean Square Deviation) and TM-Score (Template Modeling Score).
"""

import numpy as np
from scipy.spatial import distance


def rmsd(coords1, coords2):
    """
    Calculate the Root Mean Square Deviation (RMSD) between two sets of coordinates.
    
    Args:
        coords1 (numpy.ndarray): First set of coordinates with shape (N, 3).
        coords2 (numpy.ndarray): Second set of coordinates with shape (N, 3).
        
    Returns:
        float: RMSD value.
    """
    # Ensure inputs are numpy arrays
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    
    # Check that the shapes match
    if coords1.shape != coords2.shape:
        raise ValueError(f"Coordinate shapes do not match: {coords1.shape} vs {coords2.shape}")
    
    # Calculate squared differences
    diff_sq = np.sum((coords1 - coords2) ** 2, axis=1)
    
    # Calculate RMSD
    return np.sqrt(np.mean(diff_sq))


def tm_score(coords1, coords2, d0=None):
    """
    Calculate the Template Modeling Score (TM-Score) between two sets of coordinates.
    
    Args:
        coords1 (numpy.ndarray): First set of coordinates with shape (N, 3).
        coords2 (numpy.ndarray): Second set of coordinates with shape (N, 3).
        d0 (float, optional): Normalization factor. If None, calculated as 1.24 * (N - 15)^(1/3) - 1.8.
        
    Returns:
        float: TM-Score value.
    """
    # Ensure inputs are numpy arrays
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    
    # Check that the shapes match
    if coords1.shape != coords2.shape:
        raise ValueError(f"Coordinate shapes do not match: {coords1.shape} vs {coords2.shape}")
    
    # Get number of residues
    N = coords1.shape[0]
    
    # Calculate d0 if not provided
    if d0 is None:
        d0 = 1.24 * (N - 15) ** (1/3) - 1.8
        d0 = max(d0, 0.5)  # Ensure d0 is at least 0.5
    
    # Calculate distances between corresponding residues
    distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
    
    # Calculate TM-Score
    tm_sum = np.sum(1.0 / (1.0 + (distances / d0) ** 2))
    tm = tm_sum / N
    
    return tm


def evaluate_predictions(true_coords, pred_coords):
    """
    Evaluate the quality of predicted RNA 3D structures.
    
    Args:
        true_coords (numpy.ndarray): True coordinates with shape (N, 3).
        pred_coords (numpy.ndarray): Predicted coordinates with shape (N, 3).
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Calculate RMSD
    rmsd_value = rmsd(true_coords, pred_coords)
    
    # Calculate TM-Score
    tm_value = tm_score(true_coords, pred_coords)
    
    # Return metrics
    return {
        'rmsd': rmsd_value,
        'tm_score': tm_value
    }
