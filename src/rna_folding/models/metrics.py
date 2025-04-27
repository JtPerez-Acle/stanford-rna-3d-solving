"""
Metrics for evaluating RNA 3D structure prediction models.

This module provides functions for calculating various metrics to evaluate
the quality of predicted RNA 3D structures, with a focus on TM-score which
is the primary metric for the Stanford RNA 3D Folding competition.
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist


def rmsd(coords1, coords2):
    """
    Calculate Root Mean Square Deviation (RMSD) between two sets of coordinates.

    Args:
        coords1 (numpy.ndarray): First set of coordinates with shape (N, 3).
        coords2 (numpy.ndarray): Second set of coordinates with shape (N, 3).

    Returns:
        float: RMSD value.
    """
    try:
        # Ensure inputs are numpy arrays
        if isinstance(coords1, torch.Tensor):
            coords1 = coords1.detach().cpu().numpy()
        if isinstance(coords2, torch.Tensor):
            coords2 = coords2.detach().cpu().numpy()

        # Check for NaN values
        if np.isnan(coords1).any() or np.isnan(coords2).any():
            # Replace NaN values with zeros for calculation
            coords1_clean = np.nan_to_num(coords1, nan=0.0)
            coords2_clean = np.nan_to_num(coords2, nan=0.0)

            # Return a high RMSD to indicate poor prediction
            return 10.0
        else:
            coords1_clean = coords1
            coords2_clean = coords2

        # Check that shapes match
        if coords1_clean.shape != coords2_clean.shape:
            print(f"Warning: Coordinate shapes do not match: {coords1_clean.shape} vs {coords2_clean.shape}")
            return 10.0

        # Calculate RMSD
        squared_diff = np.sum((coords1_clean - coords2_clean) ** 2, axis=1)
        rmsd_value = np.sqrt(np.mean(squared_diff))

        # Check for invalid RMSD
        if np.isnan(rmsd_value) or np.isinf(rmsd_value):
            return 10.0

        return rmsd_value

    except Exception as e:
        print(f"Error calculating RMSD: {str(e)}")
        return 10.0


def tm_score(coords1, coords2, sequence_length=None):
    """
    Calculate Template Modeling Score (TM-score) between two sets of coordinates.

    TM-score is a measure of similarity between two protein structures with values
    in the range [0, 1], where 1 indicates a perfect match. It is more sensitive to
    the global fold than RMSD and is the primary metric for the Stanford RNA 3D
    Folding competition.

    Args:
        coords1 (numpy.ndarray): First set of coordinates with shape (N, 3).
        coords2 (numpy.ndarray): Second set of coordinates with shape (N, 3).
        sequence_length (int, optional): Length of the sequence. If None, uses the
            number of coordinates.

    Returns:
        float: TM-score value.
    """
    try:
        # Ensure inputs are numpy arrays
        if isinstance(coords1, torch.Tensor):
            coords1 = coords1.detach().cpu().numpy()
        if isinstance(coords2, torch.Tensor):
            coords2 = coords2.detach().cpu().numpy()

        # Check for NaN values
        if np.isnan(coords1).any() or np.isnan(coords2).any():
            # Replace NaN values with zeros for calculation
            coords1_clean = np.nan_to_num(coords1, nan=0.0)
            coords2_clean = np.nan_to_num(coords2, nan=0.0)

            # Return a low score to indicate poor prediction
            return 0.1
        else:
            coords1_clean = coords1
            coords2_clean = coords2

        # Check that shapes match
        if coords1_clean.shape != coords2_clean.shape:
            print(f"Warning: Coordinate shapes do not match: {coords1_clean.shape} vs {coords2_clean.shape}")
            return 0.0

        # Get sequence length
        L = sequence_length if sequence_length is not None else coords1_clean.shape[0]

        # Ensure L is positive
        if L <= 0:
            return 0.0

        # Calculate d0 (normalization factor)
        # Handle small sequence lengths to avoid complex numbers
        if L <= 15:
            d0 = 0.5  # Minimum value for d0
        else:
            d0 = 1.24 * (L - 15) ** (1/3) - 1.8
            d0 = max(d0, 0.5)  # Ensure d0 is at least 0.5

        # Calculate distances between all pairs of points
        distances = np.sqrt(np.sum((coords1_clean - coords2_clean) ** 2, axis=1))

        # Check for invalid distances
        if np.isnan(distances).any() or np.isinf(distances).any():
            return 0.0

        # Calculate TM-score
        tm_sum = np.sum(1.0 / (1.0 + (distances / d0) ** 2))
        tm = tm_sum / L

        # Ensure the score is in the valid range [0, 1]
        tm = max(0.0, min(1.0, tm))

        return tm

    except Exception as e:
        print(f"Error calculating TM-score: {str(e)}")
        return 0.0


def calculate_all_metrics(predicted_coords, true_coords, sequence=None):
    """
    Calculate all metrics for evaluating RNA 3D structure prediction.

    Args:
        predicted_coords (numpy.ndarray): Predicted coordinates with shape (N, 3).
        true_coords (numpy.ndarray): True coordinates with shape (N, 3).
        sequence (str, optional): RNA sequence.

    Returns:
        dict: Dictionary containing all metrics.
    """
    try:
        # Ensure inputs are numpy arrays
        if isinstance(predicted_coords, torch.Tensor):
            predicted_coords = predicted_coords.detach().cpu().numpy()
        if isinstance(true_coords, torch.Tensor):
            true_coords = true_coords.detach().cpu().numpy()

        # Check for empty or invalid inputs
        if (predicted_coords is None or true_coords is None or
            predicted_coords.size == 0 or true_coords.size == 0):
            return {
                'rmsd': 10.0,
                'tm_score': 0.0
            }

        # Calculate sequence length
        if sequence is not None:
            sequence_length = len(sequence)
        else:
            sequence_length = predicted_coords.shape[0]

        # Calculate metrics with error handling
        try:
            rmsd_value = rmsd(predicted_coords, true_coords)
        except Exception as e:
            print(f"Error calculating RMSD: {str(e)}")
            rmsd_value = 10.0

        try:
            tm_score_value = tm_score(predicted_coords, true_coords, sequence_length)
        except Exception as e:
            print(f"Error calculating TM-score: {str(e)}")
            tm_score_value = 0.0

        # Create metrics dictionary
        metrics = {
            'rmsd': rmsd_value,
            'tm_score': tm_score_value
        }

        return metrics

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'rmsd': 10.0,
            'tm_score': 0.0
        }
