"""
Prediction script for RNA 3D structure prediction models.

This module provides functions for generating predictions with trained RNA 3D
structure prediction models, including ensemble predictions with uncertainty.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

from rna_folding.models.base import RNAModel
from rna_folding.models.multi_scale import MultiScaleRNA, MultiScaleModelConfig


def load_model(model_path):
    """
    Load a trained model.

    Args:
        model_path (str or Path): Path to the model checkpoint.

    Returns:
        RNAModel: Loaded model.
    """
    model_path = Path(model_path)

    # Load checkpoint with weights_only=False to handle PyTorch 2.6 compatibility
    try:
        # First try with weights_only=False (safer for loading our own models)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except (TypeError, ValueError, AttributeError) as e:
        # For older PyTorch versions that don't have weights_only parameter
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
        except Exception as e2:
            print(f"Error loading model: {str(e2)}")
            print("Trying to load with pickle safety disabled...")
            # Try with pickle safety disabled as a last resort
            import pickle
            with open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)

    # Handle case where we might have a direct state dict instead of a checkpoint
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        # Load config
        config = MultiScaleModelConfig.from_dict(checkpoint['config'])

        # Create model
        model = MultiScaleRNA(config)

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try to load the model config from a separate file
        config_path = model_path.parent / 'model_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = MultiScaleModelConfig.from_dict(config_dict)
            model = MultiScaleRNA(config)
            model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Could not load model config. Please ensure model_config.json exists in {model_path.parent}")

    return model


def predict_structure(model, sequence, num_predictions=5, device=None):
    """
    Predict 3D structure for an RNA sequence.

    Args:
        model (RNAModel): Trained model.
        sequence (str): RNA sequence.
        num_predictions (int): Number of predictions to generate.
        device (str, optional): Device to run prediction on.

    Returns:
        list: List of predicted 3D structures.
    """
    # Set device
    if device:
        model = model.to(device)

    # Generate predictions
    predictions = model.predict(sequence, num_predictions=num_predictions)

    return predictions


def predict_structures(model, sequences_file, output_file, num_predictions=5, device=None):
    """
    Predict 3D structures for RNA sequences in a file.

    Args:
        model (RNAModel): Trained model.
        sequences_file (str or Path): Path to the sequences CSV file.
        output_file (str or Path): Path to save predictions to.
        num_predictions (int): Number of predictions to generate.
        device (str, optional): Device to run prediction on.

    Returns:
        pandas.DataFrame: DataFrame containing predictions.
    """
    # Set device
    if device:
        model = model.to(device)

    # Load sequences
    sequences = pd.read_csv(sequences_file)

    # Initialize results
    results = []

    # Generate predictions for each sequence
    for _, row in tqdm(sequences.iterrows(), total=len(sequences), desc="Generating predictions"):
        target_id = row['target_id']
        sequence = row['sequence']

        # Generate predictions
        predictions = predict_structure(model, sequence, num_predictions=num_predictions)

        # Add predictions to results
        # First, create a dictionary for each nucleotide
        for j in range(len(sequence)):
            result = {
                'ID': f"{target_id}_{j+1}",
                'resname': sequence[j],
                'resid': j+1
            }

            # Add coordinates for each prediction
            for i, coords in enumerate(predictions):
                if j < len(coords):  # Make sure we have coordinates for this nucleotide
                    result[f'x_{i+1}'] = coords[j][0]
                    result[f'y_{i+1}'] = coords[j][1]
                    result[f'z_{i+1}'] = coords[j][2]

            results.append(result)

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Pivot to get the required format
    pivot_columns = ['ID', 'resname', 'resid']
    for i in range(1, num_predictions+1):
        pivot_columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

    # Ensure all columns are present
    for col in pivot_columns:
        if col not in results_df.columns:
            results_df[col] = np.nan

    # Select and order columns
    results_df = results_df[pivot_columns]

    # Save to file
    results_df.to_csv(output_file, index=False)

    return results_df


def main():
    """Main function to generate predictions with a trained model."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate predictions with a trained model")
    parser.add_argument("--model-path", required=True, help="Path to the model checkpoint")
    parser.add_argument("--sequences-file", required=True, help="Path to the sequences CSV file")
    parser.add_argument("--output-file", required=True, help="Path to save predictions to")
    parser.add_argument("--num-predictions", type=int, default=5, help="Number of predictions to generate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run prediction on")

    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path)

    # Generate predictions
    predict_structures(
        model,
        args.sequences_file,
        args.output_file,
        num_predictions=args.num_predictions,
        device=args.device
    )

    print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
