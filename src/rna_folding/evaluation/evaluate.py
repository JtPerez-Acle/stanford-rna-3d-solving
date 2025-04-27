"""
Script to evaluate RNA 3D structure prediction models.

This script evaluates the performance of RNA 3D structure prediction models
by comparing predicted structures to known structures.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from rna_folding.evaluation.metrics import evaluate_predictions
from rna_folding.models.predict import load_model, predict_structure


def load_true_structures(structures_file):
    """
    Load true RNA 3D structures from a file.

    Args:
        structures_file (str or Path): Path to the structures CSV file.

    Returns:
        dict: Dictionary mapping RNA IDs to true 3D coordinates.
    """
    # Load structures
    structures = pd.read_csv(structures_file)

    # Check column names
    if 'ID' in structures.columns:
        # Extract RNA ID from the ID column (format: RNA_ID_POSITION)
        structures['rna_id'] = structures['ID'].apply(lambda x: x.split('_')[0])
        id_column = 'rna_id'
    elif 'target_id' in structures.columns:
        id_column = 'target_id'
    else:
        raise ValueError("Could not find RNA ID column in structures file")

    # Check coordinate columns
    if 'x' in structures.columns and 'y' in structures.columns and 'z' in structures.columns:
        x_col, y_col, z_col = 'x', 'y', 'z'
    elif 'x_1' in structures.columns and 'y_1' in structures.columns and 'z_1' in structures.columns:
        x_col, y_col, z_col = 'x_1', 'y_1', 'z_1'
    else:
        raise ValueError("Could not find coordinate columns in structures file")

    # Group by RNA ID
    true_structures = {}
    for rna_id, group in structures.groupby(id_column):
        # Extract coordinates
        coords = []
        for _, row in group.iterrows():
            coords.append([row[x_col], row[y_col], row[z_col]])

        # Store coordinates
        true_structures[rna_id] = np.array(coords)

    return true_structures


def evaluate_model(model, sequences_file, structures_file, output_file=None):
    """
    Evaluate a trained RNA 3D structure prediction model.

    Args:
        model: Trained model.
        sequences_file (str or Path): Path to the sequences CSV file.
        structures_file (str or Path): Path to the structures CSV file.
        output_file (str or Path, optional): Path to save evaluation results to.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Load sequences
    sequences = pd.read_csv(sequences_file)

    # Load true structures
    true_structures = load_true_structures(structures_file)

    # Initialize results
    results = []

    # Evaluate model on each sequence
    for _, row in tqdm(sequences.iterrows(), total=len(sequences), desc="Evaluating model"):
        rna_id = row['target_id']
        sequence = row['sequence']

        # Skip if true structure is not available
        if rna_id not in true_structures:
            print(f"Warning: True structure not available for {rna_id}")
            continue

        # Get true structure
        true_coords = true_structures[rna_id]

        # Generate predictions
        predictions = predict_structure(model, sequence, num_predictions=1)
        pred_coords = predictions[0]

        # Ensure predicted and true coordinates have the same shape
        min_len = min(len(true_coords), len(pred_coords))
        true_coords = true_coords[:min_len]
        pred_coords = pred_coords[:min_len]

        # Evaluate predictions
        metrics = evaluate_predictions(true_coords, pred_coords)

        # Add RNA ID to metrics
        metrics['rna_id'] = rna_id

        # Add to results
        results.append(metrics)

    # Calculate average metrics
    avg_metrics = {}
    for metric in ['rmsd', 'tm_score']:
        avg_metrics[metric] = np.mean([r[metric] for r in results])

    # Save results if output file is provided
    if output_file:
        # Save detailed results
        with open(output_file, 'w') as f:
            json.dump({
                'results': results,
                'avg_metrics': avg_metrics
            }, f, indent=2)

    return avg_metrics


def main():
    """Main function to evaluate a trained model."""
    parser = argparse.ArgumentParser(description="Evaluate a trained RNA 3D structure prediction model")
    parser.add_argument("--model-path", required=True, help="Path to the model checkpoint")
    parser.add_argument("--sequences-file", required=True, help="Path to the sequences CSV file")
    parser.add_argument("--structures-file", required=True, help="Path to the structures CSV file")
    parser.add_argument("--output-file", help="Path to save evaluation results to")

    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path)

    # Evaluate model
    avg_metrics = evaluate_model(
        model,
        args.sequences_file,
        args.structures_file,
        args.output_file
    )

    # Print average metrics
    print("\nAverage Metrics:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
