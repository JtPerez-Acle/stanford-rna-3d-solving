#!/usr/bin/env python3
"""
Script to generate predictions with a trained RNA 3D structure prediction model.
"""

import os
import sys
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate predictions with a trained model")
    
    parser.add_argument(
        "--model-path", 
        required=True, 
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--sequences-file", 
        default="data/raw/test_sequences.csv", 
        help="Path to the sequences CSV file"
    )
    parser.add_argument(
        "--output-file", 
        default="submission.csv", 
        help="Path to save predictions to"
    )
    parser.add_argument(
        "--num-predictions", 
        type=int, 
        default=5, 
        help="Number of predictions to generate"
    )
    
    return parser.parse_args()

def main():
    """Main function to generate predictions with a trained model."""
    args = parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train a model first.")
        return 1
    
    # Check if sequences file exists
    sequences_file = Path(args.sequences_file)
    if not sequences_file.exists():
        print(f"Sequences file not found at {sequences_file}.")
        return 1
    
    # Create output directory if needed
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build command to run prediction
    cmd = [
        "python", "-m", "rna_folding.models.predict",
        f"--model-path={args.model_path}",
        f"--sequences-file={args.sequences_file}",
        f"--output-file={args.output_file}",
        f"--num-predictions={args.num_predictions}"
    ]
    
    # Run prediction
    cmd_str = " ".join(cmd)
    print(f"Running command: {cmd_str}")
    return os.system(cmd_str)

if __name__ == "__main__":
    sys.exit(main())
