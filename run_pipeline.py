#!/usr/bin/env python3
"""
Script to run the entire RNA 3D folding pipeline.
"""

import os
import sys
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the RNA 3D folding pipeline")
    
    parser.add_argument(
        "--download", 
        action="store_true", 
        help="Download competition data"
    )
    parser.add_argument(
        "--analyze", 
        action="store_true", 
        help="Analyze the dataset"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true", 
        help="Visualize RNA structures"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=3, 
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--target-id", 
        help="Specific target ID to visualize"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the pipeline."""
    args = parse_args()
    
    # If no arguments provided, run the full pipeline
    if not (args.download or args.analyze or args.visualize):
        args.download = True
        args.analyze = True
        args.visualize = True
    
    # Download data
    if args.download:
        print("\n=== Downloading Data ===")
        os.system("python -m rna_folding download")
    
    # Analyze data
    if args.analyze:
        print("\n=== Analyzing Data ===")
        os.system("python -m rna_folding analyze")
    
    # Visualize data
    if args.visualize:
        print("\n=== Visualizing Data ===")
        cmd = "python -m rna_folding visualize"
        
        if args.target_id:
            cmd += f" --target-id {args.target_id}"
        else:
            cmd += f" --num-samples {args.num_samples}"
        
        os.system(cmd)
    
    print("\n=== Pipeline Complete ===")
    print("Results can be found in:")
    print("  - data/raw: Raw data")
    print("  - data/analysis: Analysis results")
    print("  - data/visualizations: Visualization results")

if __name__ == "__main__":
    main()
