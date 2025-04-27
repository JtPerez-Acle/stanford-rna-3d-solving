#!/usr/bin/env python3
"""
Script to run the core RNA analysis module.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from src.rna_folding.analysis.core_analysis import RNAAnalysis

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run core RNA analysis")
    
    parser.add_argument(
        "--data-dir", 
        default="data/raw", 
        help="Directory containing the RNA data"
    )
    parser.add_argument(
        "--output-dir", 
        default=None, 
        help="Directory to save analysis results (default: data/analysis/YYYYMMDD)"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the core analysis."""
    args = parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        date_str = datetime.now().strftime("%Y%m%d")
        args.output_dir = f"data/analysis/{date_str}/core"
    
    # Create RNA analysis object
    analyzer = RNAAnalysis(args.data_dir, args.output_dir)
    
    # Run comprehensive analysis
    analyzer.generate_comprehensive_report()
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review the README.md file for key findings and recommendations")
    print("2. Explore the detailed analysis in the JSON files")
    print("3. Use the visualizations to gain insights into the RNA data")

if __name__ == "__main__":
    main()
