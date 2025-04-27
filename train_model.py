#!/usr/bin/env python3
"""
Script to train an RNA 3D structure prediction model.
"""

import os
import sys
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RNA 3D structure prediction model")

    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing the data"
    )
    parser.add_argument(
        "--output-dir",
        default="models/multi_scale",
        help="Directory to save model and results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to train on (auto, cpu, or cuda)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients"
    )
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Use memory-efficient training techniques"
    )
    parser.add_argument(
        "--model-type",
        choices=["multi_scale"],
        default="multi_scale",
        help="Type of model to train"
    )

    return parser.parse_args()

def main():
    """Main function to train an RNA 3D structure prediction model."""
    args = parse_args()

    # Check if data exists
    data_dir = Path(args.data_dir)
    train_seq_path = data_dir / "train_sequences.csv"
    train_labels_path = data_dir / "train_labels.csv"

    if not train_seq_path.exists() or not train_labels_path.exists():
        print(f"Data not found in {data_dir}. Please download the data first.")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command to run training
    cmd = [
        "python", "-m", "rna_folding.models.train",
        f"--data-dir={args.data_dir}",
        f"--output-dir={args.output_dir}",
        f"--batch-size={args.batch_size}",
        f"--num-epochs={args.num_epochs}",
        f"--learning-rate={args.learning_rate}",
        f"--weight-decay={args.weight_decay}",
        f"--num-workers={args.num_workers}",
        f"--early-stopping-patience={args.early_stopping_patience}"
    ]

    # Add device parameter if not auto
    if args.device != "auto":
        cmd.append(f"--device={args.device}")

    # Add gradient accumulation steps if not default
    if args.gradient_accumulation_steps > 1:
        cmd.append(f"--gradient-accumulation-steps={args.gradient_accumulation_steps}")

    # Add memory-efficient flag if enabled
    if args.memory_efficient:
        cmd.append("--memory-efficient")

    # Run training
    cmd_str = " ".join(cmd)
    print(f"Running command: {cmd_str}")
    return os.system(cmd_str)

if __name__ == "__main__":
    sys.exit(main())
