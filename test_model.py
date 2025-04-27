#!/usr/bin/env python3
"""
Script to test the RNA 3D structure prediction model with minimal resources.
This script uses a small subset of data and reduced model size to verify functionality.
"""

import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path
import time

from rna_folding.models.multi_scale import MultiScaleRNA, MultiScaleModelConfig
from rna_folding.models.physics import PhysicsInformedLoss
from rna_folding.models.data import RNADataset, rna_collate_fn
from torch.utils.data import DataLoader, Subset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test RNA 3D structure prediction model with minimal resources")
    
    parser.add_argument(
        "--data-dir", 
        default="data/raw", 
        help="Directory containing the data"
    )
    parser.add_argument(
        "--output-dir", 
        default="models/test", 
        help="Directory to save test results"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=10, 
        help="Number of samples to use for testing"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=2, 
        help="Batch size"
    )
    parser.add_argument(
        "--num-epochs", 
        type=int, 
        default=2, 
        help="Number of epochs"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (auto, cpu, or cuda)"
    )
    
    return parser.parse_args()


def test_model(args):
    """Test the model with minimal resources."""
    print(f"Testing model with {args.num_samples} samples, batch size {args.batch_size}, and {args.num_epochs} epochs")
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load a small subset of data
    data_dir = Path(args.data_dir)
    sequences_file = data_dir / "train_sequences.csv"
    labels_file = data_dir / "train_labels.csv"
    
    if not sequences_file.exists() or not labels_file.exists():
        print(f"Data files not found in {data_dir}")
        return 1
    
    # Create dataset
    dataset = RNADataset(sequences_file, labels_file)
    
    # Use only a small subset of the data
    subset_indices = list(range(min(args.num_samples, len(dataset))))
    subset_dataset = Subset(dataset, subset_indices)
    
    # Create data loader
    data_loader = DataLoader(
        subset_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers to minimize memory usage
        collate_fn=rna_collate_fn
    )
    
    # Create a small model
    config = MultiScaleModelConfig(
        nucleotide_features=32,  # Reduced from default 64
        motif_features=64,       # Reduced from default 128
        global_features=128,     # Reduced from default 256
        num_layers_per_scale=2,  # Reduced from default 3
        dropout=0.1,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=args.batch_size
    )
    
    model = MultiScaleRNA(config)
    model.to(device)
    
    # Create loss function with minimal physics weight
    loss_fn = PhysicsInformedLoss(prediction_weight=1.0, physics_weight=0.001)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(data_loader):
            # Get data
            sequence_encoding = batch['sequence_encoding'].to(device)
            coordinates = batch['coordinates'].to(device)
            mask = batch['mask'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predicted_coords, uncertainty = model(sequence_encoding)
            
            # Apply mask to only consider actual sequence positions (not padding)
            masked_pred_coords = predicted_coords * mask.unsqueeze(-1)
            masked_coordinates = coordinates * mask.unsqueeze(-1)
            
            # Calculate loss
            loss, loss_components = loss_fn(masked_pred_coords, masked_coordinates, batch['sequence'])
            
            # Backward pass and optimize
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            
            # Print progress
            print(f"Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx+1}/{len(data_loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Pred Loss: {loss_components['prediction_loss'].item():.4f}, "
                  f"Physics Loss: {loss_components['physics_loss'].item():.4f}")
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    
    # Save model
    model_path = output_dir / "test_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict()
    }, model_path)
    
    print(f"Model saved to {model_path}")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
            # Get data
            sequence_encoding = batch['sequence_encoding'].to(device)
            
            # Forward pass
            start_time = time.time()
            predicted_coords, uncertainty = model(sequence_encoding)
            inference_time = time.time() - start_time
            
            print(f"Inference time for batch of {args.batch_size}: {inference_time:.4f} seconds")
            print(f"Output shape: {predicted_coords.shape}")
            break  # Only test one batch
    
    return 0


def main():
    """Main function."""
    args = parse_args()
    return test_model(args)


if __name__ == "__main__":
    sys.exit(main())
