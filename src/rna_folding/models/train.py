"""
Training script for RNA 3D structure prediction models.

This module provides functions for training and evaluating RNA 3D structure
prediction models, including multi-scale equivariant architectures.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from rna_folding.models.base import RNAModel, ModelConfig
from rna_folding.models.multi_scale import MultiScaleRNA, MultiScaleModelConfig
from rna_folding.models.physics import PhysicsInformedLoss
from rna_folding.models.metrics import calculate_all_metrics
from rna_folding.models.data import RNADataset, create_data_loaders


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler=None,
    num_epochs=100,
    device='cuda',
    checkpoint_dir=None,
    early_stopping_patience=10,
    gradient_accumulation_steps=1,
    memory_efficient=False
):
    """
    Train an RNA 3D structure prediction model.

    Args:
        model (RNAModel): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        loss_fn (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        scheduler (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        num_epochs (int): Number of epochs to train for.
        device (str): Device to train on.
        checkpoint_dir (str or Path, optional): Directory to save checkpoints to.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating weights.
        memory_efficient (bool): Whether to use memory-efficient training techniques.

    Returns:
        dict: Dictionary containing training history.
    """
    # Move model to device
    model = model.to(device)

    # Create checkpoint directory if specified
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    # Print training configuration
    print(f"\nTraining configuration:")
    print(f"  • Device: {device}")
    print(f"  • Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  • Memory-efficient mode: {memory_efficient}")
    print(f"  • Early stopping patience: {early_stopping_patience}")

    # Training loop
    for epoch in range(num_epochs):
        try:
            # Training phase
            model.train()
            train_loss = 0.0

            # Progress bar for training
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

            # Reset gradients at the beginning of each epoch
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_pbar):
                try:
                    # Get data
                    sequence_encoding = batch['sequence_encoding'].to(device)
                    coordinates = batch['coordinates'].to(device)
                    mask = batch['mask'].to(device)

                    # Forward pass
                    predicted_coords, uncertainty = model(sequence_encoding)

                    # Apply mask to only consider actual sequence positions (not padding)
                    masked_pred_coords = predicted_coords * mask.unsqueeze(-1)
                    masked_coordinates = coordinates * mask.unsqueeze(-1)

                    # Calculate loss
                    loss, loss_components = loss_fn(masked_pred_coords, masked_coordinates, batch['sequence'])

                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                    # Backward pass
                    loss.backward()

                    # Update weights only after accumulating gradients
                    if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        # Add gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        # Update weights
                        optimizer.step()

                        # Reset gradients
                        optimizer.zero_grad()

                    # Update progress bar
                    train_loss += loss.item() * gradient_accumulation_steps  # Scale back for reporting
                    train_pbar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

                    # Free up memory if in memory-efficient mode
                    if memory_efficient:
                        del sequence_encoding, coordinates, mask, predicted_coords, uncertainty
                        del masked_pred_coords, masked_coordinates, loss, loss_components
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                except Exception as e:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    continue

            # Calculate average training loss
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_metrics_sum = None

            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_pbar):
                    try:
                        # Get data
                        sequence_encoding = batch['sequence_encoding'].to(device)
                        coordinates = batch['coordinates'].to(device)
                        mask = batch['mask'].to(device)

                        # Forward pass
                        predicted_coords, uncertainty = model(sequence_encoding)

                        # Apply mask to only consider actual sequence positions (not padding)
                        masked_pred_coords = predicted_coords * mask.unsqueeze(-1)
                        masked_coordinates = coordinates * mask.unsqueeze(-1)

                        # Calculate loss
                        loss, _ = loss_fn(masked_pred_coords, masked_coordinates, batch['sequence'])

                        # Calculate metrics - we need to handle the mask for metrics too
                        # Extract only the valid (non-padded) coordinates for each sequence
                        batch_metrics = []
                        for i in range(len(batch['sequence'])):
                            seq_len = mask[i].sum().item()
                            if seq_len > 0:
                                seq_pred = predicted_coords[i, :seq_len].cpu()
                                seq_true = coordinates[i, :seq_len].cpu()
                                seq_metrics = calculate_all_metrics(
                                    seq_pred,
                                    seq_true,
                                    batch['sequence'][i]
                                )
                                batch_metrics.append(seq_metrics)

                        # Average metrics across the batch
                        if batch_metrics:
                            metrics = {}
                            for key in batch_metrics[0].keys():
                                metrics[key] = sum(m[key] for m in batch_metrics) / len(batch_metrics)

                            # Update validation loss and metrics
                            val_loss += loss.item()

                            # Initialize val_metrics_sum if needed
                            if val_metrics_sum is None:
                                val_metrics_sum = {k: v for k, v in metrics.items()}
                            else:
                                for k, v in metrics.items():
                                    val_metrics_sum[k] += v

                            # Update progress bar
                            val_pbar.set_postfix({'loss': loss.item(), 'tm_score': metrics['tm_score']})

                        # Free up memory if in memory-efficient mode
                        if memory_efficient:
                            del sequence_encoding, coordinates, mask, predicted_coords, uncertainty
                            del masked_pred_coords, masked_coordinates, loss
                            if 'batch_metrics' in locals():
                                del batch_metrics
                            if 'metrics' in locals():
                                del metrics
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    except Exception as e:
                        print(f"\nError in validation batch {batch_idx}: {str(e)}")
                        continue

            # Calculate average validation loss and metrics
            if val_metrics_sum is not None:
                val_loss /= len(val_loader)
                val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}

                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)

                # Print epoch summary
                print(f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val TM-Score: {val_metrics['tm_score']:.4f}")

                # Update learning rate if scheduler is provided
                if scheduler:
                    scheduler.step(val_loss)

                # Save checkpoint if this is the best model so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_state = model.state_dict().copy()

                    if checkpoint_dir:
                        checkpoint_path = checkpoint_dir / f"best_model.pt"
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_metrics': val_metrics,
                            'config': model.config.to_dict()
                        }, checkpoint_path)

                        print(f"Saved best model checkpoint to {checkpoint_path}")

                # Early stopping
                if epoch - best_epoch >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                print(f"Warning: No valid metrics for epoch {epoch+1}. Skipping validation.")

        except Exception as e:
            print(f"\nError in epoch {epoch+1}: {str(e)}")
            print("Continuing to next epoch...")
            continue

    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)

    return history


def plot_training_history(history, output_dir=None):
    """
    Plot training history.

    Args:
        history (dict): Training history.
        output_dir (str or Path, optional): Directory to save plots to.
    """
    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    if output_dir:
        plt.savefig(output_dir / 'loss.png', dpi=300, bbox_inches='tight')

    plt.show()

    # Plot TM-score
    plt.figure(figsize=(10, 5))
    tm_scores = [metrics['tm_score'] for metrics in history['val_metrics']]
    plt.plot(tm_scores, label='Validation TM-Score')
    plt.xlabel('Epoch')
    plt.ylabel('TM-Score')
    plt.title('Validation TM-Score')
    plt.legend()
    plt.grid(True)

    if output_dir:
        plt.savefig(output_dir / 'tm_score.png', dpi=300, bbox_inches='tight')

    plt.show()


def main():
    """Main function to train an RNA 3D structure prediction model."""
    import argparse

    parser = argparse.ArgumentParser(description="Train RNA 3D structure prediction model")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing the data")
    parser.add_argument("--output-dir", default="models/multi_scale", help="Directory to save model and results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--memory-efficient", action="store_true", help="Use memory-efficient training techniques")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model with reduced size for memory efficiency
    # Check if we're running a small model (for limited memory environments)
    small_model = args.batch_size <= 4
    micro_model = args.batch_size == 1 and args.memory_efficient

    if micro_model:
        print("\nUsing ultra-memory-efficient micro model configuration")
        config = MultiScaleModelConfig(
            nucleotide_features=16,  # Ultra small
            motif_features=32,       # Ultra small
            global_features=64,      # Ultra small
            num_layers_per_scale=1,  # Minimum layers
            dropout=0.1,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size
        )
    elif small_model:
        print("\nUsing memory-efficient small model configuration")
        config = MultiScaleModelConfig(
            nucleotide_features=32,  # Reduced from 64
            motif_features=64,       # Reduced from 128
            global_features=128,     # Reduced from 256
            num_layers_per_scale=2,  # Reduced from 3
            dropout=0.1,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size
        )
    else:
        config = MultiScaleModelConfig(
            nucleotide_features=64,
            motif_features=128,
            global_features=256,
            num_layers_per_scale=3,
            dropout=0.1,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size
        )

    model = MultiScaleRNA(config)

    # Create loss function with a very small weight for physics constraints initially
    # This helps stabilize training in the beginning
    # Use an even smaller weight for micro models to ensure stability
    if micro_model:
        loss_fn = PhysicsInformedLoss(prediction_weight=1.0, physics_weight=0.0001)
    else:
        loss_fn = PhysicsInformedLoss(prediction_weight=1.0, physics_weight=0.01)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Determine if we should use memory-efficient training
    # For small batch sizes, we'll automatically enable memory-efficient mode
    memory_efficient = args.memory_efficient or args.batch_size <= 4

    # For small batch sizes, we'll also use gradient accumulation
    gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.batch_size <= 4 and gradient_accumulation_steps == 1:
        gradient_accumulation_steps = 4  # Accumulate gradients for effective batch size of 16
        print(f"\nAutomatically setting gradient accumulation steps to {gradient_accumulation_steps} for small batch size")

    # Train model
    history = train_model(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        num_epochs=args.num_epochs,
        device=args.device,
        checkpoint_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=gradient_accumulation_steps,
        memory_efficient=memory_efficient
    )

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        # Convert history to JSON-serializable format
        json_history = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'val_metrics': [
                {k: float(v) for k, v in metrics.items()}
                for metrics in history['val_metrics']
            ]
        }
        json.dump(json_history, f, indent=2)

    # Plot training history
    plot_training_history(history, output_dir)

    # Save model config
    config.save(output_dir / 'model_config.json')

    # Evaluate on test set
    model.eval()
    test_metrics = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            # Get data
            sequence_encoding = batch['sequence_encoding'].to(args.device)
            coordinates = batch['coordinates'].to(args.device)
            mask = batch['mask'].to(args.device)

            # Forward pass
            predicted_coords, uncertainty = model(sequence_encoding)

            # Calculate metrics - handle the mask for metrics
            batch_metrics = []
            for i in range(len(batch['sequence'])):
                seq_len = mask[i].sum().item()
                if seq_len > 0:
                    seq_pred = predicted_coords[i, :seq_len].cpu()
                    seq_true = coordinates[i, :seq_len].cpu()
                    seq_metrics = calculate_all_metrics(
                        seq_pred,
                        seq_true,
                        batch['sequence'][i]
                    )
                    batch_metrics.append(seq_metrics)

            # Average metrics across the batch
            if batch_metrics:
                avg_batch_metrics = {}
                for key in batch_metrics[0].keys():
                    avg_batch_metrics[key] = sum(m[key] for m in batch_metrics) / len(batch_metrics)
                test_metrics.append(avg_batch_metrics)

    # Calculate average test metrics
    avg_test_metrics = {}
    for metric in test_metrics[0].keys():
        avg_test_metrics[metric] = np.mean([m[metric] for m in test_metrics])

    # Save test metrics
    # Convert NumPy values to Python native types for JSON serialization
    serializable_metrics = {}
    for k, v in avg_test_metrics.items():
        if hasattr(v, 'item'):  # Check if it's a NumPy value or tensor
            serializable_metrics[k] = v.item()
        else:
            serializable_metrics[k] = float(v)

    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    # Print test metrics
    print("\nTest Metrics:")
    for metric, value in serializable_metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
