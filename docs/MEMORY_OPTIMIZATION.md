# Memory Optimization Techniques

This document explains the memory optimization techniques used in the RNA 3D structure prediction model to enable training on systems with limited memory.

## Overview

Training deep learning models for RNA 3D structure prediction can be memory-intensive due to:
1. Large model sizes with many parameters
2. Complex loss functions with physics-based constraints
3. Batch processing of variable-length sequences
4. Storage of intermediate activations for backpropagation

To address these challenges, we've implemented several memory optimization techniques that allow the model to train on systems with limited memory (as low as 8GB RAM).

## Techniques

### 1. Model Size Reduction

The model architecture automatically adjusts its size based on the available memory:

#### Standard Model (16GB+ RAM)
```python
config = MultiScaleModelConfig(
    nucleotide_features=64,
    motif_features=128,
    global_features=256,
    num_layers_per_scale=3,
    dropout=0.1
)
```

#### Small Model (8-16GB RAM)
```python
config = MultiScaleModelConfig(
    nucleotide_features=32,  # 50% reduction
    motif_features=64,       # 50% reduction
    global_features=128,     # 50% reduction
    num_layers_per_scale=2,  # 33% reduction
    dropout=0.1
)
```

#### Micro Model (<8GB RAM)
```python
config = MultiScaleModelConfig(
    nucleotide_features=16,  # 75% reduction
    motif_features=32,       # 75% reduction
    global_features=64,      # 75% reduction
    num_layers_per_scale=1,  # 66% reduction
    dropout=0.1
)
```

The model size is automatically selected based on the batch size and memory-efficient flag:
- Batch size ≤ 4: Small model
- Batch size = 1 + memory-efficient: Micro model
- Otherwise: Standard model

### 2. Gradient Accumulation

Gradient accumulation allows using smaller batch sizes while maintaining the benefits of larger effective batch sizes:

```python
# Forward pass
predicted_coords, uncertainty = model(sequence_encoding)

# Calculate loss
loss, loss_components = loss_fn(masked_pred_coords, masked_coordinates, batch['sequence'])

# Scale loss for gradient accumulation
loss = loss / gradient_accumulation_steps

# Backward pass
loss.backward()

# Update weights only after accumulating gradients
if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
    # Update weights
    optimizer.step()
    
    # Reset gradients
    optimizer.zero_grad()
```

This technique:
- Reduces memory usage by processing smaller batches
- Maintains the effective batch size by accumulating gradients
- Improves training stability with larger effective batch sizes

### 3. Memory-Efficient Training Mode

The memory-efficient training mode includes several optimizations:

```python
# Free up memory if in memory-efficient mode
if memory_efficient:
    del sequence_encoding, coordinates, mask, predicted_coords, uncertainty
    del masked_pred_coords, masked_coordinates, loss, loss_components
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

This mode:
- Aggressively deletes tensors after they're no longer needed
- Explicitly calls garbage collection
- Clears CUDA cache if using GPU
- Reduces the number of workers for data loading

### 4. Data Subset Training

For systems with very limited memory, we provide options to train on a subset of the data:

```bash
# Create a subset of the data
mkdir -p data/subset
head -n 1 data/raw/train_sequences.csv > data/subset/train_sequences.csv
head -n 101 data/raw/train_sequences.csv | tail -n 100 >> data/subset/train_sequences.csv
```

The micro training script automatically creates and uses a subset of the data:
- 100 training samples
- 20 validation samples
- Minimal batch size (1)
- Gradient accumulation (4 steps)

### 5. Numerical Stability Improvements

Memory issues can be exacerbated by numerical instability. We've implemented several techniques to improve stability:

#### Handling NaN Values
```python
# Check for NaN values
if torch.isnan(predicted_coords).any() or torch.isnan(true_coords).any():
    # Replace NaN values with zeros for calculation
    pred_coords_clean = predicted_coords.clone()
    true_coords_clean = true_coords.clone()
    
    # Replace NaN values with zeros
    pred_coords_clean[torch.isnan(pred_coords_clean)] = 0.0
    true_coords_clean[torch.isnan(true_coords_clean)] = 0.0
```

#### Gradient Clipping
```python
# Add gradient clipping to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Epsilon Values
```python
# Add a small epsilon to avoid numerical instability
epsilon = 1e-8
```

### 6. Error Handling

Robust error handling prevents crashes due to memory issues:

```python
try:
    # Training code
except Exception as e:
    print(f"\nError in batch {batch_idx}: {str(e)}")
    continue
```

This allows the training to continue even if individual batches fail due to memory issues.

## Usage

To use these memory optimization techniques:

### Micro Training (Extremely Limited Memory)
```bash
./run_micro_training.sh
```

### Small Training (Limited Memory)
```bash
./run_training.sh --small
```

### Custom Memory-Efficient Training
```bash
./run_training.sh --batch-size 1 --gradient-accumulation-steps 8 --memory-efficient --num-workers 0
```

## Conclusion

These memory optimization techniques allow the RNA 3D structure prediction model to train on systems with limited memory, from high-end workstations to modest laptops. By automatically adjusting the model size, using gradient accumulation, and implementing memory-efficient training modes, we can make deep learning for RNA 3D structure prediction accessible to a wider range of researchers and hardware configurations.
