# Multi-Scale Equivariant RNA 3D Structure Prediction Model

This document provides detailed information about the multi-scale equivariant architecture for RNA 3D structure prediction with physics-informed neural networks.

## Architecture Overview

The model architecture consists of several key components:

1. **Multi-Scale Representation**: Processes RNA at three different scales simultaneously
2. **Physics-Informed Constraints**: Enforces physical constraints on predicted structures
3. **Uncertainty-Aware Prediction**: Generates multiple plausible structures with confidence estimates

## Detailed Components

### 1. Base Model Architecture

The base model architecture provides the foundation for all RNA 3D structure prediction models:

- `ModelConfig`: Configuration class that stores model parameters
- `RNAModel`: Base class that implements common functionality like device management and saving/loading

```python
class RNAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")
    
    def predict(self, sequence, num_predictions=5):
        raise NotImplementedError("Subclasses must implement predict method")
```

### 2. Multi-Scale Architecture

#### 2.1 Nucleotide-Level Encoder

The nucleotide-level encoder captures local sequence context and base-specific features:

- Input: One-hot encoded RNA sequence
- Processing: 1D convolutional layers with residual connections
- Output: Nucleotide-level features

```python
class NucleotideEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Linear(num_nucleotide_types, config.nucleotide_features)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(config.nucleotide_features, config.nucleotide_features, kernel_size=3, padding=1)
            for _ in range(config.num_layers_per_scale)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.nucleotide_features)
            for _ in range(config.num_layers_per_scale)
        ])
        self.dropout = nn.Dropout(config.dropout)
```

#### 2.2 Motif-Level Encoder

The motif-level encoder identifies and encodes RNA motifs and intermediate-scale structural patterns:

- Input: Nucleotide-level features
- Processing: Self-attention or convolutional layers
- Output: Motif-level features

```python
class MotifEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(config.nucleotide_features, config.motif_features)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=config.motif_features, num_heads=8, dropout=config.dropout)
            for _ in range(config.num_layers_per_scale)
        ])
        # Additional layers for feed-forward networks and normalization
```

#### 2.3 Global-Level Encoder

The global-level encoder captures long-range interactions and global structural patterns:

- Input: Motif-level features
- Processing: Global attention mechanisms
- Output: Global-level features

```python
class GlobalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(config.motif_features, config.global_features)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=config.global_features, num_heads=8, dropout=config.dropout)
            for _ in range(config.num_layers_per_scale)
        ])
        # Additional layers for feed-forward networks and normalization
```

#### 2.4 Structure Decoder

The structure decoder predicts 3D coordinates from multi-scale features:

- Input: Concatenated features from all scales
- Processing: Multi-layer perceptron
- Output: 3D coordinates and uncertainty estimates

```python
class StructureDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        combined_features = config.nucleotide_features + config.motif_features + config.global_features
        self.projection = nn.Linear(combined_features, config.global_features)
        self.coord_layers = nn.Sequential(
            nn.Linear(config.global_features, config.global_features),
            nn.ReLU(),
            nn.Linear(config.global_features, config.global_features // 2),
            nn.ReLU(),
            nn.Linear(config.global_features // 2, 3)  # 3D coordinates (x, y, z)
        )
        self.uncertainty_layers = nn.Sequential(
            nn.Linear(config.global_features, config.global_features // 2),
            nn.ReLU(),
            nn.Linear(config.global_features // 2, 3)  # Uncertainty for each coordinate
        )
```

### 3. Physics-Informed Constraints

The physics-informed constraints ensure that predicted structures are biologically plausible:

#### 3.1 Bond Length Constraints

Enforces realistic distances between consecutive nucleotides:

```python
def bond_length_energy(self, coords):
    # Calculate distances between consecutive C1' atoms
    consecutive_coords = coords[:, :-1, :]
    next_coords = coords[:, 1:, :]
    
    # Calculate squared distances for numerical stability
    squared_distances = torch.sum((consecutive_coords - next_coords) ** 2, dim=-1)
    
    # Add small epsilon to avoid sqrt of zero
    epsilon = 1e-8
    distances = torch.sqrt(squared_distances + epsilon)
    
    # Calculate energy term (harmonic potential)
    energy = torch.mean((distances - self.c1_c1_mean_distance) ** 2)
    
    return energy
```

#### 3.2 Steric Clash Prevention

Prevents atoms from occupying the same space:

```python
def steric_clash_energy(self, coords):
    # Calculate all pairwise distances
    coords_expanded1 = coords.unsqueeze(2)
    coords_expanded2 = coords.unsqueeze(1)
    
    # Calculate squared distances
    squared_distances = torch.sum((coords_expanded1 - coords_expanded2) ** 2, dim=-1)
    
    # Create mask to exclude consecutive nucleotides
    mask = torch.ones_like(squared_distances, dtype=torch.bool)
    for i in range(seq_len):
        for j in range(max(0, i-1), min(seq_len, i+2)):
            mask[:, i, j] = False
    
    # Calculate energy term for steric clashes
    min_distance_tensor = torch.tensor(self.c1_c1_min_distance, device=coords.device)
    steric_energy = torch.sum(
        torch.relu(min_distance_tensor - masked_distances) ** 2,
        dim=(1, 2)
    )
    
    return torch.mean(steric_energy)
```

#### 3.3 Base-Pairing Constraints

Encourages proper base pairing (A-U, G-C, G-U):

```python
def base_pairing_energy(self, coords, sequence):
    # Process each sequence in the batch
    for b in range(batch_size):
        seq = sequence[b]
        actual_seq_len = min(len(seq), seq_len)
        
        # Find potential base pairs
        for i in range(actual_seq_len - 3):
            for j in range(i + 3, actual_seq_len):
                base_i = seq[i]
                base_j = seq[j]
                
                # Check if bases can form a pair
                if (base_i, base_j) in self.base_pair_distances:
                    # Get target distance
                    target_distance = self.base_pair_distances[(base_i, base_j)]
                    
                    # Calculate actual distance
                    actual_distance = torch.sqrt(torch.sum((coords[b, i, :] - coords[b, j, :]) ** 2))
                    
                    # Add to energy term
                    energy = energy + (actual_distance - target_distance) ** 2
```

### 4. Physics-Informed Loss Function

The physics-informed loss function combines standard prediction loss with physics-based energy terms:

```python
class PhysicsInformedLoss(nn.Module):
    def __init__(self, prediction_weight=1.0, physics_weight=0.01):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.physics_weight = physics_weight
        self.physics_constraints = PhysicsConstraints()
    
    def forward(self, predicted_coords, true_coords, sequence=None):
        # Calculate prediction loss (MSE)
        prediction_loss = F.mse_loss(predicted_coords, true_coords)
        
        # Calculate physics-based energy terms
        energy_terms = self.physics_constraints(predicted_coords, sequence)
        
        # Calculate total loss
        total_loss = (
            self.prediction_weight * prediction_loss +
            self.physics_weight * energy_terms['total_energy']
        )
        
        return total_loss, loss_components
```

## Training Process

The training process includes several key features:

1. **Gradient Clipping**: Prevents exploding gradients
2. **Early Stopping**: Stops training when validation loss stops improving
3. **Learning Rate Scheduling**: Reduces learning rate when validation loss plateaus
4. **Numerical Stability**: Handles NaN values and adds epsilon to avoid division by zero

```python
def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler=None, num_epochs=100, device='cuda', checkpoint_dir=None, early_stopping_patience=10):
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        
        for batch in train_loader:
            # Forward pass
            predicted_coords, uncertainty = model(sequence_encoding)
            
            # Calculate loss
            loss, loss_components = loss_fn(masked_pred_coords, masked_coordinates, batch['sequence'])
            
            # Backward pass and optimize
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Validation phase
        model.eval()
        
        # Early stopping
        if epoch - best_epoch >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
```

## Inference Process

The inference process generates multiple plausible structures with uncertainty estimates:

```python
def predict(self, sequence, num_predictions=5):
    # Convert sequence to one-hot encoding
    encoding = torch.zeros(1, seq_len, 6)
    
    for i, nucleotide in enumerate(sequence.upper()):
        if nucleotide in nucleotide_to_idx:
            idx = nucleotide_to_idx[nucleotide]
        else:
            idx = nucleotide_to_idx['N']
        
        encoding[0, i, idx] = 1.0
    
    # Generate predictions
    predictions = []
    
    for _ in range(num_predictions):
        # Forward pass
        with torch.no_grad():
            coords, uncertainty = self.forward(encoding)
        
        # Add noise based on uncertainty for diverse predictions
        if _ > 0:  # First prediction is deterministic
            noise = torch.randn_like(coords) * uncertainty
            coords = coords + noise
        
        # Convert to numpy and add to predictions
        coords_np = coords[0].cpu().numpy()
        predictions.append(coords_np)
    
    return predictions
```

## Model Variants

The architecture is designed to be modular, allowing for different variants:

### 1. Basic Multi-Scale

The default implementation with nucleotide, motif, and global encoders:

```python
config = MultiScaleModelConfig(
    nucleotide_features=64,
    motif_features=128,
    global_features=256,
    num_layers_per_scale=3,
    dropout=0.1
)
model = MultiScaleRNA(config)
```

### 2. Physics-Enhanced

Increases the weight of physics constraints for more realistic structures:

```python
config = MultiScaleModelConfig(
    nucleotide_features=64,
    motif_features=128,
    global_features=256,
    num_layers_per_scale=3,
    dropout=0.1
)
model = MultiScaleRNA(config)
loss_fn = PhysicsInformedLoss(prediction_weight=1.0, physics_weight=0.1)  # Higher physics weight
```

### 3. Uncertainty-Aware

Generates multiple predictions with uncertainty estimates:

```python
config = MultiScaleModelConfig(
    nucleotide_features=64,
    motif_features=128,
    global_features=256,
    num_layers_per_scale=3,
    dropout=0.1
)
model = MultiScaleRNA(config)
predictions = model.predict(sequence, num_predictions=10)  # Generate 10 predictions
```

## Performance Optimization

The implementation includes several optimizations for performance:

1. **Memory Efficiency**: Optimized physics calculations to be more memory-efficient
2. **Numerical Stability**: Added checks for NaN values and epsilon to avoid division by zero
3. **Gradient Clipping**: Prevents exploding gradients
4. **Device Flexibility**: Automatically uses GPU if available, falls back to CPU if not

## References

1. Template Modeling Score: Zhang & Skolnick, 2004
2. Equivariant Neural Networks: Cohen & Welling, 2016
3. Physics-Informed Neural Networks: Raissi et al., 2019
4. RNA 3D Structure Prediction: Das & Baker, 2007
