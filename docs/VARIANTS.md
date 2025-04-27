# Creating New Model Variants

This document provides guidance on how to create new variants of the multi-scale equivariant RNA 3D structure prediction model.

## Overview

The model architecture is designed to be modular, allowing for easy creation of new variants. You can modify various aspects of the model, including:

1. **Architecture Components**: Change the encoders, decoder, or add new components
2. **Model Configuration**: Adjust hyperparameters like feature dimensions and number of layers
3. **Physics Constraints**: Modify the physics-based constraints or their weights
4. **Training Process**: Change the loss function, optimizer, or training schedule

## Step-by-Step Guide

### 1. Creating a New Model Configuration

To create a new model configuration, subclass `MultiScaleModelConfig` and override the default parameters:

```python
from rna_folding.models.multi_scale import MultiScaleModelConfig

class EnhancedModelConfig(MultiScaleModelConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Override default parameters
        self.nucleotide_features = 128  # Increased from default 64
        self.motif_features = 256       # Increased from default 128
        self.global_features = 512      # Increased from default 256
        self.num_layers_per_scale = 4   # Increased from default 3
        self.dropout = 0.2              # Increased from default 0.1
        
        # Add new parameters
        self.use_gated_attention = True
        self.use_layer_scale = True
```

### 2. Creating a New Encoder

To create a new encoder, subclass one of the existing encoders and override the `forward` method:

```python
from rna_folding.models.multi_scale import MotifEncoder
import torch.nn as nn

class GatedMotifEncoder(MotifEncoder):
    def __init__(self, config):
        super().__init__(config)
        
        # Add gated attention components
        self.gate_layers = nn.ModuleList([
            nn.Linear(config.motif_features, config.motif_features)
            for _ in range(config.num_layers_per_scale)
        ])
    
    def forward(self, x):
        # Project nucleotide features to motif features
        x = self.projection(x)
        
        # Apply attention or convolutional layers with gating
        for i in range(len(self.ffn_layers)):
            if self.attention_layers:
                # Self-attention
                x_norm = self.layer_norms1[i](x)
                x_attn, _ = self.attention_layers[i](
                    x_norm.transpose(0, 1),
                    x_norm.transpose(0, 1),
                    x_norm.transpose(0, 1)
                )
                x_attn = x_attn.transpose(0, 1)
                
                # Apply gating
                gate = torch.sigmoid(self.gate_layers[i](x_attn))
                x_attn = x_attn * gate
                
                x = x + self.dropout(x_attn)
            
            # Feed-forward network
            x_norm = self.layer_norms2[i](x)
            x_ffn = self.ffn_layers[i](x_norm)
            x = x + self.dropout(x_ffn)
        
        return x
```

### 3. Creating a New Model Class

To create a new model class, subclass `MultiScaleRNA` and override the `__init__` method to use your custom components:

```python
from rna_folding.models.multi_scale import MultiScaleRNA, NucleotideEncoder, GlobalEncoder, StructureDecoder

class EnhancedMultiScaleRNA(MultiScaleRNA):
    def __init__(self, config):
        super().__init__(config)
        
        # Replace the motif encoder with the gated version
        self.motif_encoder = GatedMotifEncoder(config)
        
        # You can also replace other components if needed
        # self.nucleotide_encoder = CustomNucleotideEncoder(config)
        # self.global_encoder = CustomGlobalEncoder(config)
        # self.structure_decoder = CustomStructureDecoder(config)
```

### 4. Creating a Custom Physics Constraint

To create a custom physics constraint, subclass `PhysicsConstraints` and add new energy terms:

```python
from rna_folding.models.physics import PhysicsConstraints
import torch

class EnhancedPhysicsConstraints(PhysicsConstraints):
    def __init__(self):
        super().__init__()
        
        # Add new physical constants
        self.planarity_weight = 0.5
    
    def planarity_energy(self, coords):
        """Calculate energy term for base planarity constraints."""
        batch_size, seq_len, _ = coords.shape
        
        # Skip if sequence is too short
        if seq_len < 4:
            return torch.tensor(0.0, device=coords.device)
        
        # Calculate planarity for each set of 4 consecutive nucleotides
        planarity_energy = torch.tensor(0.0, device=coords.device)
        
        for i in range(seq_len - 3):
            # Get coordinates of 4 consecutive nucleotides
            points = coords[:, i:i+4, :]
            
            # Calculate normal vector of the plane
            v1 = points[:, 1, :] - points[:, 0, :]
            v2 = points[:, 2, :] - points[:, 0, :]
            normal = torch.cross(v1, v2, dim=-1)
            normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-8)
            
            # Calculate distance of the fourth point from the plane
            v3 = points[:, 3, :] - points[:, 0, :]
            distance = torch.abs(torch.sum(v3 * normal, dim=-1))
            
            # Add to energy term
            planarity_energy = planarity_energy + torch.mean(distance ** 2)
        
        return planarity_energy / (seq_len - 3)
    
    def forward(self, coords, sequence=None):
        # Get energy terms from parent class
        energy_terms = super().forward(coords, sequence)
        
        # Calculate planarity energy
        planarity_energy = self.planarity_energy(coords)
        
        # Add to total energy
        total_energy = energy_terms['total_energy'] + self.planarity_weight * planarity_energy
        
        # Update energy terms dictionary
        energy_terms['planarity_energy'] = planarity_energy
        energy_terms['total_energy'] = total_energy
        
        return energy_terms
```

### 5. Creating a Custom Loss Function

To create a custom loss function, subclass `PhysicsInformedLoss` and override the `forward` method:

```python
from rna_folding.models.physics import PhysicsInformedLoss
import torch.nn.functional as F

class EnhancedLoss(PhysicsInformedLoss):
    def __init__(self, prediction_weight=1.0, physics_weight=0.01, consistency_weight=0.5):
        super().__init__(prediction_weight, physics_weight)
        
        # Replace the physics constraints with the enhanced version
        self.physics_constraints = EnhancedPhysicsConstraints()
        
        # Add new weight parameter
        self.consistency_weight = consistency_weight
    
    def consistency_loss(self, predicted_coords, uncertainty):
        """Calculate consistency loss based on uncertainty estimates."""
        # Higher uncertainty should correspond to higher error
        # This encourages the model to be well-calibrated
        return torch.mean(uncertainty)
    
    def forward(self, predicted_coords, true_coords, sequence=None, uncertainty=None):
        # Calculate prediction loss and physics loss from parent class
        total_loss, loss_components = super().forward(predicted_coords, true_coords, sequence)
        
        # Calculate consistency loss if uncertainty is provided
        if uncertainty is not None:
            consistency_loss = self.consistency_loss(predicted_coords, uncertainty)
            total_loss = total_loss + self.consistency_weight * consistency_loss
            loss_components['consistency_loss'] = consistency_loss
            loss_components['total_loss'] = total_loss
        
        return total_loss, loss_components
```

### 6. Training a Custom Model

To train a custom model, create an instance of your model and loss function, and pass them to the `train_model` function:

```python
from rna_folding.models.train import train_model
import torch.optim as optim

# Create custom configuration
config = EnhancedModelConfig(
    nucleotide_features=128,
    motif_features=256,
    global_features=512,
    num_layers_per_scale=4,
    dropout=0.2,
    learning_rate=1e-4,
    weight_decay=1e-5,
    batch_size=4
)

# Create custom model
model = EnhancedMultiScaleRNA(config)

# Create custom loss function
loss_fn = EnhancedLoss(prediction_weight=1.0, physics_weight=0.01, consistency_weight=0.5)

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

# Train model
history = train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    num_epochs=50,
    device='cuda',
    checkpoint_dir='models/enhanced',
    early_stopping_patience=10
)
```

## Example Variants

### 1. High-Resolution Model

A model with increased feature dimensions for higher resolution predictions:

```python
config = MultiScaleModelConfig(
    nucleotide_features=128,
    motif_features=256,
    global_features=512,
    num_layers_per_scale=4,
    dropout=0.1
)
model = MultiScaleRNA(config)
```

### 2. Physics-Dominated Model

A model that prioritizes physics constraints over prediction accuracy:

```python
config = MultiScaleModelConfig(
    nucleotide_features=64,
    motif_features=128,
    global_features=256,
    num_layers_per_scale=3,
    dropout=0.1
)
model = MultiScaleRNA(config)
loss_fn = PhysicsInformedLoss(prediction_weight=0.5, physics_weight=0.5)  # Equal weights
```

### 3. Ensemble Model

A model that combines predictions from multiple models:

```python
class EnsembleRNA:
    def __init__(self, models):
        self.models = models
    
    def predict(self, sequence, num_predictions=5):
        all_predictions = []
        
        for model in self.models:
            predictions = model.predict(sequence, num_predictions=1)
            all_predictions.extend(predictions)
        
        return all_predictions[:num_predictions]

# Create ensemble
models = [
    MultiScaleRNA(MultiScaleModelConfig(nucleotide_features=64)),
    MultiScaleRNA(MultiScaleModelConfig(nucleotide_features=128)),
    MultiScaleRNA(MultiScaleModelConfig(motif_features=256))
]
ensemble = EnsembleRNA(models)
```

## Tips for Creating Effective Variants

1. **Start Small**: Begin with small changes to the base model to understand their impact
2. **Monitor Performance**: Track metrics like TM-score and RMSD to evaluate your variants
3. **Balance Complexity**: More complex models may overfit on limited data
4. **Consider Computational Resources**: Larger models require more memory and computation
5. **Combine Approaches**: The best variants often combine multiple improvements
6. **Validate Thoroughly**: Test your variants on multiple RNA sequences to ensure robustness

## Conclusion

The modular design of the multi-scale equivariant architecture allows for easy creation of new model variants. By modifying the architecture components, model configuration, physics constraints, or training process, you can create variants tailored to specific RNA families or prediction tasks.
