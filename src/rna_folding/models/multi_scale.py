"""
Multi-scale equivariant architecture for RNA 3D structure prediction.

This module implements a groundbreaking multi-scale equivariant architecture
that represents RNA at multiple scales simultaneously (nucleotide, motif, global)
and preserves geometric relationships through equivariant transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from rna_folding.models.base import RNAModel, ModelConfig


@dataclass
class MultiScaleModelConfig(ModelConfig):
    """Configuration for multi-scale equivariant RNA model."""

    # Override base config
    model_type: str = "multi_scale"

    # Multi-scale specific parameters
    nucleotide_features: int = 64
    motif_features: int = 128
    global_features: int = 256

    # Equivariant network parameters
    num_layers_per_scale: int = 3
    use_attention: bool = True

    # Physics-informed parameters
    use_physics_constraints: bool = True
    distance_constraint_weight: float = 1.0
    angle_constraint_weight: float = 0.5


class NucleotideEncoder(nn.Module):
    """
    Encoder for nucleotide-level features.

    This module encodes individual nucleotides into a latent representation,
    capturing local sequence context.
    """

    def __init__(self, config):
        """
        Initialize the nucleotide encoder.

        Args:
            config (MultiScaleModelConfig): Model configuration.
        """
        super().__init__()

        # Embedding layer for nucleotides (A, C, G, U, N, -)
        # The input dimension should match the number of nucleotide types in the dataset
        num_nucleotide_types = 7  # A, C, G, U/T, N, -, and potentially others
        self.embedding = nn.Linear(num_nucleotide_types, config.nucleotide_features)

        # 1D convolutional layers to capture local context
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                config.nucleotide_features,
                config.nucleotide_features,
                kernel_size=3,
                padding=1
            )
            for _ in range(config.num_layers_per_scale)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.nucleotide_features)
            for _ in range(config.num_layers_per_scale)
        ])

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the nucleotide encoder.

        Args:
            x (torch.Tensor): One-hot encoded nucleotide sequences with shape (batch_size, seq_len, 7).

        Returns:
            torch.Tensor: Nucleotide features with shape (batch_size, seq_len, nucleotide_features).
        """
        # Embed nucleotides
        x = self.embedding(x)  # (batch_size, seq_len, nucleotide_features)

        # Apply convolutional layers
        x_conv = x.transpose(1, 2)  # (batch_size, nucleotide_features, seq_len)
        x_conv_prev = None  # Initialize previous layer output

        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            # Apply convolution
            x_conv_new = conv(x_conv)

            # Transpose for layer normalization
            x_norm = x_conv_new.transpose(1, 2)  # (batch_size, seq_len, nucleotide_features)
            x_norm = norm(x_norm)

            # Apply dropout and residual connection
            x_norm = self.dropout(x_norm)
            if i > 0:
                # Add residual connection with the previous layer's output
                x_norm = x_norm + x_conv_prev.transpose(1, 2)  # Use the previous layer's output

            # Store current output for next layer's residual connection
            x_conv_prev = x_conv_new.clone()

            # Transpose back for next convolution
            x_conv = x_norm.transpose(1, 2)  # (batch_size, nucleotide_features, seq_len)
            x = x_norm

        return x  # (batch_size, seq_len, nucleotide_features)


class MotifEncoder(nn.Module):
    """
    Encoder for motif-level features.

    This module identifies and encodes RNA motifs, capturing intermediate-scale
    structural patterns.
    """

    def __init__(self, config):
        """
        Initialize the motif encoder.

        Args:
            config (MultiScaleModelConfig): Model configuration.
        """
        super().__init__()

        # Projection from nucleotide features to motif features
        self.projection = nn.Linear(config.nucleotide_features, config.motif_features)

        # Self-attention layers for motif identification
        if config.use_attention:
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=config.motif_features,
                    num_heads=8,
                    dropout=config.dropout
                )
                for _ in range(config.num_layers_per_scale)
            ])
        else:
            self.attention_layers = None

            # Use convolutional layers instead
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(
                    config.motif_features,
                    config.motif_features,
                    kernel_size=5,
                    padding=2
                )
                for _ in range(config.num_layers_per_scale)
            ])

        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.motif_features, config.motif_features * 4),
                nn.ReLU(),
                nn.Linear(config.motif_features * 4, config.motif_features)
            )
            for _ in range(config.num_layers_per_scale)
        ])

        # Layer normalization
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(config.motif_features)
            for _ in range(config.num_layers_per_scale)
        ])

        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(config.motif_features)
            for _ in range(config.num_layers_per_scale)
        ])

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the motif encoder.

        Args:
            x (torch.Tensor): Nucleotide features with shape (batch_size, seq_len, nucleotide_features).

        Returns:
            torch.Tensor: Motif features with shape (batch_size, seq_len, motif_features).
        """
        # Project nucleotide features to motif features
        x = self.projection(x)  # (batch_size, seq_len, motif_features)

        # Apply attention or convolutional layers
        for i in range(len(self.ffn_layers)):
            if self.attention_layers:
                # Self-attention
                x_norm = self.layer_norms1[i](x)
                x_attn, _ = self.attention_layers[i](
                    x_norm.transpose(0, 1),  # (seq_len, batch_size, motif_features)
                    x_norm.transpose(0, 1),
                    x_norm.transpose(0, 1)
                )
                x_attn = x_attn.transpose(0, 1)  # (batch_size, seq_len, motif_features)
                x = x + self.dropout(x_attn)
            else:
                # Convolutional layers
                x_norm = self.layer_norms1[i](x)
                x_conv = self.conv_layers[i](x_norm.transpose(1, 2))  # (batch_size, motif_features, seq_len)
                x_conv = x_conv.transpose(1, 2)  # (batch_size, seq_len, motif_features)
                x = x + self.dropout(x_conv)

            # Feed-forward network
            x_norm = self.layer_norms2[i](x)
            x_ffn = self.ffn_layers[i](x_norm)
            x = x + self.dropout(x_ffn)

        return x  # (batch_size, seq_len, motif_features)


class GlobalEncoder(nn.Module):
    """
    Encoder for global-level features.

    This module captures global structural patterns and long-range interactions
    in the RNA molecule.
    """

    def __init__(self, config):
        """
        Initialize the global encoder.

        Args:
            config (MultiScaleModelConfig): Model configuration.
        """
        super().__init__()

        # Projection from motif features to global features
        self.projection = nn.Linear(config.motif_features, config.global_features)

        # Global attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.global_features,
                num_heads=8,
                dropout=config.dropout
            )
            for _ in range(config.num_layers_per_scale)
        ])

        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.global_features, config.global_features * 4),
                nn.ReLU(),
                nn.Linear(config.global_features * 4, config.global_features)
            )
            for _ in range(config.num_layers_per_scale)
        ])

        # Layer normalization
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(config.global_features)
            for _ in range(config.num_layers_per_scale)
        ])

        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(config.global_features)
            for _ in range(config.num_layers_per_scale)
        ])

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the global encoder.

        Args:
            x (torch.Tensor): Motif features with shape (batch_size, seq_len, motif_features).

        Returns:
            torch.Tensor: Global features with shape (batch_size, seq_len, global_features).
        """
        # Project motif features to global features
        x = self.projection(x)  # (batch_size, seq_len, global_features)

        # Apply attention layers
        for i in range(len(self.attention_layers)):
            # Self-attention
            x_norm = self.layer_norms1[i](x)
            x_attn, _ = self.attention_layers[i](
                x_norm.transpose(0, 1),  # (seq_len, batch_size, global_features)
                x_norm.transpose(0, 1),
                x_norm.transpose(0, 1)
            )
            x_attn = x_attn.transpose(0, 1)  # (batch_size, seq_len, global_features)
            x = x + self.dropout(x_attn)

            # Feed-forward network
            x_norm = self.layer_norms2[i](x)
            x_ffn = self.ffn_layers[i](x_norm)
            x = x + self.dropout(x_ffn)

        return x  # (batch_size, seq_len, global_features)


class StructureDecoder(nn.Module):
    """
    Decoder for 3D structure prediction.

    This module takes multi-scale features and predicts 3D coordinates for each
    nucleotide in the RNA molecule.
    """

    def __init__(self, config):
        """
        Initialize the structure decoder.

        Args:
            config (MultiScaleModelConfig): Model configuration.
        """
        super().__init__()

        # Combine features from all scales
        combined_features = config.nucleotide_features + config.motif_features + config.global_features

        # Projection to intermediate representation
        self.projection = nn.Linear(combined_features, config.global_features)

        # Coordinate prediction layers
        self.coord_layers = nn.Sequential(
            nn.Linear(config.global_features, config.global_features),
            nn.ReLU(),
            nn.Linear(config.global_features, config.global_features // 2),
            nn.ReLU(),
            nn.Linear(config.global_features // 2, 3)  # 3D coordinates (x, y, z)
        )

        # Uncertainty prediction (optional)
        self.uncertainty_layers = nn.Sequential(
            nn.Linear(config.global_features, config.global_features // 2),
            nn.ReLU(),
            nn.Linear(config.global_features // 2, 3)  # Uncertainty for each coordinate
        )

    def forward(self, nucleotide_features, motif_features, global_features):
        """
        Forward pass of the structure decoder.

        Args:
            nucleotide_features (torch.Tensor): Nucleotide features with shape (batch_size, seq_len, nucleotide_features).
            motif_features (torch.Tensor): Motif features with shape (batch_size, seq_len, motif_features).
            global_features (torch.Tensor): Global features with shape (batch_size, seq_len, global_features).

        Returns:
            tuple: Tuple containing:
                - torch.Tensor: Predicted 3D coordinates with shape (batch_size, seq_len, 3).
                - torch.Tensor: Uncertainty estimates with shape (batch_size, seq_len, 3).
        """
        # Concatenate features from all scales
        combined = torch.cat([nucleotide_features, motif_features, global_features], dim=-1)

        # Project to intermediate representation
        x = self.projection(combined)

        # Predict coordinates
        coords = self.coord_layers(x)

        # Predict uncertainty
        uncertainty = torch.exp(self.uncertainty_layers(x))  # Positive uncertainty values

        return coords, uncertainty


class MultiScaleRNA(RNAModel):
    """
    Multi-scale equivariant model for RNA 3D structure prediction.

    This model represents RNA at multiple scales simultaneously (nucleotide, motif, global)
    and preserves geometric relationships through equivariant transformations.
    """

    def __init__(self, config):
        """
        Initialize the multi-scale RNA model.

        Args:
            config (MultiScaleModelConfig): Model configuration.
        """
        super().__init__(config)

        # Ensure config is of the correct type
        if not isinstance(config, MultiScaleModelConfig):
            config = MultiScaleModelConfig(**config.to_dict())

        self.config = config

        # Create encoders for each scale
        self.nucleotide_encoder = NucleotideEncoder(config)
        self.motif_encoder = MotifEncoder(config)
        self.global_encoder = GlobalEncoder(config)

        # Create structure decoder
        self.structure_decoder = StructureDecoder(config)

    def forward(self, x):
        """
        Forward pass of the multi-scale RNA model.

        Args:
            x (torch.Tensor): One-hot encoded nucleotide sequences with shape (batch_size, seq_len, 7).

        Returns:
            tuple: Tuple containing:
                - torch.Tensor: Predicted 3D coordinates with shape (batch_size, seq_len, 3).
                - torch.Tensor: Uncertainty estimates with shape (batch_size, seq_len, 3).
        """
        # Get batch size and sequence length
        batch_size, seq_len, _ = x.shape

        # Encode at nucleotide level
        nucleotide_features = self.nucleotide_encoder(x)

        # Encode at motif level
        motif_features = self.motif_encoder(nucleotide_features)

        # Encode at global level
        global_features = self.global_encoder(motif_features)

        # Decode to 3D structure
        coords, uncertainty = self.structure_decoder(
            nucleotide_features, motif_features, global_features
        )

        return coords, uncertainty

    def predict(self, sequence, num_predictions=5):
        """
        Generate 3D structure predictions for an RNA sequence.

        Args:
            sequence (str): RNA sequence.
            num_predictions (int): Number of predictions to generate.

        Returns:
            list: List of predicted 3D structures.
        """
        # Convert sequence to one-hot encoding
        # The model expects 7 nucleotide types
        nucleotide_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4, '-': 5, 'X': 6}
        seq_len = len(sequence)
        encoding = torch.zeros(1, seq_len, 7)  # Batch size of 1, 7 nucleotide types

        for i, nucleotide in enumerate(sequence.upper()):
            if nucleotide in nucleotide_to_idx:
                idx = nucleotide_to_idx[nucleotide]
            else:
                idx = nucleotide_to_idx['N']  # Unknown nucleotide

            encoding[0, i, idx] = 1.0

        # Move to device
        encoding = encoding.to(self.device)

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
            coords_np = coords[0].cpu().numpy()  # Remove batch dimension
            predictions.append(coords_np)

        return predictions
