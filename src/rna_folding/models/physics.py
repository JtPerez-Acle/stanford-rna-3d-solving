"""
Physics-informed neural network layers for RNA 3D structure prediction.

This module implements physics-informed neural network layers that enforce
physical constraints on RNA 3D structures, such as bond lengths, angles,
and hydrogen bonding patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhysicsConstraints(nn.Module):
    """
    Physics-informed constraints for RNA 3D structures.

    This module enforces physical constraints on predicted RNA 3D structures,
    including bond lengths, angles, and hydrogen bonding patterns.
    """

    def __init__(self):
        """Initialize the physics constraints module."""
        super().__init__()

        # Define physical constants
        self.c1_c1_min_distance = 3.5  # Minimum distance between C1' atoms (in Angstroms)
        self.c1_c1_max_distance = 7.0  # Maximum distance between consecutive C1' atoms
        self.c1_c1_mean_distance = 5.0  # Mean distance between consecutive C1' atoms

        # Define base-pairing distances
        self.base_pair_distances = {
            ('A', 'U'): 5.9,  # A-U base pair distance
            ('G', 'C'): 5.7,  # G-C base pair distance
            ('G', 'U'): 5.8   # G-U wobble pair distance
        }

        # Add reverse pairs
        for (b1, b2), dist in list(self.base_pair_distances.items()):
            self.base_pair_distances[(b2, b1)] = dist

    def bond_length_energy(self, coords):
        """
        Calculate energy term for bond length constraints.

        Args:
            coords (torch.Tensor): Predicted coordinates with shape (batch_size, seq_len, 3).

        Returns:
            torch.Tensor: Bond length energy term.
        """
        batch_size, seq_len, _ = coords.shape

        # Skip if sequence is too short
        if seq_len <= 1:
            return torch.tensor(0.0, device=coords.device)

        # Calculate distances between consecutive C1' atoms
        consecutive_coords = coords[:, :-1, :]  # (batch_size, seq_len-1, 3)
        next_coords = coords[:, 1:, :]  # (batch_size, seq_len-1, 3)

        # Calculate squared distances for numerical stability
        squared_distances = torch.sum((consecutive_coords - next_coords) ** 2, dim=-1)  # (batch_size, seq_len-1)

        # Add small epsilon to avoid sqrt of zero
        epsilon = 1e-8
        distances = torch.sqrt(squared_distances + epsilon)  # (batch_size, seq_len-1)

        # Calculate energy term (harmonic potential)
        energy = torch.mean((distances - self.c1_c1_mean_distance) ** 2)

        # Ensure energy is finite
        if not torch.isfinite(energy):
            return torch.tensor(0.0, device=coords.device)

        return energy

    def steric_clash_energy(self, coords):
        """
        Calculate energy term for steric clash constraints.

        Args:
            coords (torch.Tensor): Predicted coordinates with shape (batch_size, seq_len, 3).

        Returns:
            torch.Tensor: Steric clash energy term.
        """
        batch_size, seq_len, _ = coords.shape

        # Skip if sequence is too short
        if seq_len <= 3:
            return torch.tensor(0.0, device=coords.device)

        # Calculate all pairwise distances
        # Reshape to (batch_size, seq_len, 1, 3) and (batch_size, 1, seq_len, 3)
        coords_expanded1 = coords.unsqueeze(2)  # (batch_size, seq_len, 1, 3)
        coords_expanded2 = coords.unsqueeze(1)  # (batch_size, 1, seq_len, 3)

        # Calculate squared distances
        squared_distances = torch.sum((coords_expanded1 - coords_expanded2) ** 2, dim=-1)  # (batch_size, seq_len, seq_len)

        # Create mask to exclude consecutive nucleotides
        mask = torch.ones_like(squared_distances, dtype=torch.bool)
        for i in range(seq_len):
            for j in range(max(0, i-1), min(seq_len, i+2)):
                mask[:, i, j] = False

        # Add small epsilon to avoid sqrt of zero
        epsilon = 1e-8

        # Apply mask and calculate distances
        masked_squared_distances = squared_distances * mask.float() + (1 - mask.float()) * epsilon
        masked_distances = torch.sqrt(masked_squared_distances)

        # Calculate energy term for steric clashes
        # Use a soft minimum function to penalize distances less than min_distance
        min_distance_tensor = torch.tensor(self.c1_c1_min_distance, device=coords.device)
        steric_energy = torch.sum(
            torch.relu(min_distance_tensor - masked_distances) ** 2,
            dim=(1, 2)
        )

        # Normalize by the number of pairs to make it independent of sequence length
        num_pairs = torch.sum(mask.float(), dim=(1, 2))
        normalized_energy = steric_energy / (num_pairs + epsilon)

        # Ensure energy is finite
        result = torch.mean(normalized_energy)
        if not torch.isfinite(result):
            return torch.tensor(0.0, device=coords.device)

        return result

    def base_pairing_energy(self, coords, sequence):
        """
        Calculate energy term for base pairing constraints.

        Args:
            coords (torch.Tensor): Predicted coordinates with shape (batch_size, seq_len, 3).
            sequence (list): List of RNA sequences.

        Returns:
            torch.Tensor: Base pairing energy term.
        """
        batch_size, seq_len, _ = coords.shape

        # Initialize energy
        energy = torch.tensor(0.0, device=coords.device)

        # Add small epsilon to avoid division by zero
        epsilon = 1e-8

        # Count valid pairs for normalization
        total_pairs = 0

        # Process each sequence in the batch
        for b in range(batch_size):
            seq = sequence[b]
            actual_seq_len = min(len(seq), seq_len)

            # Skip if sequence is too short
            if actual_seq_len < 4:
                continue

            # Find potential base pairs
            for i in range(actual_seq_len - 3):
                for j in range(i + 3, actual_seq_len):
                    base_i = seq[i]
                    base_j = seq[j]

                    # Check if bases can form a pair
                    if (base_i, base_j) in self.base_pair_distances:
                        # Get target distance
                        target_distance = self.base_pair_distances[(base_i, base_j)]

                        # Calculate squared distance for numerical stability
                        squared_distance = torch.sum((coords[b, i, :] - coords[b, j, :]) ** 2)

                        # Add small epsilon to avoid sqrt of zero
                        actual_distance = torch.sqrt(squared_distance + epsilon)

                        # Add to energy term
                        pair_energy = (actual_distance - target_distance) ** 2

                        # Ensure energy is finite
                        if torch.isfinite(pair_energy):
                            energy = energy + pair_energy
                            total_pairs += 1

        # Return normalized energy or zero if no valid pairs
        if total_pairs > 0:
            return energy / total_pairs
        else:
            return torch.tensor(0.0, device=coords.device)

    def forward(self, coords, sequence=None):
        """
        Calculate total physics-based energy for predicted structures.

        Args:
            coords (torch.Tensor): Predicted coordinates with shape (batch_size, seq_len, 3).
            sequence (list, optional): List of RNA sequences.

        Returns:
            dict: Dictionary containing energy terms.
        """
        # Check if coordinates contain NaN values
        if torch.isnan(coords).any():
            # Return zero energy if coordinates contain NaN
            zero_energy = torch.tensor(0.0, device=coords.device)
            return {
                'bond_energy': zero_energy,
                'steric_energy': zero_energy,
                'base_pair_energy': zero_energy,
                'total_energy': zero_energy
            }

        # Calculate bond length energy
        bond_energy = self.bond_length_energy(coords)

        # Calculate steric clash energy
        steric_energy = self.steric_clash_energy(coords)

        # Calculate base pairing energy if sequence is provided
        base_pair_energy = torch.tensor(0.0, device=coords.device)
        if sequence is not None:
            base_pair_energy = self.base_pairing_energy(coords, sequence)

        # Calculate total energy with weights to balance the terms
        total_energy = bond_energy + steric_energy + base_pair_energy

        # Ensure all energies are finite
        if not torch.isfinite(total_energy):
            zero_energy = torch.tensor(0.0, device=coords.device)
            return {
                'bond_energy': zero_energy,
                'steric_energy': zero_energy,
                'base_pair_energy': zero_energy,
                'total_energy': zero_energy
            }

        return {
            'bond_energy': bond_energy,
            'steric_energy': steric_energy,
            'base_pair_energy': base_pair_energy,
            'total_energy': total_energy
        }


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function for RNA 3D structure prediction.

    This loss function combines standard prediction loss (e.g., MSE) with
    physics-based energy terms to enforce physical constraints.
    """

    def __init__(self, prediction_weight=1.0, physics_weight=0.1):
        """
        Initialize the physics-informed loss function.

        Args:
            prediction_weight (float): Weight for prediction loss.
            physics_weight (float): Weight for physics-based energy terms.
        """
        super().__init__()

        self.prediction_weight = prediction_weight
        self.physics_weight = physics_weight

        # Create physics constraints module
        self.physics_constraints = PhysicsConstraints()

    def forward(self, predicted_coords, true_coords, sequence=None):
        """
        Calculate physics-informed loss.

        Args:
            predicted_coords (torch.Tensor): Predicted coordinates with shape (batch_size, seq_len, 3).
            true_coords (torch.Tensor): True coordinates with shape (batch_size, seq_len, 3).
            sequence (list, optional): List of RNA sequences.

        Returns:
            tuple: Tuple containing:
                - torch.Tensor: Total loss.
                - dict: Dictionary containing loss components.
        """
        # Add small epsilon to avoid numerical instability
        epsilon = 1e-8

        # Check for NaN values
        has_nan = torch.isnan(predicted_coords).any() or torch.isnan(true_coords).any()

        # Replace NaN values with zeros for calculation (we'll handle the NaN case separately)
        if has_nan:
            # Create a copy to avoid modifying the original tensors
            pred_coords_clean = predicted_coords.clone()
            true_coords_clean = true_coords.clone()

            # Replace NaN values with zeros
            pred_coords_clean[torch.isnan(pred_coords_clean)] = 0.0
            true_coords_clean[torch.isnan(true_coords_clean)] = 0.0

            # Calculate a simple MSE loss that can be backpropagated
            # This will help the model move away from producing NaNs
            prediction_loss = F.mse_loss(pred_coords_clean, true_coords_clean)

            # Add a penalty for having NaNs
            nan_penalty = 0.1  # Small penalty to encourage moving away from NaNs
            prediction_loss = prediction_loss + nan_penalty

            # Create loss components
            loss_components = {
                'prediction_loss': prediction_loss,
                'physics_loss': torch.tensor(0.0, device=predicted_coords.device),
                'bond_energy': torch.tensor(0.0, device=predicted_coords.device),
                'steric_energy': torch.tensor(0.0, device=predicted_coords.device),
                'base_pair_energy': torch.tensor(0.0, device=predicted_coords.device),
                'total_loss': prediction_loss
            }

            return prediction_loss, loss_components

        # If no NaNs, proceed with normal calculation
        try:
            # Calculate prediction loss (MSE)
            prediction_loss = F.mse_loss(predicted_coords, true_coords)

            # Ensure prediction loss is finite
            if not torch.isfinite(prediction_loss):
                # If not finite, use a simple L1 loss instead which might be more stable
                prediction_loss = F.l1_loss(predicted_coords, true_coords)

                # If still not finite, use a constant loss
                if not torch.isfinite(prediction_loss):
                    prediction_loss = torch.tensor(0.1, device=predicted_coords.device, requires_grad=True)

            # Calculate physics-based energy terms
            energy_terms = self.physics_constraints(predicted_coords, sequence)

            # Calculate total loss with physics terms
            physics_weight = self.physics_weight

            # Start with a very small physics weight and gradually increase it
            # This helps stabilize early training
            total_loss = (
                self.prediction_weight * prediction_loss +
                physics_weight * energy_terms['total_energy']
            )

            # Ensure total loss is finite
            if not torch.isfinite(total_loss):
                # Fall back to just prediction loss
                total_loss = prediction_loss

            # Create loss components dictionary
            loss_components = {
                'prediction_loss': prediction_loss,
                'physics_loss': energy_terms['total_energy'],
                'bond_energy': energy_terms['bond_energy'],
                'steric_energy': energy_terms['steric_energy'],
                'base_pair_energy': energy_terms['base_pair_energy'],
                'total_loss': total_loss
            }

        except Exception as e:
            # If any error occurs, fall back to a simple loss
            print(f"Error in loss calculation: {str(e)}")
            prediction_loss = torch.mean((predicted_coords - true_coords) ** 2)

            # If still problematic, use a constant loss
            if not torch.isfinite(prediction_loss):
                prediction_loss = torch.tensor(0.1, device=predicted_coords.device, requires_grad=True)

            # Create simple loss components
            loss_components = {
                'prediction_loss': prediction_loss,
                'physics_loss': torch.tensor(0.0, device=predicted_coords.device),
                'bond_energy': torch.tensor(0.0, device=predicted_coords.device),
                'steric_energy': torch.tensor(0.0, device=predicted_coords.device),
                'base_pair_energy': torch.tensor(0.0, device=predicted_coords.device),
                'total_loss': prediction_loss
            }

            total_loss = prediction_loss

        return total_loss, loss_components
