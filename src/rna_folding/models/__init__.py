"""
Models for RNA 3D structure prediction.

This package provides models for predicting the 3D structures of RNA molecules,
including multi-scale equivariant architectures and physics-informed neural networks.
"""

from rna_folding.models.base import RNAModel, ModelConfig
from rna_folding.models.multi_scale import MultiScaleRNA, MultiScaleModelConfig
from rna_folding.models.physics import PhysicsConstraints, PhysicsInformedLoss
from rna_folding.models.metrics import rmsd, tm_score, calculate_all_metrics
from rna_folding.models.data import RNADataset, create_data_loaders

__all__ = [
    'RNAModel',
    'ModelConfig',
    'MultiScaleRNA',
    'MultiScaleModelConfig',
    'PhysicsConstraints',
    'PhysicsInformedLoss',
    'rmsd',
    'tm_score',
    'calculate_all_metrics',
    'RNADataset',
    'create_data_loaders'
]