"""
Tests for the data module.
"""

import pytest
import pandas as pd
import torch
import numpy as np
from pathlib import Path

from rna_folding.models.data import RNADataset, create_data_loaders


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create sample data files for testing."""
    # Create sequences file
    sequences = pd.DataFrame({
        'target_id': ['RNA1', 'RNA2', 'RNA3'],
        'sequence': ['GGAACC', 'AUUGC', 'GCUAGC']
    })
    sequences.to_csv(tmp_path / 'train_sequences.csv', index=False)
    
    # Create labels file
    labels_data = []
    for i, target_id in enumerate(['RNA1', 'RNA2', 'RNA3']):
        for j in range(len(sequences.iloc[i]['sequence'])):
            labels_data.append({
                'ID': f"{target_id}_{j+1}",
                'x_1': float(i + j),
                'y_1': float(i + j + 1),
                'z_1': float(i + j + 2)
            })
    
    labels = pd.DataFrame(labels_data)
    labels.to_csv(tmp_path / 'train_labels.csv', index=False)
    
    # Create MSA directory and files
    msa_dir = tmp_path / 'MSA'
    msa_dir.mkdir()
    
    for target_id in ['RNA1', 'RNA2', 'RNA3']:
        with open(msa_dir / f"{target_id}.MSA.fasta", 'w') as f:
            f.write(f">{target_id}\n")
            f.write("ACGUGC\n")
            f.write(">homolog1\n")
            f.write("ACGCGC\n")
    
    return tmp_path


def test_rna_dataset_initialization(sample_data_dir):
    """Test initialization of RNADataset."""
    dataset = RNADataset(
        sample_data_dir / 'train_sequences.csv',
        sample_data_dir / 'train_labels.csv',
        sample_data_dir / 'MSA'
    )
    
    assert len(dataset) == 3
    assert dataset.target_ids == ['RNA1', 'RNA2', 'RNA3']
    assert set(dataset.target_to_idx.keys()) == {'RNA1', 'RNA2', 'RNA3'}


def test_rna_dataset_getitem(sample_data_dir):
    """Test __getitem__ method of RNADataset."""
    dataset = RNADataset(
        sample_data_dir / 'train_sequences.csv',
        sample_data_dir / 'train_labels.csv',
        sample_data_dir / 'MSA'
    )
    
    sample = dataset[0]
    
    assert 'target_id' in sample
    assert 'sequence' in sample
    assert 'sequence_encoding' in sample
    assert 'coordinates' in sample
    assert 'msa' in sample
    
    assert isinstance(sample['sequence_encoding'], torch.Tensor)
    assert isinstance(sample['coordinates'], torch.Tensor)
    assert sample['coordinates'].shape[1] == 3  # 3D coordinates


def test_sequence_encoding(sample_data_dir):
    """Test sequence encoding in RNADataset."""
    dataset = RNADataset(
        sample_data_dir / 'train_sequences.csv',
        sample_data_dir / 'train_labels.csv'
    )
    
    sample = dataset[0]
    encoding = sample['sequence_encoding']
    
    # Check shape (sequence_length, num_nucleotides)
    assert encoding.shape == (len(sample['sequence']), len(dataset.nucleotide_to_idx))
    
    # Check that each position is one-hot encoded
    for i in range(encoding.shape[0]):
        assert torch.sum(encoding[i]) == 1.0


def test_create_data_loaders(sample_data_dir):
    """Test creation of data loaders."""
    train_loader, val_loader, test_loader = create_data_loaders(
        sample_data_dir,
        batch_size=1,
        num_workers=0,
        train_val_test_split=[0.6, 0.2, 0.2]
    )
    
    # Check that loaders are created
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    
    # Check total number of samples
    total_samples = sum(len(loader) for loader in [train_loader, val_loader, test_loader])
    assert total_samples == 3  # 3 samples in total
