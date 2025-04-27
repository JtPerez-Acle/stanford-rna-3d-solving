"""
Data handling for RNA 3D structure prediction models.

This module provides classes and functions for loading, preprocessing, and
batching RNA data for model training and evaluation.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class RNADataset(Dataset):
    """
    Dataset for RNA 3D structure prediction.

    This dataset loads RNA sequences and their corresponding 3D structures,
    and provides methods for preprocessing and augmentation.
    """

    def __init__(self, sequences_file, labels_file, msa_dir=None, transform=None):
        """
        Initialize the RNA dataset.

        Args:
            sequences_file (str or Path): Path to the sequences CSV file.
            labels_file (str or Path): Path to the labels CSV file.
            msa_dir (str or Path, optional): Directory containing MSA files.
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.sequences_file = Path(sequences_file)
        self.labels_file = Path(labels_file)
        self.msa_dir = Path(msa_dir) if msa_dir else None
        self.transform = transform

        # Load data
        self.sequences = pd.read_csv(self.sequences_file)
        self.labels = pd.read_csv(self.labels_file)

        # Process data
        self._process_data()

    def _process_data(self):
        """Process the raw data into a format suitable for training."""
        # Extract target IDs from labels
        self.labels['target_id'] = self.labels['ID'].apply(lambda x: x.split('_')[0])

        # Group labels by target_id
        self.grouped_labels = self.labels.groupby('target_id')

        # Create a mapping from target_id to index
        self.target_ids = self.sequences['target_id'].tolist()
        self.target_to_idx = {target_id: i for i, target_id in enumerate(self.target_ids)}

        # Create nucleotide to index mapping
        self.nucleotide_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4, '-': 5}

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing the sample data.
        """
        # Get sequence
        seq_row = self.sequences.iloc[idx]
        target_id = seq_row['target_id']
        sequence = seq_row['sequence']

        # Convert sequence to one-hot encoding
        seq_encoding = self._encode_sequence(sequence)

        # Get labels (3D coordinates)
        target_labels = self.labels[self.labels['target_id'] == target_id]

        # Extract coordinates
        coords = self._extract_coordinates(target_labels)

        # Create sample
        sample = {
            'target_id': target_id,
            'sequence': sequence,
            'sequence_encoding': seq_encoding,
            'coordinates': coords
        }

        # Load MSA data if available
        if self.msa_dir:
            msa_file = self.msa_dir / f"{target_id}.MSA.fasta"
            if msa_file.exists():
                sample['msa'] = self._load_msa(msa_file)

        # Apply transform if specified
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _encode_sequence(self, sequence):
        """
        Encode RNA sequence as a one-hot tensor.

        Args:
            sequence (str): RNA sequence.

        Returns:
            torch.Tensor: One-hot encoded sequence.
        """
        # Convert to uppercase
        sequence = sequence.upper()

        # Create one-hot encoding
        seq_len = len(sequence)
        encoding = torch.zeros(seq_len, len(self.nucleotide_to_idx))

        for i, nucleotide in enumerate(sequence):
            if nucleotide in self.nucleotide_to_idx:
                idx = self.nucleotide_to_idx[nucleotide]
            else:
                idx = self.nucleotide_to_idx['N']  # Unknown nucleotide

            encoding[i, idx] = 1.0

        return encoding

    def _extract_coordinates(self, labels_df):
        """
        Extract 3D coordinates from labels dataframe.

        Args:
            labels_df (pandas.DataFrame): DataFrame containing labels.

        Returns:
            torch.Tensor: 3D coordinates with shape (N, 3).
        """
        # Check if there are any labels
        if len(labels_df) == 0:
            return None

        # Get the first structure (x_1, y_1, z_1)
        coords = []
        for _, row in labels_df.iterrows():
            coords.append([row['x_1'], row['y_1'], row['z_1']])

        return torch.tensor(coords, dtype=torch.float32)

    def _load_msa(self, msa_file):
        """
        Load Multiple Sequence Alignment (MSA) data.

        Args:
            msa_file (Path): Path to the MSA file.

        Returns:
            list: List of aligned sequences.
        """
        # Simple FASTA parser
        sequences = []
        current_seq = ""

        with open(msa_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                    current_seq = ""
                else:
                    current_seq += line

            if current_seq:
                sequences.append(current_seq)

        return sequences


def rna_collate_fn(batch):
    """
    Custom collate function for RNA data.

    This function handles variable-length sequences by padding them to the same length.

    Args:
        batch (list): List of samples.

    Returns:
        dict: Batched data.
    """
    # Get batch size
    batch_size = len(batch)

    # Get maximum sequence length in the batch
    max_seq_len = max(sample['sequence_encoding'].shape[0] for sample in batch)

    # Initialize tensors
    sequence_encodings = torch.zeros(batch_size, max_seq_len, batch[0]['sequence_encoding'].shape[1])
    coordinates = torch.zeros(batch_size, max_seq_len, 3)

    # Create masks for padding
    masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    # Fill tensors
    target_ids = []
    sequences = []

    for i, sample in enumerate(batch):
        # Get sequence length
        seq_len = sample['sequence_encoding'].shape[0]

        # Add sequence encoding
        sequence_encodings[i, :seq_len, :] = sample['sequence_encoding']

        # Add coordinates if available
        if sample['coordinates'] is not None:
            coordinates[i, :seq_len, :] = sample['coordinates']

        # Add mask
        masks[i, :seq_len] = True

        # Add target ID and sequence
        target_ids.append(sample['target_id'])
        sequences.append(sample['sequence'])

    # Create batched sample
    batched_sample = {
        'target_id': target_ids,
        'sequence': sequences,
        'sequence_encoding': sequence_encodings,
        'coordinates': coordinates,
        'mask': masks
    }

    # Add MSA data if available
    if 'msa' in batch[0]:
        batched_sample['msa'] = [sample.get('msa', []) for sample in batch]

    return batched_sample


def create_data_loaders(data_dir, batch_size=16, num_workers=4, train_val_test_split=[0.8, 0.1, 0.1]):
    """
    Create data loaders for training, validation, and testing.

    Args:
        data_dir (str or Path): Directory containing the data.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.
        train_val_test_split (list): Train/validation/test split ratios.

    Returns:
        tuple: Tuple containing train, validation, and test data loaders.
    """
    data_dir = Path(data_dir)

    # Check for data files
    sequences_file = data_dir / "train_sequences.csv"
    labels_file = data_dir / "train_labels.csv"
    msa_dir = data_dir / "MSA"

    if not sequences_file.exists() or not labels_file.exists():
        raise FileNotFoundError(f"Data files not found in {data_dir}")

    # Create dataset
    dataset = RNADataset(sequences_file, labels_file, msa_dir if msa_dir.exists() else None)

    # Split dataset
    dataset_size = len(dataset)
    train_size = int(train_val_test_split[0] * dataset_size)
    val_size = int(train_val_test_split[1] * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rna_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rna_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rna_collate_fn
    )

    return train_loader, val_loader, test_loader
