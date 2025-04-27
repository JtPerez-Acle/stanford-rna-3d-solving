"""
Tests for the RNA structure visualization functionality.
"""

import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from rna_folding.visualization.visualize import (
    plot_rna_sequence,
    plot_rna_structure_2d,
    create_rna_structure_3d,
    save_visualization
)

# Define constants for testing
DATA_DIR = Path("data/raw")

class TestVisualization:
    """Tests for visualization functionality."""
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Fixture to load sample data for visualization tests."""
        # Check if data exists, if not, skip tests
        train_seq_path = DATA_DIR / "train_sequences.csv"
        train_labels_path = DATA_DIR / "train_labels.csv"
        
        if not train_seq_path.exists() or not train_labels_path.exists():
            pytest.skip("Test data not available. Run download_data.py first.")
        
        # Load a small sample of data
        train_sequences = pd.read_csv(train_seq_path)
        train_labels = pd.read_csv(train_labels_path)
        
        # Get the first sequence and its corresponding labels
        sample_seq = train_sequences.iloc[0]
        sample_id = sample_seq['target_id']
        sample_labels = train_labels[train_labels['ID'].str.startswith(f"{sample_id}_")]
        
        return {
            'sequence': sample_seq,
            'labels': sample_labels
        }
    
    def test_plot_rna_sequence(self, sample_data):
        """Test plotting RNA sequence."""
        sequence = sample_data['sequence']['sequence']
        fig = plot_rna_sequence(sequence)
        
        assert isinstance(fig, plt.Figure), "plot_rna_sequence should return a matplotlib Figure"
        plt.close(fig)
    
    def test_plot_rna_structure_2d(self, sample_data):
        """Test plotting RNA structure in 2D."""
        labels = sample_data['labels']
        
        # Extract coordinates
        coords = np.array([
            labels['x_1'].values,
            labels['y_1'].values,
            labels['z_1'].values
        ]).T
        
        fig = plot_rna_structure_2d(coords)
        
        assert isinstance(fig, plt.Figure), "plot_rna_structure_2d should return a matplotlib Figure"
        plt.close(fig)
    
    def test_create_rna_structure_3d(self, sample_data):
        """Test creating 3D visualization of RNA structure."""
        labels = sample_data['labels']
        sequence = sample_data['sequence']['sequence']
        
        # Extract coordinates
        coords = np.array([
            labels['x_1'].values,
            labels['y_1'].values,
            labels['z_1'].values
        ]).T
        
        view = create_rna_structure_3d(coords, sequence)
        
        # This is a bit tricky to test as py3Dmol returns a custom object
        # We'll just check that it's not None
        assert view is not None, "create_rna_structure_3d should return a py3Dmol view object"
    
    def test_save_visualization(self, sample_data, tmp_path):
        """Test saving visualization to file."""
        sequence = sample_data['sequence']['sequence']
        fig = plot_rna_sequence(sequence)
        
        output_file = tmp_path / "test_visualization.png"
        save_visualization(fig, output_file)
        
        assert output_file.exists(), f"Visualization file {output_file} was not created"
        plt.close(fig)
