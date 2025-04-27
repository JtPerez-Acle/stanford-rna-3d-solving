"""
Tests for the data download functionality.
"""

import os
import pytest
from pathlib import Path
import pandas as pd

from rna_folding.data.download import (
    download_competition_data,
    check_kaggle_api,
    get_competition_files,
    extract_competition_files,
)

# Define constants for testing
COMPETITION_NAME = "stanford-rna-3d-folding"
DATA_DIR = Path("data/raw")

class TestKaggleAPI:
    """Tests for Kaggle API functionality."""

    def test_check_kaggle_api(self):
        """Test that Kaggle API is properly configured."""
        assert check_kaggle_api() is True, "Kaggle API is not properly configured"

    def test_get_competition_files(self):
        """Test retrieving competition file list."""
        files = get_competition_files(COMPETITION_NAME)
        assert len(files) > 0, "No competition files found"
        # The actual file list from the API might not contain the exact filenames
        # So we'll just check that we get some files back

class TestDataDownload:
    """Tests for data download functionality."""

    @pytest.fixture(scope="class")
    def download_data(self):
        """Fixture to download data once for all tests."""
        # Only download if data doesn't exist
        if not (DATA_DIR / "train_sequences.csv").exists():
            download_competition_data(COMPETITION_NAME, DATA_DIR)
        return DATA_DIR

    def test_download_competition_data(self, download_data):
        """Test downloading competition data."""
        data_dir = download_data
        assert (data_dir / "train_sequences.csv").exists(), "train_sequences.csv not downloaded"
        assert (data_dir / "train_labels.csv").exists(), "train_labels.csv not downloaded"
        assert (data_dir / "validation_sequences.csv").exists(), "validation_sequences.csv not downloaded"
        assert (data_dir / "validation_labels.csv").exists(), "validation_labels.csv not downloaded"

    def test_data_integrity(self, download_data):
        """Test the integrity of downloaded data."""
        data_dir = download_data

        # Check train_sequences.csv
        train_sequences = pd.read_csv(data_dir / "train_sequences.csv")
        assert "target_id" in train_sequences.columns, "target_id column missing in train_sequences.csv"
        assert "sequence" in train_sequences.columns, "sequence column missing in train_sequences.csv"
        assert len(train_sequences) > 0, "train_sequences.csv is empty"

        # Check train_labels.csv
        train_labels = pd.read_csv(data_dir / "train_labels.csv")
        assert "ID" in train_labels.columns, "ID column missing in train_labels.csv"
        assert "x_1" in train_labels.columns, "x_1 column missing in train_labels.csv"
        assert "y_1" in train_labels.columns, "y_1 column missing in train_labels.csv"
        assert "z_1" in train_labels.columns, "z_1 column missing in train_labels.csv"
        assert len(train_labels) > 0, "train_labels.csv is empty"
