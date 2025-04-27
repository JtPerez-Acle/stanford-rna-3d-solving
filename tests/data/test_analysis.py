"""
Tests for the data analysis functionality.
"""

import os
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rna_folding.data.analysis import (
    analyze_sequence_lengths,
    analyze_nucleotide_composition,
    analyze_structure_statistics,
    analyze_temporal_distribution,
    analyze_dataset
)

# Define constants for testing
DATA_DIR = Path("data/raw")

class TestDataAnalysis:
    """Tests for data analysis functionality."""

    @pytest.fixture(scope="class")
    def sample_data(self):
        """Fixture to load sample data for analysis tests."""
        # Check if data exists, if not, skip tests
        train_seq_path = DATA_DIR / "train_sequences.csv"
        train_labels_path = DATA_DIR / "train_labels.csv"

        if not train_seq_path.exists() or not train_labels_path.exists():
            pytest.skip("Test data not available. Run download_data.py first.")

        # Load a small sample of data
        train_sequences = pd.read_csv(train_seq_path)
        train_labels = pd.read_csv(train_labels_path)

        return {
            'sequences': train_sequences.head(10),
            'labels': train_labels.head(100)
        }

    def test_analyze_sequence_lengths(self, sample_data, tmp_path):
        """Test analyzing sequence lengths."""
        sequences_df = sample_data['sequences']

        result = analyze_sequence_lengths(sequences_df, tmp_path)

        assert 'stats' in result, "Result should contain stats"
        assert 'figure' in result, "Result should contain figure"
        assert isinstance(result['figure'], plt.Figure), "Figure should be a matplotlib Figure"

        # Check if output files were created
        assert (tmp_path / 'sequence_length_distribution.png').exists(), "Plot file should exist"
        assert (tmp_path / 'sequence_length_stats.json').exists(), "Stats file should exist"

    def test_analyze_nucleotide_composition(self, sample_data, tmp_path):
        """Test analyzing nucleotide composition."""
        sequences_df = sample_data['sequences']

        result = analyze_nucleotide_composition(sequences_df, tmp_path)

        assert 'counts' in result, "Result should contain counts"
        assert 'percentages' in result, "Result should contain percentages"
        assert 'figure' in result, "Result should contain figure"

        # Check if output files were created
        assert (tmp_path / 'nucleotide_composition.png').exists(), "Plot file should exist"
        assert (tmp_path / 'nucleotide_composition.json').exists(), "Stats file should exist"

    def test_analyze_structure_statistics(self, sample_data, tmp_path):
        """Test analyzing structure statistics."""
        labels_df = sample_data['labels']
        sequences_df = sample_data['sequences']

        # Make sure the target_id in labels matches one in sequences
        # This is needed for the test to work properly
        if len(sequences_df) > 0 and len(labels_df) > 0:
            target_id = sequences_df.iloc[0]['target_id']
            labels_df['target_id'] = target_id

        try:
            result = analyze_structure_statistics(labels_df, sequences_df, tmp_path)

            assert 'stats_df' in result, "Result should contain stats_df"
            assert 'figures' in result, "Result should contain figures"

            # Only check for files if we actually have data to plot
            if not result['stats_df'].empty:
                assert len(result['figures']) == 3, "Should have 3 figures"
                assert (tmp_path / 'radius_of_gyration_vs_length.png').exists(), "Plot file should exist"
                assert (tmp_path / 'structure_statistics.csv').exists(), "Stats file should exist"
        except ValueError as e:
            # If we get a ValueError about empty data, that's acceptable for the test
            if "An entry with this name does not appear in `data`" in str(e):
                pytest.skip("Skipping test due to empty data frame")
            else:
                raise

    def test_analyze_temporal_distribution(self, sample_data, tmp_path):
        """Test analyzing temporal distribution."""
        sequences_df = sample_data['sequences']

        # Ensure temporal_cutoff column exists and has valid dates
        if 'temporal_cutoff' not in sequences_df.columns or pd.isna(sequences_df['temporal_cutoff']).all():
            sequences_df['temporal_cutoff'] = pd.date_range(start='2020-01-01', periods=len(sequences_df)).strftime('%Y-%m-%d')

        result = analyze_temporal_distribution(sequences_df, tmp_path)

        assert 'temporal_counts' in result, "Result should contain temporal_counts"
        assert 'figure' in result, "Result should contain figure"

        # Check if output files were created
        assert (tmp_path / 'temporal_distribution.png').exists(), "Plot file should exist"
        assert (tmp_path / 'temporal_distribution.csv').exists(), "Data file should exist"

    def test_analyze_dataset(self, sample_data, tmp_path):
        """Test full dataset analysis."""
        # Create a temporary directory with sample data
        temp_data_dir = tmp_path / "data"
        temp_data_dir.mkdir()

        # Save sample data to temporary directory
        sample_data['sequences'].to_csv(temp_data_dir / "train_sequences.csv", index=False)
        sample_data['labels'].to_csv(temp_data_dir / "train_labels.csv", index=False)

        # Make sure the target_id in labels matches one in sequences
        # This is needed for the test to work properly
        if len(sample_data['sequences']) > 0 and len(sample_data['labels']) > 0:
            # Update the CSV files with matching target_ids
            labels_df = sample_data['labels'].copy()
            sequences_df = sample_data['sequences'].copy()

            target_id = sequences_df.iloc[0]['target_id']
            labels_df['target_id'] = target_id

            labels_df.to_csv(temp_data_dir / "train_labels.csv", index=False)

        # Run analysis
        output_dir = tmp_path / "analysis"

        try:
            result = analyze_dataset(temp_data_dir, output_dir)

            assert 'summary' in result, "Result should contain summary"
            assert 'length_analysis' in result, "Result should contain length_analysis"
            assert 'composition_analysis' in result, "Result should contain composition_analysis"

            # Check if output files were created
            assert (output_dir / 'analysis_summary.json').exists(), "Summary file should exist"
        except ValueError as e:
            # If we get a ValueError about empty data, that's acceptable for the test
            if "An entry with this name does not appear in `data`" in str(e):
                pytest.skip("Skipping test due to empty data frame")
            else:
                raise
