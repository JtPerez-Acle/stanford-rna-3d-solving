"""
Module for analyzing RNA dataset and extracting insights.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import json

def analyze_sequence_lengths(sequences_df, output_dir=None):
    """
    Analyze the distribution of RNA sequence lengths.

    Args:
        sequences_df (pandas.DataFrame): DataFrame containing RNA sequences.
        output_dir (Path, optional): Directory to save analysis results.

    Returns:
        dict: Dictionary with analysis results.
    """
    # Calculate sequence lengths
    sequences_df['length'] = sequences_df['sequence'].apply(len)

    # Get basic statistics
    length_stats = {
        'min_length': int(sequences_df['length'].min()),
        'max_length': int(sequences_df['length'].max()),
        'mean_length': float(sequences_df['length'].mean()),
        'median_length': float(sequences_df['length'].median()),
        'std_length': float(sequences_df['length'].std())
    }

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(sequences_df['length'], bins=30, kde=True, ax=ax)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of RNA Sequence Lengths')

    # Add vertical lines for mean and median
    ax.axvline(length_stats['mean_length'], color='red', linestyle='--', label=f"Mean: {length_stats['mean_length']:.1f}")
    ax.axvline(length_stats['median_length'], color='green', linestyle='--', label=f"Median: {length_stats['median_length']:.1f}")
    ax.legend()

    # Save results if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save plot
        fig.savefig(output_dir / 'sequence_length_distribution.png', dpi=300, bbox_inches='tight')

        # Save statistics
        with open(output_dir / 'sequence_length_stats.json', 'w') as f:
            json.dump(length_stats, f, indent=4)

    plt.close(fig)
    return {'stats': length_stats, 'figure': fig}

def analyze_nucleotide_composition(sequences_df, output_dir=None):
    """
    Analyze the nucleotide composition of RNA sequences.

    Args:
        sequences_df (pandas.DataFrame): DataFrame containing RNA sequences.
        output_dir (Path, optional): Directory to save analysis results.

    Returns:
        dict: Dictionary with analysis results.
    """
    # Count nucleotides across all sequences
    all_nucleotides = ''.join(sequences_df['sequence'].tolist())
    nucleotide_counts = Counter(all_nucleotides)

    # Calculate percentages
    total_nucleotides = sum(nucleotide_counts.values())
    nucleotide_percentages = {nt: count / total_nucleotides * 100 for nt, count in nucleotide_counts.items()}

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define standard nucleotides and their colors
    standard_nucleotides = ['A', 'C', 'G', 'U', 'T']
    colors = ['#32CD32', '#1E90FF', '#FF8C00', '#DC143C', '#DC143C']  # Same colors as in visualization

    # Extract counts for standard nucleotides
    std_counts = [nucleotide_counts.get(nt, 0) for nt in standard_nucleotides]
    std_percentages = [nucleotide_percentages.get(nt, 0) for nt in standard_nucleotides]

    # Plot standard nucleotides
    bars = ax.bar(standard_nucleotides, std_counts, color=colors)

    # Add percentage labels
    for bar, percentage in zip(bars, std_percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{percentage:.1f}%', ha='center', va='bottom')

    # Plot other nucleotides if present
    other_nucleotides = [nt for nt in nucleotide_counts.keys() if nt not in standard_nucleotides]
    if other_nucleotides:
        other_counts = [nucleotide_counts[nt] for nt in other_nucleotides]
        other_percentages = [nucleotide_percentages[nt] for nt in other_nucleotides]

        # Add other nucleotides to the plot
        other_bars = ax.bar(other_nucleotides, other_counts, color='gray')

        # Add percentage labels for other nucleotides
        for bar, percentage in zip(other_bars, other_percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom')

    ax.set_xlabel('Nucleotide')
    ax.set_ylabel('Count')
    ax.set_title('Nucleotide Composition Across All Sequences')

    # Save results if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save plot
        fig.savefig(output_dir / 'nucleotide_composition.png', dpi=300, bbox_inches='tight')

        # Save statistics
        composition_stats = {
            'counts': {k: int(v) for k, v in nucleotide_counts.items()},
            'percentages': {k: float(v) for k, v in nucleotide_percentages.items()},
            'total_nucleotides': int(total_nucleotides)
        }

        with open(output_dir / 'nucleotide_composition.json', 'w') as f:
            json.dump(composition_stats, f, indent=4)

    plt.close(fig)
    return {
        'counts': nucleotide_counts,
        'percentages': nucleotide_percentages,
        'figure': fig
    }

def analyze_structure_statistics(labels_df, sequences_df, output_dir=None):
    """
    Analyze statistics of RNA 3D structures.

    Args:
        labels_df (pandas.DataFrame): DataFrame containing RNA structure coordinates.
        sequences_df (pandas.DataFrame): DataFrame containing RNA sequences.
        output_dir (Path, optional): Directory to save analysis results.

    Returns:
        dict: Dictionary with analysis results.
    """
    # Extract target IDs from labels
    labels_df['target_id'] = labels_df['ID'].apply(lambda x: x.split('_')[0])

    # Group by target_id to analyze each structure
    structure_stats = []

    for target_id, group in labels_df.groupby('target_id'):
        # Get sequence information
        seq_info = sequences_df[sequences_df['target_id'] == target_id]
        if len(seq_info) == 0:
            continue

        seq_length = len(seq_info.iloc[0]['sequence'])

        # Extract coordinates
        coords = np.array([
            group['x_1'].values,
            group['y_1'].values,
            group['z_1'].values
        ]).T

        # Calculate structure statistics
        # 1. Radius of gyration (measure of compactness)
        center_of_mass = coords.mean(axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - center_of_mass)**2, axis=1)))

        # 2. Maximum distance between any two nucleotides
        max_distance = 0
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                max_distance = max(max_distance, dist)

        # 3. End-to-end distance
        end_to_end = np.linalg.norm(coords[0] - coords[-1]) if len(coords) > 1 else 0

        # Store statistics
        structure_stats.append({
            'target_id': target_id,
            'sequence_length': seq_length,
            'radius_of_gyration': float(rg),
            'max_distance': float(max_distance),
            'end_to_end_distance': float(end_to_end)
        })

    # Convert to DataFrame
    stats_df = pd.DataFrame(structure_stats)

    # Create empty figures list
    figures = []

    # Only create plots if we have data
    if not stats_df.empty and 'sequence_length' in stats_df.columns:
        # Create scatter plot of radius of gyration vs sequence length
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=stats_df, x='sequence_length', y='radius_of_gyration', ax=ax1)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Radius of Gyration (Å)')
        ax1.set_title('Radius of Gyration vs Sequence Length')
        figures.append(fig1)

        # Create scatter plot of max distance vs sequence length
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=stats_df, x='sequence_length', y='max_distance', ax=ax2)
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Maximum Distance (Å)')
        ax2.set_title('Maximum Distance vs Sequence Length')
        figures.append(fig2)

        # Create scatter plot of end-to-end distance vs sequence length
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=stats_df, x='sequence_length', y='end_to_end_distance', ax=ax3)
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('End-to-End Distance (Å)')
        ax3.set_title('End-to-End Distance vs Sequence Length')
        figures.append(fig3)
    else:
        print("Warning: No structure statistics data available for plotting.")
        # Create empty figures to return
        fig1 = plt.figure(figsize=(10, 6))
        fig2 = plt.figure(figsize=(10, 6))
        fig3 = plt.figure(figsize=(10, 6))
        figures = [fig1, fig2, fig3]

    # Save results if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save plots if we have data
        if not stats_df.empty and len(figures) == 3:
            figures[0].savefig(output_dir / 'radius_of_gyration_vs_length.png', dpi=300, bbox_inches='tight')
            figures[1].savefig(output_dir / 'max_distance_vs_length.png', dpi=300, bbox_inches='tight')
            figures[2].savefig(output_dir / 'end_to_end_distance_vs_length.png', dpi=300, bbox_inches='tight')

            # Save statistics
            stats_df.to_csv(output_dir / 'structure_statistics.csv', index=False)
        else:
            # Create an empty file to indicate that analysis was attempted
            with open(output_dir / 'structure_statistics_empty.txt', 'w') as f:
                f.write("No structure statistics data available for analysis.")

    # Close all figures
    for fig in figures:
        plt.close(fig)

    return {
        'stats_df': stats_df,
        'figures': figures
    }

def analyze_temporal_distribution(sequences_df, output_dir=None):
    """
    Analyze the temporal distribution of RNA structures.

    Args:
        sequences_df (pandas.DataFrame): DataFrame containing RNA sequences.
        output_dir (Path, optional): Directory to save analysis results.

    Returns:
        dict: Dictionary with analysis results.
    """
    # Convert temporal_cutoff to datetime
    sequences_df['temporal_cutoff'] = pd.to_datetime(sequences_df['temporal_cutoff'])

    # Group by year and month
    sequences_df['year'] = sequences_df['temporal_cutoff'].dt.year
    sequences_df['month'] = sequences_df['temporal_cutoff'].dt.month
    sequences_df['year_month'] = sequences_df['temporal_cutoff'].dt.strftime('%Y-%m')

    # Count structures by year-month
    temporal_counts = sequences_df.groupby('year_month').size().reset_index(name='count')
    temporal_counts['year_month'] = pd.to_datetime(temporal_counts['year_month'])
    temporal_counts = temporal_counts.sort_values('year_month')

    # Create time series plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(temporal_counts['year_month'], temporal_counts['count'], marker='o')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Structures')
    ax.set_title('Temporal Distribution of RNA Structures')

    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save results if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save plot
        fig.savefig(output_dir / 'temporal_distribution.png', dpi=300, bbox_inches='tight')

        # Save data
        temporal_counts.to_csv(output_dir / 'temporal_distribution.csv', index=False)

    plt.close(fig)
    return {
        'temporal_counts': temporal_counts,
        'figure': fig
    }

def generate_readme(output_dir, summary, analyses_run):
    """
    Generate a README file documenting the analysis outputs.

    Args:
        output_dir (Path): Directory where analysis results are saved.
        summary (dict): Summary of the analysis results.
        analyses_run (list): List of analyses that were run.
    """
    from datetime import datetime

    readme_content = f"""# RNA 3D Folding Analysis Results

## Overview
- **Date of Analysis**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Number of Sequences**: {summary.get('num_sequences', 'N/A')}
- **Number of Structures**: {summary.get('num_structures', 'N/A')}

## Analyses Performed
"""

    if 'sequence_length_stats' in summary and 'sequence_length' in analyses_run:
        stats = summary['sequence_length_stats']
        readme_content += f"""
### Sequence Length Analysis
- **Minimum Length**: {stats.get('min_length', 'N/A')}
- **Maximum Length**: {stats.get('max_length', 'N/A')}
- **Mean Length**: {stats.get('mean_length', 'N/A'):.2f}
- **Median Length**: {stats.get('median_length', 'N/A'):.2f}
- **Standard Deviation**: {stats.get('std_length', 'N/A'):.2f}

**Files**:
- `sequence_length_distribution.png`: Histogram of RNA sequence lengths
- `sequence_length_stats.json`: Detailed statistics
"""

    if 'nucleotide_composition' in summary and 'nucleotide_composition' in analyses_run:
        composition = summary['nucleotide_composition']
        readme_content += f"""
### Nucleotide Composition Analysis
- **Nucleotide Frequencies**:
"""
        for nt, pct in composition.items():
            readme_content += f"  - {nt}: {pct:.2f}%\n"

        readme_content += """
**Files**:
- `nucleotide_composition.png`: Bar chart of nucleotide frequencies
- `nucleotide_composition.json`: Detailed counts and percentages
"""

    if 'structure_statistics' in analyses_run:
        readme_content += """
### Structure Statistics Analysis
Analysis of 3D structural properties of RNA molecules.

**Files**:
- `radius_of_gyration_vs_length.png`: Scatter plot of radius of gyration vs sequence length
- `max_distance_vs_length.png`: Scatter plot of maximum distance vs sequence length
- `end_to_end_distance_vs_length.png`: Scatter plot of end-to-end distance vs sequence length
- `structure_statistics.csv`: Detailed statistics for each RNA structure
"""

    if 'temporal_distribution' in analyses_run:
        readme_content += """
### Temporal Distribution Analysis
Analysis of how RNA structures are distributed over time.

**Files**:
- `temporal_distribution.png`: Time series plot of RNA structures over time
- `temporal_distribution.csv`: Counts of structures by year-month
"""

    readme_content += """
## Next Steps
- Consider running secondary structure prediction analysis
- Analyze base-pairing probabilities
- Identify common structural motifs
- Compare structures across different RNA families
"""

    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)

def analyze_dataset(data_dir, output_dir=None):
    """
    Perform comprehensive analysis of the RNA dataset.

    Args:
        data_dir (str or Path): Directory containing the data files.
        output_dir (str or Path, optional): Directory to save analysis results.

    Returns:
        dict: Dictionary with analysis results.
    """
    data_dir = Path(data_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_sequences = pd.read_csv(data_dir / "train_sequences.csv")
    train_labels = pd.read_csv(data_dir / "train_labels.csv")

    # Track which analyses were run successfully
    analyses_run = []

    # Run analyses
    print("Analyzing sequence lengths...")
    length_analysis = analyze_sequence_lengths(train_sequences, output_dir)
    analyses_run.append('sequence_length')

    print("Analyzing nucleotide composition...")
    composition_analysis = analyze_nucleotide_composition(train_sequences, output_dir)
    analyses_run.append('nucleotide_composition')

    print("Analyzing structure statistics...")
    structure_analysis = analyze_structure_statistics(train_labels, train_sequences, output_dir)
    if not structure_analysis['stats_df'].empty:
        analyses_run.append('structure_statistics')

    print("Analyzing temporal distribution...")
    temporal_analysis = analyze_temporal_distribution(train_sequences, output_dir)
    analyses_run.append('temporal_distribution')

    # Create summary report
    summary = {
        'num_sequences': len(train_sequences),
        'num_structures': len(train_sequences.target_id.unique()),
        'sequence_length_stats': length_analysis['stats'],
        'nucleotide_composition': {k: float(v) for k, v in composition_analysis['percentages'].items()},
    }

    if output_dir:
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)

        # Generate README file
        generate_readme(output_dir, summary, analyses_run)

    print("Analysis complete!")
    return {
        'summary': summary,
        'length_analysis': length_analysis,
        'composition_analysis': composition_analysis,
        'structure_analysis': structure_analysis,
        'temporal_analysis': temporal_analysis
    }

def main():
    """Main function to demonstrate analysis capabilities."""
    data_dir = Path("data/raw")
    output_dir = Path("data/analysis")

    analyze_dataset(data_dir, output_dir)

if __name__ == "__main__":
    main()
