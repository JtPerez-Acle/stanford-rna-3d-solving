"""
Core analysis module for RNA 3D structure prediction.

This module provides comprehensive analysis of RNA sequences and structures,
focusing on extracting features that are critical for developing groundbreaking
RNA 3D structure prediction models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import Counter, defaultdict
import warnings

# Try to import BioPython modules
try:
    from Bio import SeqIO
    from Bio.PDB import PDBParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    # Suppress BioPython warnings
    warnings.filterwarnings('ignore', category=PDBConstructionWarning)
except ImportError:
    print("BioPython not installed. Some functionality may be limited.")
    SeqIO = None
    PDBParser = None

class RNAAnalysis:
    """
    Comprehensive analysis of RNA sequences and structures.

    This class provides methods for analyzing RNA data at multiple levels:
    1. Sequence analysis
    2. Structure analysis
    3. Evolutionary analysis

    It serves as the foundation for developing groundbreaking RNA 3D structure
    prediction models by extracting and visualizing key features.
    """

    def __init__(self, data_dir, output_dir=None):
        """
        Initialize the RNA analysis module.

        Args:
            data_dir (str or Path): Directory containing the RNA data.
            output_dir (str or Path, optional): Directory to save analysis results.
        """
        self.data_dir = Path(data_dir)

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

        # Load data
        self.sequences = None
        self.labels = None
        self.msa_data = None

        # Analysis results
        self.sequence_analysis = {}
        self.structure_analysis = {}
        self.evolutionary_analysis = {}

        # Load data if available
        self._load_data()

    def _load_data(self):
        """Load RNA data from the data directory."""
        # Load sequence data
        seq_path = self.data_dir / "train_sequences.csv"
        if seq_path.exists():
            self.sequences = pd.read_csv(seq_path)
            print(f"Loaded {len(self.sequences)} RNA sequences.")

        # Load structure data
        labels_path = self.data_dir / "train_labels.csv"
        if labels_path.exists():
            self.labels = pd.read_csv(labels_path)
            print(f"Loaded {len(self.labels)} structure labels.")

        # Check for MSA data
        msa_dir = self.data_dir / "MSA"
        if msa_dir.exists() and msa_dir.is_dir():
            self.msa_data = True
            print("MSA data directory found.")

    def analyze_sequences(self):
        """
        Perform comprehensive analysis of RNA sequences.

        This method analyzes:
        1. Sequence length distribution
        2. Nucleotide composition
        3. Sequence complexity
        4. Motif identification

        Returns:
            dict: Dictionary containing sequence analysis results.
        """
        if self.sequences is None:
            print("No sequence data available.")
            return {}

        print("Analyzing RNA sequences...")

        # Calculate sequence lengths
        self.sequences['length'] = self.sequences['sequence'].apply(len)

        # Basic statistics
        length_stats = {
            'min_length': int(self.sequences['length'].min()),
            'max_length': int(self.sequences['length'].max()),
            'mean_length': float(self.sequences['length'].mean()),
            'median_length': float(self.sequences['length'].median()),
            'std_length': float(self.sequences['length'].std()),
            'quartiles': [
                float(self.sequences['length'].quantile(0.25)),
                float(self.sequences['length'].quantile(0.5)),
                float(self.sequences['length'].quantile(0.75))
            ]
        }

        # Nucleotide composition
        all_nucleotides = ''.join(self.sequences['sequence'].tolist())
        nucleotide_counts = Counter(all_nucleotides)
        total_nucleotides = sum(nucleotide_counts.values())
        nucleotide_percentages = {nt: count / total_nucleotides * 100
                                 for nt, count in nucleotide_counts.items()}

        # Sequence complexity (entropy)
        def calculate_entropy(sequence):
            """Calculate Shannon entropy of a sequence."""
            counts = Counter(sequence)
            length = len(sequence)
            probabilities = [count / length for count in counts.values()]
            return -sum(p * np.log2(p) for p in probabilities)

        self.sequences['entropy'] = self.sequences['sequence'].apply(calculate_entropy)
        entropy_stats = {
            'min_entropy': float(self.sequences['entropy'].min()),
            'max_entropy': float(self.sequences['entropy'].max()),
            'mean_entropy': float(self.sequences['entropy'].mean()),
            'median_entropy': float(self.sequences['entropy'].median())
        }

        # Dinucleotide frequencies
        dinucleotides = []
        for seq in self.sequences['sequence']:
            for i in range(len(seq) - 1):
                dinucleotides.append(seq[i:i+2])

        dinucleotide_counts = Counter(dinucleotides)
        total_dinucleotides = sum(dinucleotide_counts.values())
        dinucleotide_percentages = {dn: count / total_dinucleotides * 100
                                   for dn, count in dinucleotide_counts.items()}

        # Store results
        self.sequence_analysis = {
            'length_stats': length_stats,
            'nucleotide_composition': nucleotide_percentages,
            'entropy_stats': entropy_stats,
            'dinucleotide_frequencies': dinucleotide_percentages
        }

        # Generate visualizations if output directory is provided
        if self.output_dir:
            self._visualize_sequence_analysis()

        print("Sequence analysis complete.")
        return self.sequence_analysis

    def _visualize_sequence_analysis(self):
        """Generate visualizations for sequence analysis."""
        viz_dir = self.output_dir / "sequence_analysis"
        viz_dir.mkdir(exist_ok=True)

        # Sequence length distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.sequences['length'], bins=50, kde=True)
        plt.xlabel('Sequence Length')
        plt.ylabel('Count')
        plt.title('Distribution of RNA Sequence Lengths')
        plt.axvline(self.sequence_analysis['length_stats']['mean_length'],
                   color='red', linestyle='--',
                   label=f"Mean: {self.sequence_analysis['length_stats']['mean_length']:.1f}")
        plt.axvline(self.sequence_analysis['length_stats']['median_length'],
                   color='green', linestyle='--',
                   label=f"Median: {self.sequence_analysis['length_stats']['median_length']:.1f}")
        plt.legend()
        plt.savefig(viz_dir / 'sequence_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Nucleotide composition
        plt.figure(figsize=(10, 6))
        standard_nucleotides = ['A', 'C', 'G', 'U', 'T']
        nucleotide_comp = self.sequence_analysis['nucleotide_composition']

        # Extract standard nucleotides
        std_nucleotides = {nt: nucleotide_comp.get(nt, 0) for nt in standard_nucleotides}
        other_nucleotides = {nt: pct for nt, pct in nucleotide_comp.items()
                            if nt not in standard_nucleotides}

        # Plot standard nucleotides
        colors = ['#32CD32', '#1E90FF', '#FF8C00', '#DC143C', '#DC143C']
        bars = plt.bar(std_nucleotides.keys(), std_nucleotides.values(), color=colors)

        # Add percentage labels
        for bar, pct in zip(bars, std_nucleotides.values()):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom')

        plt.xlabel('Nucleotide')
        plt.ylabel('Percentage')
        plt.title('Nucleotide Composition')
        plt.savefig(viz_dir / 'nucleotide_composition.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Sequence entropy distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.sequences['entropy'], bins=30, kde=True)
        plt.xlabel('Sequence Entropy')
        plt.ylabel('Count')
        plt.title('Distribution of RNA Sequence Entropy')
        plt.savefig(viz_dir / 'sequence_entropy_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Dinucleotide frequencies
        plt.figure(figsize=(12, 8))
        dinucleotide_freq = self.sequence_analysis['dinucleotide_frequencies']

        # Sort dinucleotides by frequency
        sorted_dinucleotides = sorted(dinucleotide_freq.items(), key=lambda x: x[1], reverse=True)

        # Plot top 20 dinucleotides
        top_dinucleotides = sorted_dinucleotides[:20]
        plt.bar([x[0] for x in top_dinucleotides], [x[1] for x in top_dinucleotides])
        plt.xlabel('Dinucleotide')
        plt.ylabel('Percentage')
        plt.title('Top 20 Dinucleotide Frequencies')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / 'dinucleotide_frequencies.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save analysis results as JSON
        with open(viz_dir / 'sequence_analysis.json', 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            analysis_json = json.dumps(self.sequence_analysis, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
            f.write(analysis_json)

    def analyze_structures(self):
        """
        Perform comprehensive analysis of RNA structures.

        This method analyzes:
        1. 3D coordinate statistics
        2. Structural features (radius of gyration, end-to-end distance)
        3. Base-pairing patterns
        4. Structural motifs

        Returns:
            dict: Dictionary containing structure analysis results.
        """
        if self.labels is None or self.sequences is None:
            print("Structure or sequence data not available.")
            return {}

        print("Analyzing RNA structures...")

        # Extract target IDs from labels
        self.labels['target_id'] = self.labels['ID'].apply(lambda x: x.split('_')[0])

        # Group by target_id to analyze each structure
        structure_stats = []

        for target_id, group in self.labels.groupby('target_id'):
            # Get sequence information
            seq_info = self.sequences[self.sequences['target_id'] == target_id]
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

            # 4. Average distance between consecutive nucleotides
            consecutive_distances = []
            for i in range(len(coords) - 1):
                dist = np.linalg.norm(coords[i] - coords[i+1])
                consecutive_distances.append(dist)
            avg_consecutive_distance = np.mean(consecutive_distances) if consecutive_distances else 0

            # 5. Structural compactness (ratio of Rg to max possible Rg for linear chain)
            # Max Rg for linear chain scales with sqrt(N)
            max_possible_rg = np.sqrt(seq_length) * avg_consecutive_distance / np.sqrt(6)
            compactness = rg / max_possible_rg if max_possible_rg > 0 else 0

            # Store statistics
            structure_stats.append({
                'target_id': target_id,
                'sequence_length': seq_length,
                'radius_of_gyration': float(rg),
                'max_distance': float(max_distance),
                'end_to_end_distance': float(end_to_end),
                'avg_consecutive_distance': float(avg_consecutive_distance),
                'compactness': float(compactness)
            })

        # Convert to DataFrame
        if structure_stats:
            stats_df = pd.DataFrame(structure_stats)

            # Calculate overall statistics
            overall_stats = {
                'mean_radius_of_gyration': float(stats_df['radius_of_gyration'].mean()),
                'mean_compactness': float(stats_df['compactness'].mean()),
                'mean_end_to_end_distance': float(stats_df['end_to_end_distance'].mean()),
                'rg_length_correlation': float(np.corrcoef(
                    stats_df['sequence_length'], stats_df['radius_of_gyration']
                )[0, 1]) if len(stats_df) > 1 else 0
            }

            # Store results
            self.structure_analysis = {
                'structure_stats': structure_stats,
                'overall_stats': overall_stats
            }

            # Generate visualizations if output directory is provided
            if self.output_dir:
                self._visualize_structure_analysis(stats_df)

            print("Structure analysis complete.")
            return self.structure_analysis
        else:
            print("No structure statistics could be calculated.")
            return {}

    def _visualize_structure_analysis(self, stats_df):
        """Generate visualizations for structure analysis."""
        viz_dir = self.output_dir / "structure_analysis"
        viz_dir.mkdir(exist_ok=True)

        # Radius of gyration vs sequence length
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=stats_df, x='sequence_length', y='radius_of_gyration')
        plt.xlabel('Sequence Length')
        plt.ylabel('Radius of Gyration (Å)')
        plt.title('Radius of Gyration vs Sequence Length')

        # Add trend line
        if len(stats_df) > 1:
            x = stats_df['sequence_length']
            y = stats_df['radius_of_gyration']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.8,
                    label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
            plt.legend()

        plt.savefig(viz_dir / 'radius_of_gyration_vs_length.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Compactness distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(stats_df['compactness'], bins=30, kde=True)
        plt.xlabel('Compactness (Rg/Rg_max)')
        plt.ylabel('Count')
        plt.title('Distribution of RNA Structure Compactness')
        plt.savefig(viz_dir / 'compactness_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # End-to-end distance vs sequence length
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=stats_df, x='sequence_length', y='end_to_end_distance')
        plt.xlabel('Sequence Length')
        plt.ylabel('End-to-End Distance (Å)')
        plt.title('End-to-End Distance vs Sequence Length')
        plt.savefig(viz_dir / 'end_to_end_distance_vs_length.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Consecutive distance distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(stats_df['avg_consecutive_distance'], bins=30, kde=True)
        plt.xlabel('Average Distance Between Consecutive Nucleotides (Å)')
        plt.ylabel('Count')
        plt.title('Distribution of Average Consecutive Nucleotide Distances')
        plt.savefig(viz_dir / 'consecutive_distance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save structure statistics
        stats_df.to_csv(viz_dir / 'structure_statistics.csv', index=False)

        # Save analysis results as JSON
        with open(viz_dir / 'structure_analysis.json', 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            analysis_json = json.dumps(self.structure_analysis, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
            f.write(analysis_json)

    def analyze_evolutionary_information(self):
        """
        Analyze evolutionary information from MSA data if available.

        This method analyzes:
        1. Sequence conservation
        2. Covariation between positions
        3. Evolutionary couplings

        Returns:
            dict: Dictionary containing evolutionary analysis results.
        """
        if not self.msa_data:
            print("MSA data not available.")
            return {}

        print("Analyzing evolutionary information...")

        # This is a placeholder for MSA analysis
        # In a real implementation, we would parse and analyze the MSA files

        self.evolutionary_analysis = {
            "status": "MSA analysis not yet implemented",
            "note": "This will be implemented in the next iteration"
        }

        print("Evolutionary analysis placeholder complete.")
        return self.evolutionary_analysis

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive report of all analyses.

        This method combines the results of sequence, structure, and evolutionary
        analyses into a single report with visualizations and insights.

        Returns:
            dict: Dictionary containing all analysis results.
        """
        if self.output_dir is None:
            print("Output directory not specified. Cannot generate report.")
            return {}

        print("Generating comprehensive report...")

        # Run all analyses if not already done
        if not self.sequence_analysis:
            self.analyze_sequences()

        if not self.structure_analysis:
            self.analyze_structures()

        if not self.evolutionary_analysis and self.msa_data:
            self.analyze_evolutionary_information()

        # Combine all analyses
        comprehensive_report = {
            'sequence_analysis': self.sequence_analysis,
            'structure_analysis': self.structure_analysis,
            'evolutionary_analysis': self.evolutionary_analysis
        }

        # Generate model development insights
        insights = self._generate_model_insights()

        # Save comprehensive report
        report_dir = self.output_dir

        with open(report_dir / 'comprehensive_report.json', 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            report_json = json.dumps(comprehensive_report, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
            f.write(report_json)

        # Generate README with key findings
        self._generate_report_readme(report_dir, insights)

        print("Comprehensive report generated.")
        return comprehensive_report

    def _generate_model_insights(self):
        """Generate insights for model development based on analysis results."""
        insights = {
            'key_findings': [],
            'modeling_recommendations': [],
            'data_characteristics': {}
        }

        # Extract key sequence characteristics
        if self.sequence_analysis:
            length_stats = self.sequence_analysis.get('length_stats', {})
            insights['data_characteristics']['sequence_length'] = {
                'min': length_stats.get('min_length'),
                'max': length_stats.get('max_length'),
                'mean': length_stats.get('mean_length'),
                'median': length_stats.get('median_length')
            }

            # Add key findings based on sequence analysis
            if length_stats.get('max_length', 0) > 1000:
                insights['key_findings'].append(
                    "Dataset contains very long sequences (>1000 nt), requiring special handling."
                )

            if length_stats.get('std_length', 0) > 100:
                insights['key_findings'].append(
                    "High variability in sequence lengths suggests need for adaptive architectures."
                )

            # Add modeling recommendations
            insights['modeling_recommendations'].append(
                "Use hierarchical architecture to handle variable sequence lengths."
            )
            insights['modeling_recommendations'].append(
                "Implement sequence embedding that captures nucleotide context."
            )

        # Extract key structure characteristics
        if self.structure_analysis:
            overall_stats = self.structure_analysis.get('overall_stats', {})
            insights['data_characteristics']['structure'] = {
                'mean_radius_of_gyration': overall_stats.get('mean_radius_of_gyration'),
                'mean_compactness': overall_stats.get('mean_compactness'),
                'rg_length_correlation': overall_stats.get('rg_length_correlation')
            }

            # Add key findings based on structure analysis
            if overall_stats.get('rg_length_correlation', 0) > 0.8:
                insights['key_findings'].append(
                    "Strong correlation between sequence length and radius of gyration suggests scaling relationship."
                )

            # Add modeling recommendations
            insights['modeling_recommendations'].append(
                "Incorporate physics-based constraints on nucleotide distances."
            )
            insights['modeling_recommendations'].append(
                "Use equivariant neural networks to preserve 3D geometric relationships."
            )

        # Add general modeling recommendations
        insights['modeling_recommendations'].extend([
            "Implement multi-scale representation of RNA structures.",
            "Integrate evolutionary information through MSA features.",
            "Use uncertainty quantification to generate ensemble predictions."
        ])

        return insights

    def _generate_report_readme(self, report_dir, insights):
        """Generate a README file with key findings and insights."""
        readme_content = """# RNA 3D Structure Analysis Report

## Overview
This report contains comprehensive analysis of RNA sequences and structures,
designed to inform the development of groundbreaking RNA 3D structure prediction models.

## Key Findings
"""

        # Add key findings
        for finding in insights['key_findings']:
            readme_content += f"- {finding}\n"

        readme_content += """
## Modeling Recommendations
"""

        # Add modeling recommendations
        for recommendation in insights['modeling_recommendations']:
            readme_content += f"- {recommendation}\n"

        readme_content += """
## Directory Structure
- `sequence_analysis/`: Analysis of RNA sequences
- `structure_analysis/`: Analysis of RNA 3D structures
- `comprehensive_report.json`: Complete analysis results in JSON format

## Next Steps
1. Implement baseline models using the recommended architectures
2. Analyze MSA data to incorporate evolutionary information
3. Develop physics-informed neural network layers
4. Create multi-scale representation of RNA structures
"""

        with open(report_dir / 'README.md', 'w') as f:
            f.write(readme_content)

def main():
    """Main function to demonstrate the RNA analysis module."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="RNA 3D Structure Analysis")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing RNA data")
    parser.add_argument("--output-dir", default="data/analysis/core", help="Directory to save analysis results")
    args = parser.parse_args()

    # Create RNA analysis object
    analyzer = RNAAnalysis(args.data_dir, args.output_dir)

    # Run comprehensive analysis
    analyzer.generate_comprehensive_report()

if __name__ == "__main__":
    main()
