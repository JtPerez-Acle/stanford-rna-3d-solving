"""
Main entry point for the RNA folding package.
"""

import argparse
import sys
from pathlib import Path

from rna_folding.data.download import download_competition_data, check_kaggle_api
from rna_folding.visualization.visualize import visualize_rna_sample
from rna_folding.data.analysis import analyze_dataset

import pandas as pd

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RNA 3D Folding Pipeline")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download data command
    download_parser = subparsers.add_parser("download", help="Download competition data")
    download_parser.add_argument(
        "--competition",
        default="stanford-rna-3d-folding",
        help="Kaggle competition name"
    )
    download_parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to save downloaded data"
    )

    # Visualize data command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize RNA structures")
    visualize_parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing the data"
    )
    visualize_parser.add_argument(
        "--output-dir",
        default="data/visualizations",
        help="Directory to save visualizations"
    )
    visualize_parser.add_argument(
        "--target-id",
        help="Specific target ID to visualize"
    )
    visualize_parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to visualize"
    )

    # Analyze data command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze RNA dataset")
    analyze_parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing the data"
    )
    analyze_parser.add_argument(
        "--output-dir",
        default="data/analysis",
        help="Directory to save analysis results"
    )

    return parser.parse_args()

def download_command(args):
    """Run the download command."""
    if not check_kaggle_api():
        print("Kaggle API is not properly configured. Please run init_project.py first.")
        return 1

    success = download_competition_data(args.competition, args.output_dir)
    return 0 if success else 1

def create_visualization_index(output_dir, visualized_ids):
    """
    Create an HTML index file for easy browsing of visualizations.

    Args:
        output_dir (Path): Directory where visualizations are saved.
        visualized_ids (list): List of target IDs that were visualized.
    """
    from datetime import datetime

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>RNA Structure Visualizations</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
        .structure-card {{ border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; padding: 15px; }}
        .structure-title {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }}
        .viz-container {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .viz-item {{ flex: 1; min-width: 300px; }}
        .viz-item img {{ max-width: 100%; border: 1px solid #eee; }}
        .viz-caption {{ font-size: 0.9em; color: #666; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RNA Structure Visualizations</h1>
        <p>Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Number of Structures: {len(visualized_ids)}</p>
    </div>
"""

    for target_id in visualized_ids:
        html_content += f"""
    <div class="structure-card">
        <div class="structure-title">{target_id}</div>
        <div class="viz-container">
            <div class="viz-item">
                <img src="{target_id}_sequence.png" alt="Sequence visualization">
                <div class="viz-caption">Sequence</div>
            </div>
            <div class="viz-item">
                <img src="{target_id}_structure_xy.png" alt="XY projection">
                <div class="viz-caption">XY Projection</div>
            </div>
            <div class="viz-item">
                <img src="{target_id}_structure_xz.png" alt="XZ projection">
                <div class="viz-caption">XZ Projection</div>
            </div>
            <div class="viz-item">
                <img src="{target_id}_structure_yz.png" alt="YZ projection">
                <div class="viz-caption">YZ Projection</div>
            </div>
        </div>
    </div>
"""

    html_content += """
</body>
</html>
"""

    with open(output_dir / 'index.html', 'w') as f:
        f.write(html_content)

def generate_model_insights(analysis_results, output_dir):
    """
    Generate insights specifically focused on model development.

    Args:
        analysis_results (dict): Results from the data analysis.
        output_dir (Path): Directory to save the insights.
    """
    from datetime import datetime

    # Extract key information from analysis results
    summary = analysis_results['summary']
    length_stats = summary.get('sequence_length_stats', {})
    nucleotide_comp = summary.get('nucleotide_composition', {})

    # Create a model-focused insights document
    insights_md = f"""# RNA 3D Structure Prediction: Model Development Insights

## Dataset Characteristics
- **Total Sequences**: {summary.get('num_sequences', 'N/A')}
- **Sequence Length Range**: {length_stats.get('min_length', 'N/A')} - {length_stats.get('max_length', 'N/A')} nucleotides
- **Mean Sequence Length**: {length_stats.get('mean_length', 'N/A'):.1f} nucleotides
- **Median Sequence Length**: {length_stats.get('median_length', 'N/A'):.1f} nucleotides

## Key Modeling Considerations

### 1. Sequence Length Variability
The dataset shows significant variability in sequence lengths (std: {length_stats.get('std_length', 'N/A'):.1f}).
Model architecture should handle this variability through:
- Padding and masking strategies
- Hierarchical approaches for very long sequences
- Potential sequence segmentation for extremely long RNAs

### 2. Nucleotide Distribution
"""

    # Add nucleotide distribution insights
    for nt, pct in nucleotide_comp.items():
        if pct > 0.1:  # Only include significant nucleotides
            insights_md += f"- **{nt}**: {pct:.1f}%\n"

    insights_md += """
This distribution should inform:
- Embedding strategies
- Data augmentation approaches
- Potential class weighting in loss functions

### 3. Recommended Model Architectures
Based on the dataset characteristics, consider:

1. **Transformer-based models**:
   - Attention mechanisms can capture long-range dependencies
   - Position encodings can handle variable sequence lengths
   - Pre-training on larger RNA datasets could improve performance

2. **Graph Neural Networks**:
   - Represent RNA as a graph with nucleotides as nodes
   - Edge features can encode distances and angles
   - Can naturally incorporate secondary structure information

3. **Hybrid approaches**:
   - Combine sequence models with 3D coordinate prediction
   - Multi-task learning for joint prediction of secondary and tertiary structure
   - Incorporate physics-based energy terms as regularization

### 4. Evaluation Strategy
- Use TM-score as primary metric
- Consider ensemble approaches (generate multiple predictions)
- Implement cross-validation with stratification by RNA family

### 5. Next Steps for Model Development
1. Implement baseline models using each architecture
2. Analyze performance on different RNA classes/lengths
3. Incorporate secondary structure predictions as features
4. Explore transfer learning from protein structure models
5. Develop custom loss functions that optimize for TM-score
"""

    # Save the insights document
    with open(output_dir / 'model_development_insights.md', 'w') as f:
        f.write(insights_md)

    # Create a more technical JSON file with model hyperparameter suggestions
    model_params = {
        "dataset_stats": {
            "num_sequences": summary.get('num_sequences'),
            "sequence_length_stats": length_stats,
            "nucleotide_composition": nucleotide_comp
        },
        "recommended_hyperparameters": {
            "transformer": {
                "embedding_dim": 256,
                "num_layers": 6,
                "num_heads": 8,
                "dropout": 0.1,
                "max_seq_length": min(1024, int(length_stats.get('max_length', 1000) * 1.2))
            },
            "gnn": {
                "node_features": 64,
                "edge_features": 32,
                "num_layers": 4,
                "aggregation": "mean"
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "epochs": 100,
                "early_stopping_patience": 10
            }
        },
        "data_preprocessing": {
            "normalization": "per_feature_standard",
            "augmentation_techniques": [
                "random_rotations",
                "nucleotide_masking",
                "coordinate_noise"
            ],
            "train_val_test_split": [0.8, 0.1, 0.1]
        }
    }

    import json
    with open(output_dir / 'model_hyperparameters.json', 'w') as f:
        json.dump(model_params, f, indent=2)

    # Create a quick-reference cheatsheet for model development
    cheatsheet = """# RNA 3D Structure Prediction Cheatsheet

## Key Libraries
- **PyTorch Geometric**: For graph neural networks
- **ESM**: For protein/RNA language models
- **BioPython**: For biological sequence processing
- **MDTraj**: For molecular dynamics analysis
- **PyMOL**: For visualization and analysis

## Data Preprocessing Pipeline
1. Parse RNA sequences
2. Generate multiple sequence alignments
3. Predict secondary structure
4. Create feature vectors
5. Normalize coordinates

## Model Training Workflow
1. Initialize model architecture
2. Define custom loss function (RMSD + angle terms)
3. Train with gradient accumulation
4. Validate with TM-score
5. Generate ensemble predictions

## Common Issues & Solutions
- **Overfitting**: Increase dropout, add regularization
- **Slow convergence**: Adjust learning rate schedule
- **Poor generalization**: Add more diverse training data
- **Memory issues**: Use gradient checkpointing, mixed precision

## Useful Commands
```bash
# Train model
python train.py --model transformer --batch_size 16

# Evaluate model
python evaluate.py --model_path models/best_model.pt --test_set validation

# Generate predictions
python predict.py --input sequences.csv --output predictions.csv
```
"""

    with open(output_dir / 'model_cheatsheet.md', 'w') as f:
        f.write(cheatsheet)

def generate_visualization_readme(output_dir, visualized_ids):
    """
    Generate a README file documenting the visualization outputs.

    Args:
        output_dir (Path): Directory where visualizations are saved.
        visualized_ids (list): List of target IDs that were visualized.
    """
    from datetime import datetime

    readme_content = f"""# RNA 3D Structure Visualizations

## Overview
- **Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Structures**: {len(visualized_ids)} RNA molecules visualized
- **Purpose**: Human-readable visualization of RNA 3D structures

## Quick Access
- Open `index.html` in a web browser for an interactive gallery

## Structure IDs
"""

    # Add a simple list of structure IDs
    for target_id in visualized_ids:
        readme_content += f"- {target_id}\n"

    readme_content += """
## File Naming Convention
- `{structure_id}_sequence.png`: Nucleotide sequence visualization
- `{structure_id}_structure_xy.png`: Top-view projection (XY plane)
- `{structure_id}_structure_xz.png`: Front-view projection (XZ plane)
- `{structure_id}_structure_yz.png`: Side-view projection (YZ plane)

## Color Legend
- **Green**: Adenine (A)
- **Blue**: Cytosine (C)
- **Orange**: Guanine (G)
- **Red**: Uracil (U/T)

## Notes for Model Development
- These visualizations are for human inspection only
- For model input, use the raw coordinate data from the CSV files
- Consider the 3D spatial relationships visible in these projections when designing attention mechanisms
"""

    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)

def visualize_command(args):
    """Run the visualize command."""
    data_dir = Path(args.data_dir)
    base_output_dir = Path(args.output_dir)

    # Check if data exists
    train_seq_path = data_dir / "train_sequences.csv"
    train_labels_path = data_dir / "train_labels.csv"

    if not train_seq_path.exists() or not train_labels_path.exists():
        print(f"Data not found in {data_dir}. Please download the data first.")
        return 1

    # Create a more intuitive output structure
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")

    # Create main output directory with date
    date_dir = base_output_dir / date_str
    date_dir.mkdir(parents=True, exist_ok=True)

    # Use the viz subdirectory for visualizations
    output_dir = date_dir / "viz"
    output_dir.mkdir(exist_ok=True)

    # Load data
    train_sequences = pd.read_csv(train_seq_path)
    train_labels = pd.read_csv(train_labels_path)

    # Filter by target_id if specified
    if args.target_id:
        sequences = train_sequences[train_sequences['target_id'] == args.target_id]
        if len(sequences) == 0:
            print(f"Target ID {args.target_id} not found in the data.")
            return 1
    else:
        # Take the first N samples
        sequences = train_sequences.head(args.num_samples)

    # Track which structures were visualized
    visualized_ids = []

    # Visualize each sample
    for _, seq in sequences.iterrows():
        target_id = seq['target_id']
        print(f"Visualizing {target_id}...")

        # Get corresponding labels
        labels = train_labels[train_labels['ID'].str.startswith(f"{target_id}_")]

        if len(labels) == 0:
            print(f"No labels found for {target_id}. Skipping.")
            continue

        # Create visualizations
        visualize_rna_sample(seq, labels, output_dir)
        visualized_ids.append(target_id)

    # Generate README file
    if visualized_ids:
        generate_visualization_readme(output_dir, visualized_ids)

    # Create a symlink to today's visualizations
    today_link = base_output_dir / "today"
    if today_link.exists():
        today_link.unlink()
    today_link.symlink_to(date_str)

    # Create an index.html file for easy browsing
    create_visualization_index(output_dir, visualized_ids)

    print(f"Visualizations saved to {output_dir}")
    print(f"Access today's results via the 'today' symlink: {today_link}")
    return 0

def analyze_command(args):
    """Run the analyze command."""
    data_dir = Path(args.data_dir)
    base_output_dir = Path(args.output_dir)

    # Check if data exists
    train_seq_path = data_dir / "train_sequences.csv"
    train_labels_path = data_dir / "train_labels.csv"

    if not train_seq_path.exists() or not train_labels_path.exists():
        print(f"Data not found in {data_dir}. Please download the data first.")
        return 1

    # Create a more intuitive output structure
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")

    # Create main output directory with date
    output_dir = base_output_dir / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different types of analyses
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(exist_ok=True)

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    insights_dir = output_dir / "insights"
    insights_dir.mkdir(exist_ok=True)

    # Run analysis with the stats directory
    result = analyze_dataset(data_dir, stats_dir)

    # Generate additional insights and summaries
    generate_model_insights(result, insights_dir)

    # Create a symlink to today's analysis
    today_link = base_output_dir / "today"
    if today_link.exists():
        today_link.unlink()
    today_link.symlink_to(date_str)

    print(f"Analysis results saved to {output_dir}")
    print(f"  - Statistics: {stats_dir}")
    print(f"  - Visualizations: {viz_dir}")
    print(f"  - Insights: {insights_dir}")
    print(f"Access today's analysis via the 'today' symlink: {today_link}")
    return 0

def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "download":
        return download_command(args)
    elif args.command == "visualize":
        return visualize_command(args)
    elif args.command == "analyze":
        return analyze_command(args)
    else:
        print("Please specify a command. Use --help for more information.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
