# Stanford RNA 3D Folding Project Memory

## Project Overview

This project focuses on developing groundbreaking models for predicting the 3D structures of RNA molecules for the Stanford RNA 3D Folding Kaggle competition. The goal is to create innovative approaches that leverage multi-scale representation, physics-informed neural networks, and evolutionary information.

## Technical Setup

- **Python Version**: 3.11 in a virtual environment
- **Kaggle API**: Configuration file (`kaggle.json`) placed at the project root
- **Key Dependencies**: numpy, pandas, matplotlib, seaborn, biopython, pytest

## Project Structure

```
stanford-rna-3d-solving/
├── data/                      # Data directory
│   ├── raw/                   # Raw data downloaded from Kaggle
│   │   ├── MSA/               # Multiple Sequence Alignment data
│   │   ├── train_sequences.csv
│   │   ├── train_labels.csv
│   │   └── ...
│   ├── analysis/              # Analysis outputs
│   │   └── YYYYMMDD/          # Date-based organization
│   │       ├── core/          # Core analysis results
│   │       ├── stats/         # Statistical analysis
│   │       ├── viz/           # Visualizations
│   │       └── insights/      # Model development insights
│   └── visualizations/        # Structure visualizations
│       └── YYYYMMDD/          # Date-based organization
│           └── viz/           # Visualization files
├── src/                       # Source code
│   └── rna_folding/           # Main package
│       ├── data/              # Data handling modules
│       │   ├── download.py    # Data download functionality
│       │   └── analysis.py    # Data analysis functionality
│       ├── analysis/          # Advanced analysis modules
│       │   └── core_analysis.py # Comprehensive RNA analysis
│       ├── visualization/     # Visualization modules
│       │   └── visualize.py   # Visualization functionality
│       └── models/            # Model implementations (to be added)
├── notebooks/                 # Jupyter notebooks
│   └── RNA_Structure_Analysis.ipynb # Main analysis notebook
├── tests/                     # Test files
├── init_project.py            # Project initialization script
├── run_pipeline.py            # Script to run the entire pipeline
├── run_core_analysis.py       # Script to run core analysis
└── README.md                  # Project documentation
```

## Key Capabilities

### 1. Data Management
- **Download**: `python -m rna_folding download` - Downloads competition data from Kaggle
- **Initialization**: `python init_project.py` - Sets up the project environment

### 2. Analysis
- **Core Analysis**: `python run_core_analysis.py` - Performs comprehensive analysis of RNA sequences and structures
- **Pipeline**: `python run_pipeline.py --analyze` - Runs the full analysis pipeline with date-based organization

### 3. Visualization
- **Structure Visualization**: `python run_pipeline.py --visualize --num-samples N` - Visualizes N RNA structures
- **Specific Structure**: `python run_pipeline.py --visualize --target-id TARGET_ID` - Visualizes a specific RNA structure

### 4. Jupyter Notebook
- The `notebooks/RNA_Structure_Analysis.ipynb` notebook provides an interactive environment for analyzing RNA data and developing models

## Groundbreaking Model Approach

Our approach to RNA 3D structure prediction is based on several innovative components:

1. **Multi-Scale Equivariant Architecture**
   - Represents RNA at multiple scales simultaneously (nucleotide, motif, global)
   - Preserves geometric relationships through equivariant transformations

2. **Physics-Informed Neural Networks**
   - Integrates physical constraints directly into the neural network
   - Enforces hydrogen bonding and other physical constraints

3. **Evolutionary + Synthetic Data Fusion**
   - Leverages both natural evolutionary data from MSAs and synthetic data
   - Addresses data limitations in RNA structure prediction

4. **Uncertainty-Aware Ensemble Prediction**
   - Generates multiple plausible structures
   - Quantifies uncertainty in different regions

## Analysis Outputs

The analysis pipeline generates several key outputs:

1. **Sequence Analysis**
   - Length distribution statistics
   - Nucleotide composition
   - Sequence entropy
   - Dinucleotide frequencies

2. **Structure Analysis**
   - Radius of gyration statistics
   - Compactness measures
   - End-to-end distances
   - Consecutive nucleotide distances

3. **Model Development Insights**
   - Key findings about the dataset
   - Modeling recommendations
   - Architecture suggestions

## Visualization Outputs

The visualization pipeline generates:

1. **Sequence Visualizations**
   - Colored representation of RNA sequences

2. **Structure Projections**
   - 2D projections of 3D structures (XY, XZ, YZ planes)

3. **Interactive Gallery**
   - HTML-based gallery for browsing visualizations

## Running Tests

- **All Tests**: `python run_tests.py --all`
- **Package Tests**: `python run_tests.py --package`
- **Download Tests**: `python run_tests.py --download`
- **Visualization Tests**: `python run_tests.py --visualization`
- **Analysis Tests**: `python run_tests.py --analysis`

## Development Workflow

1. **Data Exploration**: Use the core analysis module to understand the dataset
2. **Feature Engineering**: Extract relevant features from sequences and structures
3. **Model Development**: Implement the multi-scale equivariant architecture
4. **Training and Evaluation**: Train models and evaluate using TM-score
5. **Ensemble Generation**: Create ensemble predictions with uncertainty quantification

## User Preferences

- Focus on core functionality before creating presentation notebooks
- Prefer logical implementation steps that can later be abstracted
- Organize output files with clear naming conventions
- Design pipeline for extremely quick iteration in model training and result analysis
- Support professional and transparent result presentation

## Next Development Steps

1. Enhance structure analysis with base-pairing and motif identification
2. Implement MSA analysis to extract evolutionary features
3. Develop the multi-scale equivariant network architecture
4. Create physics-informed neural network layers
5. Implement uncertainty-aware prediction mechanisms

## L4 GPU Deployment

The project has been optimized for deployment on systems with NVIDIA L4 GPU (24GB VRAM), 50GB RAM, and 12 vCPUs:

1. **Unified Pipeline Script**: `./run_rna_pipeline.sh` - A single script for all operations
2. **L4 GPU Deployment**: `./deploy_l4_gpu.sh` - Automated deployment script for L4 GPU systems
3. **Optimized Configuration**: `config/l4_gpu_config.json` - Configuration optimized for L4 GPU

### Deployment Commands

- **Deploy**: `./deploy_l4_gpu.sh` - Set up the environment and download data
- **Train**: `./run_rna_pipeline.sh train --device cuda` - Train a model on GPU
- **Predict**: `./run_rna_pipeline.sh predict --model-path models/l4_gpu/best_model.pt` - Generate predictions
- **Evaluate**: `./run_rna_pipeline.sh evaluate --model-path models/l4_gpu/best_model.pt` - Evaluate model performance

### Optimization Techniques

1. **Mixed Precision Training**: Uses FP16/BF16 precision to accelerate training
2. **Gradient Checkpointing**: Trades computation for memory by recomputing intermediate activations
3. **Optimized Batch Size**: Uses a batch size of 16 with gradient accumulation
4. **Multi-Scale Architecture**: Processes RNA at multiple scales simultaneously
5. **Attention Mechanisms**: Uses self-attention to capture long-range dependencies
6. **Physics-Informed Constraints**: Incorporates physical constraints for better prediction

For detailed instructions, see the `L4_GPU_DEPLOYMENT.md` file.
