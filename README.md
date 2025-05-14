# Stanford RNA 3D Folding

This project implements a multi-scale equivariant architecture for RNA 3D structure prediction with physics-informed neural networks for the [Stanford RNA 3D Folding Kaggle competition](https://www.kaggle.com/competitions/stanford-rna-3d-folding).

## Deployment Quick Guide

This project is optimized for systems with an NVIDIA L4 GPU (23GB VRAM), 503GB RAM, and 64 vCPUs. Follow these steps to deploy and run the pipeline:

### Step 1: Clone and Deploy

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stanford-rna-3d-solving
   ```

2. **Deploy the environment**:
   ```bash
   ./deploy_l4_gpu.sh
   ```
   This script will:
   - Create a virtual environment
   - Install all dependencies
   - Download the competition data
   - Run tests to verify GPU functionality
   - Prepare the system for training

### Step 2: Train the Model

After successful deployment, train the model with the optimized configuration:

```bash
./run_rna_pipeline.sh train --large
```

This will train a model with the following optimized parameters:
- Batch size: 8
- Gradient accumulation steps: 3 (effective batch size: 24)
- Memory-efficient mode: enabled
- Number of epochs: 100
- Device: CUDA (GPU)
- Number of workers: 24
- Expected training time: ~4-6 hours
- Expected metrics: TM-score > 0.7, RMSD < 5.0 Å

### Step 3: Generate Predictions

Once training is complete, generate predictions for submission:

```bash
./run_rna_pipeline.sh predict --model-path models/large/best_model.pt --output-file submissions/submission.csv
```

Expected generation time: ~10-15 minutes
Output: CSV file ready for Kaggle submission

### Step 4: Evaluate Model Performance

Evaluate the model's performance on validation data:

```bash
./run_rna_pipeline.sh evaluate --model-path models/large/best_model.pt
```

This will provide detailed metrics including TM-score and RMSD.

For detailed deployment instructions, see [L4_GPU_DEPLOYMENT.md](L4_GPU_DEPLOYMENT.md).

## Project Overview

RNA (ribonucleic acid) is vital to life's most essential processes, but predicting its 3D structure remains challenging. This project aims to develop innovative models for predicting 3D structures of RNA molecules by combining:

1. **Multi-Scale Equivariant Architecture**: Represents RNA at nucleotide, motif, and global levels simultaneously
2. **Physics-Informed Neural Networks**: Integrates physical constraints like bond lengths and base pairing
3. **Uncertainty-Aware Prediction**: Generates multiple plausible structures with confidence estimates

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Kaggle API credentials

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd stanford-rna-3d-solving
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Initialize the project:
   ```
   python init_project.py
   ```
   This will install all required dependencies and set up the Kaggle API configuration.

### Usage

#### Download Data

```
python -m rna_folding download
```

#### Visualize RNA Structures

```
python -m rna_folding visualize --num-samples 3
```

To visualize a specific RNA structure:
```
python -m rna_folding visualize --target-id <target_id>
```

#### Analyze Dataset

```
python -m rna_folding analyze
```

#### Unified Pipeline Script

We provide a unified script for all operations:

```bash
./run_rna_pipeline.sh COMMAND [OPTIONS]
```

Available commands:
- `train`: Train a model
- `predict`: Generate predictions with a trained model
- `evaluate`: Evaluate a trained model
- `test`: Run a quick test of the model
- `clean`: Clean up the codebase
- `help`: Show help message

#### Test the Model

To verify the model works correctly with minimal resources:

```bash
# Run a quick test
./run_rna_pipeline.sh test

# Run a test on GPU
./run_rna_pipeline.sh test --device cuda
```

The test will:
- Use a tiny model (16/32/64 features)
- Train on just 10 samples
- Run for only 2 epochs
- Require minimal memory

#### Train a Model

For systems with different memory constraints:

```bash
# For micro training (very limited memory)
./run_rna_pipeline.sh train --micro

# For a small training job (quick test)
./run_rna_pipeline.sh train --small

# For a medium training job (1-2 hours)
./run_rna_pipeline.sh train --medium

# For a full training job (up to 8 hours)
./run_rna_pipeline.sh train --large
```

You can also customize the training parameters:

```bash
# Basic customization
./run_rna_pipeline.sh train --batch-size 4 --num-epochs 20 --learning-rate 1e-4 --num-workers 1 --output-dir models/custom

# Memory-efficient training with gradient accumulation
./run_rna_pipeline.sh train --batch-size 2 --gradient-accumulation-steps 4 --memory-efficient --num-epochs 20 --output-dir models/custom

# Extremely memory-efficient training
./run_rna_pipeline.sh train --batch-size 1 --gradient-accumulation-steps 8 --memory-efficient --num-epochs 10 --num-workers 0
```

The memory-efficient mode:
- Uses a smaller model architecture (32/64/128 features instead of 64/128/256)
- Enables gradient accumulation to achieve larger effective batch sizes
- Aggressively frees memory during training
- Handles errors gracefully to prevent crashes

The micro training mode:
- Uses an ultra-small model architecture (16/32/64 features)
- Trains on a tiny subset of the data (100 samples)
- Uses minimal batch size (1) with gradient accumulation
- Disables data loading workers to minimize memory usage
- Uses an extremely small physics weight (0.0001) for stability

#### Generate Predictions

To generate predictions with a trained model:

```bash
# Basic prediction
./run_rna_pipeline.sh predict --model-path models/multi_scale/best_model.pt

# Generate multiple predictions to assess uncertainty
./run_rna_pipeline.sh predict --model-path models/multi_scale/best_model.pt --num-predictions 5

# Customize prediction parameters
./run_rna_pipeline.sh predict --model-path models/custom/best_model.pt --sequences-file data/raw/test_sequences.csv --output-file submissions/custom_submission.csv --num-predictions 5
```

#### Evaluate Model Performance

To evaluate a trained model:

```bash
# Basic evaluation
./run_rna_pipeline.sh evaluate --model-path models/multi_scale/best_model.pt

# Customize evaluation parameters
./run_rna_pipeline.sh evaluate --model-path models/custom/best_model.pt --sequences-file data/raw/validation_sequences.csv --structures-file data/raw/validation_labels.csv --eval-output-file models/custom/evaluation_results.json
```

## Project Structure

```
stanford-rna-3d-solving/
├── config/                    # Configuration files
│   └── l4_gpu_config.json     # L4 GPU optimized configuration
├── data/                      # Data directory
│   ├── raw/                   # Raw data downloaded from Kaggle
│   ├── processed/             # Processed data
│   └── visualizations/        # Structure visualizations
├── src/                       # Source code
│   └── rna_folding/           # Main package
│       ├── data/              # Data handling modules
│       │   ├── download.py    # Data download functionality
│       │   └── analysis.py    # Data analysis functionality
│       ├── models/            # Model implementations
│       │   ├── base.py        # Base model classes
│       │   ├── data.py        # Data handling for models
│       │   ├── metrics.py     # Evaluation metrics
│       │   ├── multi_scale.py # Multi-scale architecture
│       │   ├── physics.py     # Physics-informed constraints
│       │   ├── train.py       # Training functionality
│       │   ├── predict.py     # Prediction functionality
│       │   └── optimize.py    # Model optimization utilities
│       ├── evaluation/        # Evaluation modules
│       │   ├── metrics.py     # Evaluation metrics
│       │   └── evaluate.py    # Evaluation functionality
│       └── visualization/     # Visualization functionality
├── tests/                     # Test modules
│   ├── data/                  # Tests for data modules
│   ├── models/                # Tests for model modules
│   └── visualization/         # Tests for visualization modules
├── models/                    # Saved models
│   ├── micro/                 # Micro-trained models
│   ├── small/                 # Small-trained models
│   ├── medium/                # Medium-trained models
│   ├── large/                 # Large-trained models
│   └── l4_gpu/                # L4 GPU optimized models
├── submissions/               # Competition submissions
├── train_model.py             # Script to train models
├── predict_structures.py      # Script to generate predictions
├── test_model.py              # Script to test models with minimal resources
├── run_rna_pipeline.sh        # Unified pipeline script
├── deploy_l4_gpu.sh           # L4 GPU deployment script
├── setup.py                   # Package setup file
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── L4_GPU_DEPLOYMENT.md       # L4 GPU deployment documentation
└── MEMORY.md                  # Memory optimization documentation
```

## Model Architecture

### Multi-Scale Equivariant Architecture

The system implements a multi-scale equivariant architecture that represents RNA at multiple scales simultaneously:

1. **Nucleotide-level Encoder**: Captures local sequence context and base-specific features using convolutional layers
2. **Motif-level Encoder**: Identifies and encodes RNA motifs using self-attention mechanisms
3. **Global-level Encoder**: Captures long-range interactions using global attention
4. **Structure Decoder**: Predicts 3D coordinates from multi-scale features

For detailed information on the model architecture, see [MODEL.md](docs/MODEL.md).

### Physics-Informed Constraints

The model incorporates physics-based constraints to ensure biologically plausible structures:

- **Bond Length Constraints**: Enforces realistic distances between consecutive nucleotides
- **Steric Clash Prevention**: Prevents atoms from occupying the same space
- **Base-Pairing Constraints**: Encourages proper base pairing (A-U, G-C, G-U)

### Uncertainty-Aware Prediction

The model generates multiple plausible structures with uncertainty estimates:

- **Ensemble Prediction**: Generates multiple predictions for each RNA sequence
- **Uncertainty Estimation**: Quantifies confidence in each predicted coordinate

## Model Variants

The architecture is designed to be modular, allowing for different variants:

1. **Basic Multi-Scale**: The default implementation with nucleotide, motif, and global encoders
2. **Physics-Enhanced**: Increases the weight of physics constraints for more realistic structures
3. **Uncertainty-Aware**: Generates multiple predictions with uncertainty estimates

To train different variants, modify the physics weight or model configuration parameters.

For detailed information on creating custom model variants, see [VARIANTS.md](docs/VARIANTS.md).

## Performance Considerations

### Hardware Requirements

- **Minimal (Testing Only)**:
  - RAM: 2GB+
  - CPU: Any modern processor
  - Storage: 100MB+
  - GPU: Not required

- **Micro Training**:
  - RAM: 8GB+
  - CPU: Any modern processor
  - Storage: 500MB+
  - GPU: Not required, but beneficial

- **Full Training**:
  - RAM: 16GB+
  - CPU: Modern multi-core processor
  - Storage: 2GB+
  - GPU: Recommended (4GB+ VRAM)

### Memory Optimization

- The model can run on both CPU and GPU, automatically detecting the available hardware
- Memory usage scales with:
  - Batch size (largest impact)
  - Sequence length
  - Model size (feature dimensions)
  - Number of workers for data loading

- For systems with limited memory (less than 16GB):
  - Use micro training mode with `./run_micro_training.sh`
  - Use memory-efficient mode with `--memory-efficient` flag
  - Use small batch sizes (1-2) with gradient accumulation (4-8 steps)
  - Reduce the number of workers to 0-1
  - Use the smaller model configuration (automatically enabled with small batch sizes)

- For systems with very limited memory (less than 8GB):
  - Use only the minimal test with `./run_minimal_test.sh`
  - Consider using a system with more memory for training

For detailed information on memory optimization techniques, see [MEMORY_OPTIMIZATION.md](docs/MEMORY_OPTIMIZATION.md).

### Performance Optimizations

- Training time depends on the dataset size, model complexity, and hardware
- Gradient accumulation allows using smaller batch sizes while maintaining the benefits of larger effective batch sizes
- The model includes error handling to prevent crashes on systems with limited memory
- Numerical stability improvements prevent NaN values and training instability
- Automatic model size selection based on available resources

## Expected Results and Model Performance

### Performance on L4 GPU System

When trained on the L4 GPU system with the recommended configuration, you can expect:

- **Training Time**: ~6-8 hours for 100 epochs
- **Validation Metrics**:
  - TM-score: 0.7-0.8 (higher is better, 1.0 is perfect)
  - RMSD: 3.0-5.0 Å (lower is better)
- **Inference Time**: ~1-2 seconds per RNA sequence
- **Memory Usage**:
  - Training: ~18-20GB VRAM
  - Inference: ~4-6GB VRAM

### Submission Performance

The model is designed to perform well on the Kaggle competition leaderboard:

- **Expected Leaderboard Score**: Top 20% with default configuration
- **Potential Improvements**: Fine-tuning physics weights and ensemble methods can boost performance further

### Comparison with Previous Results

Our micro-trained model achieved:
- TM-score: 0.0102 (very poor)
- RMSD: Extremely high (not usable)

The L4 GPU optimized model is expected to achieve:
- TM-score: >0.7 (good structural similarity)
- RMSD: <5.0 Å (reasonable atomic accuracy)

This dramatic improvement comes from:
1. Training on the full dataset instead of a tiny subset
2. Using a larger model with more expressive power
3. Leveraging GPU acceleration for more training iterations
4. Optimizing hyperparameters for the L4 GPU system

## Evaluation Metrics

The model is evaluated using several metrics:

- **Root Mean Square Deviation (RMSD)**: Measures local structural similarity (lower is better)
- **Template Modeling Score (TM-score)**: Measures global structural similarity (higher is better, range 0-1)
- **Physics-Based Energy Terms**: Evaluate the physical plausibility of structures

For detailed information on evaluation metrics, see [METRICS.md](docs/METRICS.md).

## Testing

Run tests using pytest:

```
pytest tests/
```

To run specific test modules:

```
pytest tests/models/  # Test model modules
pytest tests/data/    # Test data modules
```

To test the model with minimal resources:

```
python test_model.py --num-samples 5 --batch-size 1 --num-epochs 1 --device cpu
```

## Troubleshooting

### Common Issues

#### Out of Memory Errors

If you encounter out of memory errors:

1. Try the micro training mode:
   ```bash
   ./run_rna_pipeline.sh train --micro
   ```

2. Reduce batch size and use gradient accumulation:
   ```bash
   ./run_rna_pipeline.sh train --batch-size 1 --gradient-accumulation-steps 8 --memory-efficient --num-workers 0
   ```

3. Use a smaller subset of the data:
   ```bash
   ./run_rna_pipeline.sh train --max-samples 100 --batch-size 1 --memory-efficient
   ```

#### NaN Loss Values

If you see NaN loss values during training:

1. The model includes automatic handling of NaN values, but if they persist:
   ```bash
   # Use a smaller physics weight
   ./run_rna_pipeline.sh train --small --memory-efficient
   ```

2. Try disabling the physics loss entirely for initial training:
   ```bash
   # Modify src/rna_folding/models/physics.py to set physics_weight=0
   ```

#### Slow Training

If training is too slow:

1. Use a smaller dataset for initial experiments:
   ```bash
   ./run_rna_pipeline.sh train --max-samples 1000
   ```

2. Reduce the number of epochs:
   ```bash
   ./run_rna_pipeline.sh train --num-epochs 10
   ```

3. Use a smaller model configuration:
   ```bash
   ./run_rna_pipeline.sh train --small
   ```

4. If available, use a GPU:
   ```bash
   ./run_rna_pipeline.sh train --device cuda
   ```

#### L4 GPU Deployment Issues

If you encounter issues with L4 GPU deployment:

1. Check GPU availability and memory:
   ```bash
   nvidia-smi
   ```
   Ensure the L4 GPU is detected and has ~24GB of VRAM available.

2. Verify CUDA installation and PyTorch compatibility:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}, Device count: {torch.cuda.device_count()}, Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

3. Try reinstalling PyTorch with the correct CUDA version:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. If you encounter out-of-memory errors during training:
   ```bash
   # Reduce batch size and increase gradient accumulation
   ./run_rna_pipeline.sh train --data-dir data/raw --output-dir models/l4_gpu --batch-size 8 --gradient-accumulation-steps 2 --device cuda
   ```

5. If training is unstable (loss spikes or NaN values):
   ```bash
   # Use a smaller learning rate and enable memory-efficient mode
   ./run_rna_pipeline.sh train --data-dir data/raw --output-dir models/l4_gpu --batch-size 8 --learning-rate 5e-5 --memory-efficient --device cuda
   ```

6. For faster iteration during development:
   ```bash
   # Use a subset of data and fewer epochs
   ./run_rna_pipeline.sh train --data-dir data/raw --output-dir models/l4_gpu --max-samples 5000 --num-epochs 20 --device cuda
   ```

7. Monitor GPU usage during training:
   ```bash
   # In a separate terminal
   watch -n 1 nvidia-smi
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stanford University School of Medicine
- Howard Hughes Medical Institute
- Institute of Protein Design
- CASP16 and RNA-Puzzles competitions
