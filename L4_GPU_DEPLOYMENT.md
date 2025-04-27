# RNA 3D Structure Prediction - L4 GPU Deployment

This document provides instructions for deploying and running the RNA 3D structure prediction pipeline on a system with an NVIDIA L4 GPU (24GB VRAM), 50GB RAM, and 12 vCPUs.

## System Requirements

- NVIDIA L4 GPU with 24GB VRAM
- 50GB RAM
- 12 vCPUs
- Ubuntu 20.04 or later
- CUDA 11.8 or later
- Python 3.11

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stanford-rna-3d-solving.git
   cd stanford-rna-3d-solving
   ```

2. Place your Kaggle API credentials in the project root:
   ```bash
   # Download kaggle.json from https://www.kaggle.com/settings
   # and place it in the project root directory
   ```

3. Run the deployment script:
   ```bash
   ./deploy_l4_gpu.sh
   ```

4. Train a model:
   ```bash
   ./run_rna_pipeline.sh train --data-dir data/raw --output-dir models/l4_gpu --batch-size 16 --device cuda
   ```

5. Generate predictions:
   ```bash
   ./run_rna_pipeline.sh predict --model-path models/l4_gpu/best_model.pt --output-file submissions/l4_gpu_submission.csv
   ```

## Optimized Configuration

The L4 GPU configuration is optimized for:

- Mixed precision training (FP16/BF16)
- Gradient checkpointing for memory efficiency
- Optimal batch size for L4 GPU memory
- Multi-scale model architecture with attention mechanisms
- Physics-informed constraints for better structure prediction

## Directory Structure

```
stanford-rna-3d-solving/
├── config/                    # Configuration files
│   └── l4_gpu_config.json     # L4 GPU optimized configuration
├── data/                      # Data directory
│   ├── raw/                   # Raw data from Kaggle
│   └── processed/             # Processed data
├── models/                    # Saved models
│   └── l4_gpu/                # L4 GPU optimized models
├── src/                       # Source code
│   └── rna_folding/           # RNA folding package
├── submissions/               # Competition submissions
├── deploy_l4_gpu.sh           # L4 GPU deployment script
└── run_rna_pipeline.sh        # Unified pipeline script
```

## Pipeline Commands

### Training

```bash
./run_rna_pipeline.sh train [OPTIONS]
```

Options:
- `--data-dir DIR`: Directory containing the data
- `--output-dir DIR`: Directory to save model and results
- `--batch-size SIZE`: Batch size
- `--num-epochs NUM`: Number of epochs
- `--device DEVICE`: Device to train on (cuda or cpu)
- `--small`, `--medium`, `--large`: Predefined configurations

### Prediction

```bash
./run_rna_pipeline.sh predict [OPTIONS]
```

Options:
- `--model-path PATH`: Path to the model checkpoint
- `--sequences-file FILE`: Path to the sequences CSV file
- `--output-file FILE`: Path to save predictions to
- `--num-predictions NUM`: Number of predictions to generate

### Evaluation

```bash
./run_rna_pipeline.sh evaluate [OPTIONS]
```

Options:
- `--model-path PATH`: Path to the model checkpoint
- `--sequences-file FILE`: Path to the sequences CSV file
- `--structures-file FILE`: Path to the structures CSV file
- `--eval-output-file FILE`: Path to save evaluation results to

## Performance Optimization

The pipeline is optimized for the L4 GPU in several ways:

1. **Mixed Precision Training**: Uses FP16/BF16 precision to accelerate training and reduce memory usage.

2. **Gradient Checkpointing**: Trades computation for memory by recomputing intermediate activations during backpropagation.

3. **Optimized Batch Size**: Uses a batch size of 16 with gradient accumulation for optimal performance.

4. **Multi-Scale Architecture**: Processes RNA at multiple scales simultaneously for better structure prediction.

5. **Attention Mechanisms**: Uses self-attention to capture long-range dependencies in RNA sequences.

6. **Physics-Informed Constraints**: Incorporates physical constraints to improve structure prediction accuracy.

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce batch size:
   ```bash
   ./run_rna_pipeline.sh train --batch-size 8 --gradient-accumulation-steps 2 --memory-efficient
   ```

2. Enable memory-efficient mode:
   ```bash
   ./run_rna_pipeline.sh train --memory-efficient
   ```

3. Use a smaller model configuration:
   ```bash
   ./run_rna_pipeline.sh train --small
   ```

### CUDA Issues

If you encounter CUDA-related issues:

1. Check CUDA installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. Verify PyTorch CUDA compatibility:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Try reinstalling PyTorch with the correct CUDA version:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Contact

For questions or issues, please open an issue on the GitHub repository.
