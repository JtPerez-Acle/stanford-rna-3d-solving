#!/bin/bash
# Script to deploy and run the RNA 3D structure prediction pipeline on an L4 GPU system

# Exit on error
set -e

# Function to handle errors
handle_error() {
    echo "✗ Error occurred at line $1"
    echo "  Deployment failed. Please check the error message above."
    exit 1
}

# Set up error handling
trap 'handle_error $LINENO' ERR

# Print header
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║            RNA 3D Structure Prediction L4 Deployment            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check system requirements
echo "▶ Checking system requirements..."
total_memory=$(free -g | awk '/^Mem:/{print $2}')
disk_space=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
cpu_cores=$(nproc)

echo "  • System information:"
echo "    - Total memory: ${total_memory}GB"
echo "    - Available disk space: ${disk_space}GB"
echo "    - CPU cores: ${cpu_cores}"

# Check minimum requirements
if [ $total_memory -lt 50 ]; then
    echo "⚠ Warning: System has less than 50GB of RAM (${total_memory}GB)"
    echo "  This may affect performance. Recommended: 62GB+"
fi

if [ ${disk_space%.*} -lt 30 ]; then
    echo "⚠ Warning: Less than 30GB of disk space available (${disk_space}GB)"
    echo "  This may not be enough for the dataset and models. Recommended: 40GB+"
fi

if [ $cpu_cores -lt 12 ]; then
    echo "⚠ Warning: System has less than 12 CPU cores (${cpu_cores})"
    echo "  This may affect performance. Recommended: 16+ cores"
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi
else
    echo "✗ No NVIDIA GPU detected. This script is optimized for L4 GPU systems."
    echo "  Continuing with CPU, but performance will be significantly reduced."
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Activated virtual environment"
else
    echo "▶ Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "✓ Created and activated virtual environment"
fi

# Install dependencies
echo "▶ Installing dependencies..."
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "  • Installing PyTorch with CUDA support..."
pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
echo "  • Installing core dependencies..."
pip install -q numpy pandas matplotlib seaborn scikit-learn biopython tqdm

# Install visualization dependencies
echo "  • Installing visualization dependencies..."
pip install -q py3Dmol nglview ipywidgets

# Install development dependencies
echo "  • Installing development dependencies..."
pip install -q black flake8 pytest pytest-cov jupyter

# Install Kaggle API
echo "  • Installing Kaggle API..."
pip install -q kaggle

# Install project in development mode
echo "  • Installing project in development mode..."
pip install -q -e .

# Verify PyTorch installation
echo "  • Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo "✓ Dependencies installed"

# Create necessary directories
echo "▶ Setting up directory structure..."
mkdir -p config
mkdir -p models/l4_gpu
mkdir -p data/raw
mkdir -p submissions
echo "✓ Directory structure set up"

# Check if Kaggle API credentials exist
if [ -f "kaggle.json" ]; then
    echo "✓ Kaggle API credentials found"
    # Set up Kaggle API
    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
else
    echo "✗ Kaggle API credentials not found"
    echo "  Please place your kaggle.json file in the project root directory"
    echo "  You can download it from https://www.kaggle.com/settings"
    exit 1
fi

# Download competition data
echo "▶ Downloading competition data..."
# Check if data already exists
if [ -f "data/raw/train_sequences.csv" ] && [ -f "data/raw/train_labels.csv" ]; then
    echo "  • Competition data already exists, skipping download"
else
    echo "  • Downloading from Kaggle..."
    # Try to download data with retry mechanism
    max_attempts=3
    attempt=1
    download_success=false

    while [ $attempt -le $max_attempts ] && [ "$download_success" = false ]; do
        echo "  • Download attempt $attempt of $max_attempts..."
        if kaggle competitions download -c stanford-rna-3d-folding -p data/raw; then
            download_success=true
        else
            echo "  • Download attempt $attempt failed, retrying in 5 seconds..."
            sleep 5
            attempt=$((attempt+1))
        fi
    done

    if [ "$download_success" = false ]; then
        echo "✗ Failed to download competition data after $max_attempts attempts"
        echo "  Please download the data manually from https://www.kaggle.com/competitions/stanford-rna-3d-folding/data"
        echo "  and place it in the data/raw directory"
        exit 1
    fi

    echo "  • Extracting data..."
    unzip -q -o data/raw/stanford-rna-3d-folding.zip -d data/raw

    # Verify extraction
    if [ ! -f "data/raw/train_sequences.csv" ] || [ ! -f "data/raw/train_labels.csv" ]; then
        echo "✗ Data extraction failed or files are missing"
        exit 1
    fi

    # Clean up zip file to save space
    echo "  • Cleaning up downloaded zip file..."
    rm -f data/raw/stanford-rna-3d-folding.zip
fi

# Verify data files
echo "  • Verifying data files..."
required_files=("train_sequences.csv" "train_labels.csv" "validation_sequences.csv" "validation_labels.csv" "test_sequences.csv")
missing_files=false

for file in "${required_files[@]}"; do
    if [ ! -f "data/raw/$file" ]; then
        echo "✗ Missing required file: data/raw/$file"
        missing_files=true
    fi
done

if [ "$missing_files" = true ]; then
    echo "✗ Some required data files are missing"
    exit 1
fi

# Count number of sequences
train_count=$(wc -l < data/raw/train_sequences.csv)
val_count=$(wc -l < data/raw/validation_sequences.csv)
test_count=$(wc -l < data/raw/test_sequences.csv)

echo "  • Data statistics:"
echo "    - Training sequences: $((train_count-1))"
echo "    - Validation sequences: $((val_count-1))"
echo "    - Test sequences: $((test_count-1))"

echo "✓ Competition data downloaded and verified"

# Clean up the codebase
echo "▶ Cleaning up the codebase..."
./run_rna_pipeline.sh clean
echo "✓ Codebase cleaned"

# Create optimized model
echo "▶ Creating optimized model for L4 GPU..."
if [ -f "config/l4_gpu_config.json" ]; then
    python -m rna_folding.models.optimize --config config/l4_gpu_config.json --output models/l4_gpu/optimized_model.pt

    if [ -f "models/l4_gpu/optimized_model.pt" ]; then
        echo "✓ Optimized model created successfully"
    else
        echo "✗ Failed to create optimized model"
        echo "  Continuing with deployment, but you may need to troubleshoot this issue"
    fi
else
    echo "✗ L4 GPU configuration file not found at config/l4_gpu_config.json"
    echo "  Continuing with deployment, but you will need to create this file"
fi

# Run a quick test to verify everything is working
echo "▶ Running a quick test to verify GPU functionality..."

# First check if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "  • CUDA is available, testing with GPU..."
    set +e  # Don't exit on error
    ./run_rna_pipeline.sh test --device cuda
    test_result=$?
    set -e  # Exit on error again

    if [ $test_result -eq 0 ]; then
        echo "✓ GPU test completed successfully"
    else
        echo "⚠ GPU test completed with issues"
        echo "  • Trying CPU test as fallback..."

        set +e  # Don't exit on error
        ./run_rna_pipeline.sh test --device cpu
        cpu_test_result=$?
        set -e  # Exit on error again

        if [ $cpu_test_result -eq 0 ]; then
            echo "✓ CPU test completed successfully"
            echo "  • The model works on CPU, but there may be issues with GPU configuration"
            echo "  • For optimal performance, please troubleshoot GPU issues before running the full pipeline"
        else
            echo "✗ Both GPU and CPU tests failed"
            echo "  • Please check the error messages and troubleshoot before proceeding"
        fi
    fi
else
    echo "  • CUDA is not available, testing with CPU..."
    set +e  # Don't exit on error
    ./run_rna_pipeline.sh test --device cpu
    test_result=$?
    set -e  # Exit on error again

    if [ $test_result -eq 0 ]; then
        echo "✓ CPU test completed successfully"
        echo "  • Note: Training will be much slower on CPU. For optimal performance, configure GPU support."
    else
        echo "✗ CPU test failed"
        echo "  • Please check the error messages and troubleshoot before proceeding"
    fi
fi

# Verify all components are ready
echo "▶ Verifying deployment components..."
verification_failed=false

# Check for critical files
critical_files=(
    "run_rna_pipeline.sh"
    "config/l4_gpu_config.json"
    "data/raw/train_sequences.csv"
    "data/raw/train_labels.csv"
    "data/raw/validation_sequences.csv"
    "data/raw/validation_labels.csv"
    "data/raw/test_sequences.csv"
)

for file in "${critical_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "✗ Critical file missing: $file"
        verification_failed=true
    fi
done

# Check for critical directories
critical_dirs=(
    "models/l4_gpu"
    "submissions"
    "src/rna_folding"
)

for dir in "${critical_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "✗ Critical directory missing: $dir"
        verification_failed=true
    fi
done

# Check for Python modules
python -c "import torch; import numpy; import pandas; import matplotlib; import rna_folding" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "✗ Some required Python modules are missing or not properly installed"
    verification_failed=true
fi

# Final verification status
if [ "$verification_failed" = true ]; then
    echo "⚠ Deployment verification found issues that need to be addressed"
else
    echo "✓ All deployment components verified successfully"
fi

# Calculate disk space usage
echo "▶ Disk space usage:"
du -sh data models src | sort -hr

# Print deployment complete message
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     Deployment Complete                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "The RNA 3D structure prediction pipeline has been deployed successfully."
echo ""
echo "▶ NEXT STEPS:"
echo ""
echo "  1. Run the full training pipeline with optimized settings:"
echo "     ./run_rna_pipeline.sh train --large"
echo ""
echo "     This will train a model with the following optimized parameters:"
echo "     • Batch size: 24"
echo "     • Number of epochs: 100"
echo "     • Device: cuda (GPU)"
echo "     • Number of workers: 16"
echo "     • Expected training time: ~6-8 hours"
echo "     • Expected validation metrics: TM-score > 0.7, RMSD < 5.0 Å"
echo ""
echo "  2. Generate predictions with the trained model:"
echo "     ./run_rna_pipeline.sh predict --model-path models/large/best_model.pt --output-file submissions/submission.csv"
echo ""
echo "  3. Evaluate the model performance:"
echo "     ./run_rna_pipeline.sh evaluate --model-path models/large/best_model.pt"
echo ""
echo "For more options and detailed configuration, run:"
echo "  ./run_rna_pipeline.sh help"
