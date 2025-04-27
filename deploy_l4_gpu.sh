#!/bin/bash
# Script to deploy and run the RNA 3D structure prediction pipeline on an L4 GPU system

# Print header
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║            RNA 3D Structure Prediction L4 Deployment            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

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
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q -e .
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
kaggle competitions download -c stanford-rna-3d-folding -p data/raw
unzip -q -o data/raw/stanford-rna-3d-folding.zip -d data/raw
echo "✓ Competition data downloaded"

# Clean up the codebase
echo "▶ Cleaning up the codebase..."
./run_rna_pipeline.sh clean
echo "✓ Codebase cleaned"

# Create optimized model
echo "▶ Creating optimized model for L4 GPU..."
python -m rna_folding.models.optimize --config config/l4_gpu_config.json --output models/l4_gpu/optimized_model.pt
echo "✓ Optimized model created"

# Run a quick test to verify everything is working
echo "▶ Running a quick test..."
./run_rna_pipeline.sh test --device cuda
echo "✓ Test completed"

# Print deployment complete message
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     Deployment Complete                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "The RNA 3D structure prediction pipeline has been deployed successfully."
echo ""
echo "▶ To train a model with L4 GPU optimized settings:"
echo "  ./run_rna_pipeline.sh train --data-dir data/raw --output-dir models/l4_gpu --batch-size 16 --device cuda"
echo ""
echo "▶ To generate predictions with the trained model:"
echo "  ./run_rna_pipeline.sh predict --model-path models/l4_gpu/best_model.pt --output-file submissions/l4_gpu_submission.csv"
echo ""
echo "▶ To evaluate the model:"
echo "  ./run_rna_pipeline.sh evaluate --model-path models/l4_gpu/best_model.pt"
echo ""
echo "For more options, run:"
echo "  ./run_rna_pipeline.sh help"
