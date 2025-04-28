#!/bin/bash
# Unified RNA 3D Structure Prediction Pipeline
# This script combines all functionality for training, prediction, and evaluation
# Optimized for systems with L4 GPU (23GB VRAM), 62GB RAM, and 16 vCPUs

# Default parameters
COMMAND="help"
MODEL_PATH="models/multi_scale/best_model.pt"
DATA_DIR="data/raw"
OUTPUT_DIR="models/multi_scale"
BATCH_SIZE=24
NUM_EPOCHS=100
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.00001
DEVICE="cuda"
NUM_WORKERS=16
GRADIENT_ACCUMULATION_STEPS=1
MEMORY_EFFICIENT=false
NUM_PREDICTIONS=5
SEQUENCES_FILE="data/raw/test_sequences.csv"
OUTPUT_FILE="submission.csv"
STRUCTURES_FILE="data/raw/validation_labels.csv"
EVAL_OUTPUT_FILE="models/evaluation_results.json"
MAX_SAMPLES=0  # 0 means use all samples

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        train|predict|evaluate|test|clean|help)
            COMMAND="$1"
            shift
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift
            shift
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift
            shift
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift
            shift
            ;;
        --gradient-accumulation-steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift
            shift
            ;;
        --memory-efficient)
            MEMORY_EFFICIENT=true
            shift
            ;;
        --num-predictions)
            NUM_PREDICTIONS="$2"
            shift
            shift
            ;;
        --sequences-file)
            SEQUENCES_FILE="$2"
            shift
            shift
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift
            shift
            ;;
        --structures-file)
            STRUCTURES_FILE="$2"
            shift
            shift
            ;;
        --eval-output-file)
            EVAL_OUTPUT_FILE="$2"
            shift
            shift
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift
            shift
            ;;
        --small)
            # Small training configuration
            BATCH_SIZE=4
            GRADIENT_ACCUMULATION_STEPS=4
            MEMORY_EFFICIENT=true
            NUM_EPOCHS=20
            OUTPUT_DIR="models/small"
            shift
            ;;
        --medium)
            # Medium training configuration
            BATCH_SIZE=8
            GRADIENT_ACCUMULATION_STEPS=2
            MEMORY_EFFICIENT=true
            NUM_EPOCHS=50
            OUTPUT_DIR="models/medium"
            shift
            ;;
        --large)
            # Large training configuration
            BATCH_SIZE=24
            GRADIENT_ACCUMULATION_STEPS=1
            NUM_EPOCHS=100
            OUTPUT_DIR="models/large"
            shift
            ;;
        --micro)
            # Micro training configuration for testing
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION_STEPS=4
            MEMORY_EFFICIENT=true
            NUM_EPOCHS=5
            OUTPUT_DIR="models/micro"
            MAX_SAMPLES=100
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
fi

# Function to print header
print_header() {
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║            RNA 3D Structure Prediction Pipeline                 ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
}

# Function to clean up the codebase
clean_codebase() {
    echo "▶ Cleaning up the codebase..."

    # Create necessary directories if they don't exist
    echo "  • Creating necessary directories..."
    mkdir -p models/multi_scale
    mkdir -p models/test
    mkdir -p models/small
    mkdir -p models/medium
    mkdir -p models/large
    mkdir -p models/micro
    mkdir -p data/processed
    mkdir -p data/visualizations
    mkdir -p submissions

    # Remove any unnecessary files
    echo "  • Removing unnecessary files..."
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} +
    find . -name "*.egg-info" -type d -exec rm -rf {} +
    find . -name ".DS_Store" -delete

    # Remove any temporary files
    echo "  • Removing temporary files..."
    rm -f *.log
    rm -f *.tmp

    # Make sure all scripts are executable
    echo "  • Making scripts executable..."
    chmod +x run_rna_pipeline.sh

    echo "✓ Codebase cleaned and organized successfully!"
}

# Function to run training
run_training() {
    echo "▶ Running training with the following parameters:"
    echo "  • Data directory: $DATA_DIR"
    echo "  • Output directory: $OUTPUT_DIR"
    echo "  • Batch size: $BATCH_SIZE"
    echo "  • Number of epochs: $NUM_EPOCHS"
    echo "  • Learning rate: $LEARNING_RATE"
    echo "  • Weight decay: $WEIGHT_DECAY"
    echo "  • Device: $DEVICE"
    echo "  • Number of workers: $NUM_WORKERS"
    echo "  • Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
    echo "  • Memory efficient: $MEMORY_EFFICIENT"
    if [ "$MAX_SAMPLES" -gt 0 ]; then
        echo "  • Maximum samples: $MAX_SAMPLES"
    fi
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Create subset data if MAX_SAMPLES is specified
    if [ "$MAX_SAMPLES" -gt 0 ]; then
        SUBSET_DIR="data/subset"
        mkdir -p "$SUBSET_DIR"

        echo "▶ Creating a subset of the data with $MAX_SAMPLES samples..."

        # Create subset of training data
        head -n 1 "$DATA_DIR/train_sequences.csv" > "$SUBSET_DIR/train_sequences.csv"
        head -n $(($MAX_SAMPLES + 1)) "$DATA_DIR/train_sequences.csv" | tail -n $MAX_SAMPLES >> "$SUBSET_DIR/train_sequences.csv"

        head -n 1 "$DATA_DIR/train_labels.csv" > "$SUBSET_DIR/train_labels.csv"
        grep -f <(cut -d, -f1 "$SUBSET_DIR/train_sequences.csv" | tail -n +2) "$DATA_DIR/train_labels.csv" >> "$SUBSET_DIR/train_labels.csv"

        # Create subset of validation data
        head -n 1 "$DATA_DIR/validation_sequences.csv" > "$SUBSET_DIR/validation_sequences.csv"
        head -n 21 "$DATA_DIR/validation_sequences.csv" | tail -n 20 >> "$SUBSET_DIR/validation_sequences.csv"

        head -n 1 "$DATA_DIR/validation_labels.csv" > "$SUBSET_DIR/validation_labels.csv"
        grep -f <(cut -d, -f1 "$SUBSET_DIR/validation_sequences.csv" | tail -n +2 | sed 's/_[0-9]*$//' | sort -u) "$DATA_DIR/validation_labels.csv" >> "$SUBSET_DIR/validation_labels.csv"

        echo "✓ Created subset data with $MAX_SAMPLES training samples and 20 validation samples"

        # Use subset data for training
        DATA_DIR="$SUBSET_DIR"
    fi

    # Build the command
    TRAIN_COMMAND="python -m rna_folding.models.train \
        --data-dir=\"$DATA_DIR\" \
        --output-dir=\"$OUTPUT_DIR\" \
        --batch-size=\"$BATCH_SIZE\" \
        --num-epochs=\"$NUM_EPOCHS\" \
        --learning-rate=\"$LEARNING_RATE\" \
        --weight-decay=\"$WEIGHT_DECAY\" \
        --device=\"$DEVICE\" \
        --num-workers=\"$NUM_WORKERS\" \
        --gradient-accumulation-steps=\"$GRADIENT_ACCUMULATION_STEPS\""

    if [ "$MEMORY_EFFICIENT" = true ]; then
        TRAIN_COMMAND="$TRAIN_COMMAND --memory-efficient"
    fi

    # Execute the command
    eval $TRAIN_COMMAND

    echo "✓ Model saved to $OUTPUT_DIR"
    echo ""
    echo "▶ To generate predictions with the trained model, use:"
    echo "  ./run_rna_pipeline.sh predict --model-path $OUTPUT_DIR/best_model.pt"
}

# Function to run prediction
run_prediction() {
    echo "▶ Generating predictions with the following parameters:"
    echo "  • Model path: $MODEL_PATH"
    echo "  • Sequences file: $SEQUENCES_FILE"
    echo "  • Output file: $OUTPUT_FILE"
    echo "  • Number of predictions: $NUM_PREDICTIONS"
    echo ""

    # Check if model exists
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Model not found at $MODEL_PATH"
        echo "Please train a model first or specify a valid model path with --model-path"
        exit 1
    fi

    # Check if sequences file exists
    if [ ! -f "$SEQUENCES_FILE" ]; then
        echo "Sequences file not found at $SEQUENCES_FILE"
        echo "Please specify a valid sequences file with --sequences-file"
        exit 1
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$OUTPUT_FILE")"

    # Run the prediction script
    python -m rna_folding.models.predict \
        --model-path "$MODEL_PATH" \
        --sequences-file "$SEQUENCES_FILE" \
        --output-file "$OUTPUT_FILE" \
        --num-predictions "$NUM_PREDICTIONS"

    echo "✓ Results saved to $OUTPUT_FILE"
    echo ""
    echo "▶ You can now submit this file to the Kaggle competition!"
}

# Function to run evaluation
run_evaluation() {
    echo "▶ Evaluating model with the following parameters:"
    echo "  • Model path: $MODEL_PATH"
    echo "  • Sequences file: $SEQUENCES_FILE"
    echo "  • Structures file: $STRUCTURES_FILE"
    echo "  • Output file: $EVAL_OUTPUT_FILE"
    echo ""

    # Check if model exists
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Model not found at $MODEL_PATH"
        echo "Please train a model first or specify a valid model path with --model-path"
        exit 1
    fi

    # Check if sequences file exists
    if [ ! -f "$SEQUENCES_FILE" ]; then
        echo "Sequences file not found at $SEQUENCES_FILE"
        echo "Please specify a valid sequences file with --sequences-file"
        exit 1
    fi

    # Check if structures file exists
    if [ ! -f "$STRUCTURES_FILE" ]; then
        echo "Structures file not found at $STRUCTURES_FILE"
        echo "Please specify a valid structures file with --structures-file"
        exit 1
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$EVAL_OUTPUT_FILE")"

    # Run the evaluation script
    python -m rna_folding.evaluation.evaluate \
        --model-path "$MODEL_PATH" \
        --sequences-file "$SEQUENCES_FILE" \
        --structures-file "$STRUCTURES_FILE" \
        --output-file "$EVAL_OUTPUT_FILE"

    echo "✓ Results saved to $EVAL_OUTPUT_FILE"
    echo ""
    echo "▶ You can now analyze the evaluation results to assess model performance."
}

# Function to run test model
run_test() {
    echo "▶ Running test with:"
    echo "  • Number of samples: 10"
    echo "  • Batch size: 2"
    echo "  • Number of epochs: 2"
    echo "  • Device: $DEVICE"
    echo ""

    # Run the test model script
    python test_model.py --num-samples 10 --batch-size 2 --num-epochs 2 --device "$DEVICE"

    echo "▶ To run a training job, use one of the following commands:"
    echo ""
    echo "▶ For a small training job (quick test):"
    echo "  ./run_rna_pipeline.sh train --small"
    echo ""
    echo "▶ For a medium training job (1-2 hours):"
    echo "  ./run_rna_pipeline.sh train --medium"
    echo ""
    echo "▶ For a full training job (up to 8 hours):"
    echo "  ./run_rna_pipeline.sh train --large"
}

# Function to display help
show_help() {
    echo "RNA 3D Structure Prediction Pipeline"
    echo ""
    echo "Usage: ./run_rna_pipeline.sh COMMAND [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  train       Train a model"
    echo "  predict     Generate predictions with a trained model"
    echo "  evaluate    Evaluate a trained model"
    echo "  test        Run a quick test of the model"
    echo "  clean       Clean up the codebase"
    echo "  help        Show this help message"
    echo ""
    echo "Training options:"
    echo "  --small                     Use small training configuration"
    echo "  --medium                    Use medium training configuration"
    echo "  --large                     Use large training configuration"
    echo "  --micro                     Use micro training configuration (for testing)"
    echo "  --data-dir DIR              Directory containing the data"
    echo "  --output-dir DIR            Directory to save model and results"
    echo "  --batch-size SIZE           Batch size"
    echo "  --num-epochs NUM            Number of epochs"
    echo "  --learning-rate RATE        Learning rate"
    echo "  --weight-decay DECAY        Weight decay"
    echo "  --device DEVICE             Device to train on (cuda or cpu)"
    echo "  --num-workers NUM           Number of workers for data loading"
    echo "  --gradient-accumulation-steps STEPS  Number of steps to accumulate gradients"
    echo "  --memory-efficient          Use memory-efficient training techniques"
    echo "  --max-samples NUM           Maximum number of samples to use (0 = all)"
    echo ""
    echo "Prediction options:"
    echo "  --model-path PATH           Path to the model checkpoint"
    echo "  --sequences-file FILE       Path to the sequences CSV file"
    echo "  --output-file FILE          Path to save predictions to"
    echo "  --num-predictions NUM       Number of predictions to generate"
    echo ""
    echo "Evaluation options:"
    echo "  --model-path PATH           Path to the model checkpoint"
    echo "  --sequences-file FILE       Path to the sequences CSV file"
    echo "  --structures-file FILE      Path to the structures CSV file"
    echo "  --eval-output-file FILE     Path to save evaluation results to"
    echo ""
    echo "Examples:"
    echo "  ./run_rna_pipeline.sh train --small"
    echo "  ./run_rna_pipeline.sh predict --model-path models/small/best_model.pt"
    echo "  ./run_rna_pipeline.sh evaluate --model-path models/small/best_model.pt"
}

# Main function
main() {
    print_header

    case $COMMAND in
        train)
            run_training
            ;;
        predict)
            run_prediction
            ;;
        evaluate)
            run_evaluation
            ;;
        test)
            run_test
            ;;
        clean)
            clean_codebase
            ;;
        help|*)
            show_help
            ;;
    esac

    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                        Process Complete                         ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
}

# Run the main function
main
