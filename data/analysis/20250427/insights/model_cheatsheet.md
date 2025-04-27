# RNA 3D Structure Prediction Cheatsheet

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
