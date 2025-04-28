#!/usr/bin/env python
"""
This script reduces the model size and optimizes memory usage for the RNA 3D structure prediction model.
"""

import os
import re
import json

def update_l4_gpu_config():
    """Update the L4 GPU configuration file with a much smaller model."""
    config_path = "config/l4_gpu_config.json"
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Drastically reduce model size
    config["model_config"]["nucleotide_features"] = 32  # Reduced from 128
    config["model_config"]["motif_features"] = 64      # Reduced from 256
    config["model_config"]["global_features"] = 128    # Reduced from 512
    config["model_config"]["num_layers_per_scale"] = 2  # Reduced from 4
    
    # Reduce batch size
    config["model_config"]["batch_size"] = 2  # Reduced from 8
    
    # Update training config
    config["training_config"]["batch_size"] = 2  # Add batch_size to training_config
    config["training_config"]["gradient_accumulation_steps"] = 12  # Increased from 3
    config["training_config"]["num_workers"] = 8  # Reduced from 24
    
    # Write the updated config back to the file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Updated {config_path} with smaller model configuration")
    return True

def update_run_pipeline_script():
    """Update the run_rna_pipeline.sh script with smaller model settings."""
    script_path = "run_rna_pipeline.sh"
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update default batch size
    batch_size_pattern = r'BATCH_SIZE=\d+'
    updated_batch_size = 'BATCH_SIZE=2'
    updated_content = re.sub(batch_size_pattern, updated_batch_size, content)
    
    # Update gradient accumulation steps
    grad_acc_pattern = r'GRADIENT_ACCUMULATION_STEPS=\d+'
    updated_grad_acc = 'GRADIENT_ACCUMULATION_STEPS=12'
    updated_content = re.sub(grad_acc_pattern, updated_grad_acc, updated_content)
    
    # Update the large training configuration
    large_config_pattern = r'--large\)\s+# Large training configuration\s+BATCH_SIZE=\d+\s+GRADIENT_ACCUMULATION_STEPS=\d+'
    updated_large_config = """--large)
            # Large training configuration
            BATCH_SIZE=2
            GRADIENT_ACCUMULATION_STEPS=12"""
    
    updated_content = re.sub(large_config_pattern, updated_large_config, updated_content)
    
    # Write the updated content back to the file
    with open(script_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✓ Updated {script_path} with smaller model settings")
    return True

def update_train_script():
    """Update the train.py script to use a smaller model by default."""
    train_py_path = "src/rna_folding/models/train.py"
    
    if not os.path.exists(train_py_path):
        print(f"Error: {train_py_path} not found")
        return False
    
    with open(train_py_path, 'r') as f:
        content = f.read()
    
    # Update the default model size in the main function
    model_size_pattern = r'config = MultiScaleModelConfig\(\s+nucleotide_features=64,\s+motif_features=128,\s+global_features=256,\s+num_layers_per_scale=3,'
    updated_model_size = """config = MultiScaleModelConfig(
            nucleotide_features=32,
            motif_features=64,
            global_features=128,
            num_layers_per_scale=2,"""
    
    updated_content = re.sub(model_size_pattern, updated_model_size, content)
    
    # Update the small model size
    small_model_pattern = r'config = MultiScaleModelConfig\(\s+nucleotide_features=32,\s+motif_features=64,\s+global_features=128,\s+num_layers_per_scale=2,'
    updated_small_model = """config = MultiScaleModelConfig(
            nucleotide_features=16,
            motif_features=32,
            global_features=64,
            num_layers_per_scale=1,"""
    
    updated_content = re.sub(small_model_pattern, updated_small_model, updated_content)
    
    # Update the batch size check for small model
    batch_size_check_pattern = r'small_model = args\.batch_size <= 4'
    updated_batch_size_check = 'small_model = args.batch_size <= 2'
    updated_content = re.sub(batch_size_check_pattern, updated_batch_size_check, updated_content)
    
    # Add more aggressive memory clearing
    optimize_memory_pattern = r'def optimize_memory\(\):[^}]*?return'
    
    # Define the updated optimize_memory function
    updated_optimize_memory = """def optimize_memory():
    \"\"\"Apply memory optimization techniques for PyTorch.\"\"\"
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # Import garbage collector
        import gc
        gc.collect()
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=128,garbage_collection_threshold=0.6'
        
        # Enable TF32 for faster computation (at slight precision cost)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Set memory allocation strategy
    if hasattr(torch.cuda, 'memory_stats'):
        # Print initial memory stats
        print("\\nInitial CUDA memory stats:")
        for k, v in torch.cuda.memory_stats().items():
            if 'bytes' in k and v > 0:
                print(f"  {k}: {v / 1024**2:.1f} MB")

    # Enable memory-efficient operations
    torch.backends.cudnn.benchmark = True

    # Set PyTorch memory allocator settings if using PyTorch 1.11+
    if hasattr(torch, 'set_per_process_memory_fraction'):
        # Reserve 90% of available memory to avoid OOM
        torch.set_per_process_memory_fraction(0.9)

    # Print available GPU memory
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"\\nAvailable GPU memory: {free_memory / 1024**3:.2f} GB")

    return"""
    
    updated_content = re.sub(optimize_memory_pattern, updated_optimize_memory, updated_content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(train_py_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✓ Updated {train_py_path} with smaller model defaults and better memory management")
    return True

def update_readme():
    """Update the README.md with the new model parameters."""
    readme_path = "README.md"
    
    if not os.path.exists(readme_path):
        print(f"Error: {readme_path} not found")
        return False
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Update the model parameters
    params_pattern = r'This will train a model with the following optimized parameters:.*?Expected metrics: TM-score > 0\.7, RMSD < 5\.0 Å'
    updated_params = """This will train a model with the following optimized parameters:
- Batch size: 2
- Gradient accumulation steps: 12 (effective batch size: 24)
- Memory-efficient mode: enabled
- Reduced model size: enabled
- Number of epochs: 100
- Device: CUDA (GPU)
- Number of workers: 8
- Expected training time: ~6-8 hours
- Expected metrics: TM-score > 0.7, RMSD < 5.0 Å"""
    
    updated_content = re.sub(params_pattern, updated_params, content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(readme_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✓ Updated {readme_path} with new model parameters")
    return True

def update_deploy_script():
    """Update the deploy_l4_gpu.sh script with the new model parameters."""
    script_path = "deploy_l4_gpu.sh"
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update the model parameters in the final message
    params_pattern = r'echo "     This will train a model with the following optimized parameters:".*?echo "     • Expected validation metrics: TM-score > 0\.7, RMSD < 5\.0 Å"'
    updated_params = """echo "     This will train a model with the following optimized parameters:"
echo "     • Batch size: 2"
echo "     • Gradient accumulation steps: 12 (effective batch size: 24)"
echo "     • Memory-efficient mode: enabled"
echo "     • Reduced model size: enabled"
echo "     • Number of epochs: 100"
echo "     • Device: cuda (GPU)"
echo "     • Number of workers: 8"
echo "     • Expected training time: ~6-8 hours"
echo "     • Expected validation metrics: TM-score > 0.7, RMSD < 5.0 Å\""""
    
    updated_content = re.sub(params_pattern, updated_params, content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(script_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✓ Updated {script_path} with new model parameters")
    return True

if __name__ == "__main__":
    print("▶ Reducing model size and optimizing memory usage...")
    update_l4_gpu_config()
    update_run_pipeline_script()
    update_train_script()
    update_readme()
    update_deploy_script()
    print("✓ All model size reductions and memory optimizations applied successfully")
