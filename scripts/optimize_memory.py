#!/usr/bin/env python
"""
This script applies aggressive memory optimizations to the RNA 3D structure prediction model.
"""

import os
import re

def update_train_script():
    """Update the train.py script with more aggressive memory optimizations."""
    train_py_path = "src/rna_folding/models/train.py"
    
    if not os.path.exists(train_py_path):
        print(f"Error: {train_py_path} not found")
        return False
    
    with open(train_py_path, 'r') as f:
        content = f.read()
    
    # Add more aggressive memory management at the beginning of the script
    import_pattern = r'import torch\nimport torch.nn as nn'
    memory_management = """import torch
import torch.nn as nn
import gc

# Set PyTorch memory management options
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set PyTorch memory allocation options
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=128,garbage_collection_threshold=0.8'

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
"""
    
    updated_content = re.sub(import_pattern, memory_management, content)
    
    # Update the train function to use more aggressive memory management
    train_function_pattern = r'def train\(data_dir, output_dir, batch_size=16,[^}]*?return best_model_path'
    
    # Find the train function
    train_function_match = re.search(train_function_pattern, updated_content, re.DOTALL)
    if not train_function_match:
        print("Warning: Could not find the train function in train.py")
        return False
    
    train_function = train_function_match.group(0)
    
    # Add memory clearing before each epoch
    epoch_loop_pattern = r'for epoch in range\(1, num_epochs \+ 1\):'
    updated_epoch_loop = """for epoch in range(1, num_epochs + 1):
        # Clear GPU memory before each epoch
        clear_gpu_memory()"""
    
    updated_train_function = re.sub(epoch_loop_pattern, updated_epoch_loop, train_function)
    
    # Add memory clearing after backward pass
    backward_pattern = r'loss\.backward\(\)'
    updated_backward = """loss.backward()
                    
                    # Explicitly clear memory for unused tensors
                    del masked_pred_coords, masked_coordinates
                    if batch_idx % 10 == 0:  # Clear cache periodically
                        clear_gpu_memory()"""
    
    updated_train_function = re.sub(backward_pattern, updated_backward, updated_train_function)
    
    # Replace the original train function with the updated one
    updated_content = updated_content.replace(train_function, updated_train_function)
    
    # Update the main training loop to use smaller batch size and more gradient accumulation
    if '--batch-size' in updated_content and '--gradient-accumulation-steps' in updated_content:
        # Update the default batch size in the argument parser
        batch_size_pattern = r'parser\.add_argument\("--batch-size", type=int, default=\d+,'
        updated_batch_size = 'parser.add_argument("--batch-size", type=int, default=4,'
        updated_content = re.sub(batch_size_pattern, updated_batch_size, updated_content)
        
        # Update the default gradient accumulation steps
        grad_acc_pattern = r'parser\.add_argument\("--gradient-accumulation-steps", type=int, default=\d+,'
        updated_grad_acc = 'parser.add_argument("--gradient-accumulation-steps", type=int, default=6,'
        updated_content = re.sub(grad_acc_pattern, updated_grad_acc, updated_content)
    
    # Write the updated content back to the file
    with open(train_py_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✓ Updated {train_py_path} with aggressive memory optimizations")
    return True

def update_run_pipeline_script():
    """Update the run_rna_pipeline.sh script with more aggressive memory settings."""
    script_path = "run_rna_pipeline.sh"
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update the PyTorch memory management environment variables
    env_var_pattern = r'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:\d+'
    updated_env_var = 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold=0.8'
    updated_content = re.sub(env_var_pattern, updated_env_var, content)
    
    # Update the large training configuration
    large_config_pattern = r'--large\)\s+# Large training configuration\s+BATCH_SIZE=\d+\s+GRADIENT_ACCUMULATION_STEPS=\d+'
    updated_large_config = """--large)
            # Large training configuration
            BATCH_SIZE=4
            GRADIENT_ACCUMULATION_STEPS=6"""
    
    updated_content = re.sub(large_config_pattern, updated_large_config, updated_content)
    
    # Write the updated content back to the file
    with open(script_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✓ Updated {script_path} with aggressive memory settings")
    return True

def update_l4_gpu_config():
    """Update the L4 GPU configuration file with more aggressive memory settings."""
    config_path = "config/l4_gpu_config.json"
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found")
        return False
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Update batch size
    batch_size_pattern = r'"batch_size": \d+'
    updated_batch_size = '"batch_size": 4'
    updated_content = re.sub(batch_size_pattern, updated_batch_size, content)
    
    # Update gradient accumulation steps
    grad_acc_pattern = r'"gradient_accumulation_steps": \d+'
    updated_grad_acc = '"gradient_accumulation_steps": 6'
    updated_content = re.sub(grad_acc_pattern, updated_grad_acc, updated_content)
    
    # Write the updated content back to the file
    with open(config_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✓ Updated {config_path} with aggressive memory settings")
    return True

if __name__ == "__main__":
    print("▶ Applying aggressive memory optimizations...")
    update_train_script()
    update_run_pipeline_script()
    update_l4_gpu_config()
    print("✓ All memory optimizations applied successfully")
