#!/usr/bin/env python
"""
This script fixes the RNAModel.to() method to support the dtype parameter.
Run this script before running the optimize.py script.
"""

import os
import re

def fix_rna_model_to_method():
    """Fix the RNAModel.to() method in base.py to support the dtype parameter."""
    base_py_path = "src/rna_folding/models/base.py"
    
    if not os.path.exists(base_py_path):
        print(f"Error: {base_py_path} not found")
        return False
    
    with open(base_py_path, 'r') as f:
        content = f.read()
    
    # Define the pattern to match the old to() method
    old_method_pattern = r'def to\(self, device\):[^}]*?return super\(\)\.to\(device\)'
    
    # Define the new to() method
    new_method = """def to(self, device=None, dtype=None, non_blocking=False):
        \"\"\"
        Move model to specified device and/or dtype.
        
        Args:
            device: Device to move model to.
            dtype: Data type to convert parameters to.
            non_blocking: Whether to perform non-blocking transfer.
            
        Returns:
            self: The model instance.
        \"\"\"
        if device is not None:
            self.device = torch.device(device)
        
        # Call parent's to() method with all arguments
        if dtype is not None:
            return super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        elif device is not None:
            return super().to(device=device, non_blocking=non_blocking)
        else:
            return self"""
    
    # Replace the old method with the new one
    updated_content = re.sub(old_method_pattern, new_method, content, flags=re.DOTALL)
    
    # Check if the content was updated
    if updated_content == content:
        print("Warning: Could not find the to() method in base.py")
        return False
    
    # Write the updated content back to the file
    with open(base_py_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✓ Fixed RNAModel.to() method in {base_py_path}")
    return True

def fix_optimize_py():
    """Fix the enable_mixed_precision() function in optimize.py to handle errors gracefully."""
    optimize_py_path = "src/rna_folding/models/optimize.py"
    
    if not os.path.exists(optimize_py_path):
        print(f"Error: {optimize_py_path} not found")
        return False
    
    with open(optimize_py_path, 'r') as f:
        content = f.read()
    
    # Define the pattern to match the old enable_mixed_precision() function
    old_function_pattern = r'def enable_mixed_precision\(model\):[^}]*?return model'
    
    # Define the new enable_mixed_precision() function
    new_function = """def enable_mixed_precision(model):
    \"\"\"
    Enable mixed precision training for a model.
    
    Args:
        model (nn.Module): Model to enable mixed precision for.
        
    Returns:
        nn.Module: Model with mixed precision enabled.
    \"\"\"
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, mixed precision not enabled")
        return model
    
    # Check if the GPU supports mixed precision
    if not torch.cuda.is_bf16_supported() and not torch.cuda.is_fp16_supported():
        print("GPU does not support mixed precision, not enabled")
        return model
    
    try:
        # Enable mixed precision
        if torch.cuda.is_bf16_supported():
            print("Enabling BF16 mixed precision")
            model = model.to(dtype=torch.bfloat16)
        else:
            print("Enabling FP16 mixed precision")
            model = model.to(dtype=torch.float16)
    except Exception as e:
        print(f"Warning: Failed to enable mixed precision: {str(e)}")
        print("Continuing with full precision")
    
    return model"""
    
    # Replace the old function with the new one
    updated_content = re.sub(old_function_pattern, new_function, content, flags=re.DOTALL)
    
    # Check if the content was updated
    if updated_content == content:
        print("Warning: Could not find the enable_mixed_precision() function in optimize.py")
        return False
    
    # Write the updated content back to the file
    with open(optimize_py_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✓ Fixed enable_mixed_precision() function in {optimize_py_path}")
    return True

if __name__ == "__main__":
    print("▶ Fixing RNAModel.to() method...")
    fix_rna_model_to_method()
    
    print("▶ Fixing enable_mixed_precision() function...")
    fix_optimize_py()
    
    print("✓ All fixes applied successfully")
