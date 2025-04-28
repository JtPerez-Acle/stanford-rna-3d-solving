"""
Optimization utilities for RNA 3D structure prediction models.

This module provides utilities for optimizing models for different hardware
configurations, including mixed precision training, gradient checkpointing,
and memory-efficient techniques.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
import argparse
from dataclasses import asdict

from rna_folding.models.multi_scale import MultiScaleRNA, MultiScaleModelConfig


def enable_mixed_precision(model):
    """
    Enable mixed precision training for a model.
    
    Args:
        model (nn.Module): Model to enable mixed precision for.
        
    Returns:
        nn.Module: Model with mixed precision enabled.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, mixed precision not enabled")
        return model
    
    # Check if the GPU supports mixed precision
    if not torch.cuda.is_bf16_supported() and not torch.cuda.is_fp16_supported():
        print("GPU does not support mixed precision, not enabled")
        return model
    
    # Enable mixed precision
    if torch.cuda.is_bf16_supported():
        print("Enabling BF16 mixed precision")
        model = model.to(dtype=torch.bfloat16)
    else:
        print("Enabling FP16 mixed precision")
        model = model.to(dtype=torch.float16)
    
    return model


def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing for a model to reduce memory usage.
    
    Args:
        model (nn.Module): Model to enable gradient checkpointing for.
        
    Returns:
        nn.Module: Model with gradient checkpointing enabled.
    """
    # Check if the model has modules that support gradient checkpointing
    checkpointing_available = False
    
    for module in model.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = True
            checkpointing_available = True
    
    if not checkpointing_available:
        print("No modules support gradient checkpointing, not enabled")
    else:
        print("Gradient checkpointing enabled")
    
    return model


def optimize_for_inference(model):
    """
    Optimize a model for inference by disabling dropout and batch normalization.
    
    Args:
        model (nn.Module): Model to optimize for inference.
        
    Returns:
        nn.Module: Model optimized for inference.
    """
    model.eval()
    
    # Disable dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0
    
    # Freeze batch normalization
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.eval()
    
    print("Model optimized for inference")
    return model


def optimize_for_l4_gpu(model):
    """
    Optimize a model specifically for NVIDIA L4 GPU.
    
    Args:
        model (nn.Module): Model to optimize.
        
    Returns:
        nn.Module: Optimized model.
    """
    # Enable mixed precision (L4 has good Tensor Core performance)
    model = enable_mixed_precision(model)
    
    # Enable gradient checkpointing for training
    if model.training:
        model = enable_gradient_checkpointing(model)
    
    # Set optimal CUDA settings for L4
    if torch.cuda.is_available():
        # Set optimal CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("Model optimized for NVIDIA L4 GPU")
    return model


def create_optimized_model(config_path):
    """
    Create an optimized model from a configuration file.
    
    Args:
        config_path (str or Path): Path to the configuration file.
        
    Returns:
        nn.Module: Optimized model.
    """
    # Load configuration
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create model configuration
    model_config = MultiScaleModelConfig(**config_dict['model_config'])
    
    # Create model
    model = MultiScaleRNA(model_config)
    
    # Get system configuration
    system_config = config_dict.get('system_config', {})
    device = system_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device
    model = model.to(device)
    
    # Optimize model based on device
    if device == 'cuda':
        if 'gpu_memory' in system_config and system_config['gpu_memory'] >= 20:
            # Optimize for high-end GPU (like L4)
            model = optimize_for_l4_gpu(model)
        else:
            # Enable mixed precision for any GPU
            model = enable_mixed_precision(model)
    
    return model, config_dict


def save_optimized_model(model, config_dict, output_path):
    """
    Save an optimized model and its configuration.
    
    Args:
        model (nn.Module): Model to save.
        config_dict (dict): Configuration dictionary.
        output_path (str or Path): Path to save the model to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': asdict(model.config)
    }, output_path)
    
    # Save configuration
    with open(output_path.parent / 'model_config.json', 'w') as f:
        json.dump(config_dict['model_config'], f, indent=2)
    
    # Save full configuration
    with open(output_path.parent / 'full_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Optimized model saved to {output_path}")


def main():
    """Main function to create and save an optimized model."""
    parser = argparse.ArgumentParser(description="Create and save an optimized model")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--output", required=True, help="Path to save the model to")
    
    args = parser.parse_args()
    
    # Create optimized model
    model, config_dict = create_optimized_model(args.config)
    
    # Save optimized model
    save_optimized_model(model, config_dict, args.output)


if __name__ == "__main__":
    main()
