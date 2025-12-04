#!/usr/bin/env python3
"""
Script to print model sizes for all models defined in runs/*.py files
"""

import os
import importlib
import importlib.util
import torch
from pathlib import Path

from transformer.TransformerLM import TransformerLM
from transformer.TransformerRegressor import TransformerRegressor

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_parameter_count(num_params):
    """Format parameter count in a human-readable way"""
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)

def print_model_size(name, attn_class, attn_args, n_layers, is_hep=False, T=512):
    """Create a model and print its size"""
    device = 'cpu'  # Use CPU to avoid GPU memory issues
    
    # Model hyperparameters
    dim = 256
    heads = 8
    ffdim = 4 * dim
    
    try:
        if is_hep:
            # HEP regression model
            feature_dim = 6
            target_dim = 4
            model = TransformerRegressor(
                feature_dim=feature_dim,
                dim=dim,
                heads=heads,
                ffdim=ffdim,
                T=T,
                n_layers=n_layers,
                attn_class=attn_class,
                attn_args=attn_args,
                target_dim=target_dim,
                pooling='mean'
            ).to(device)
        else:
            # Language modeling model
            V = 256  # vocabulary size
            model = TransformerLM(
                dim=dim,
                heads=heads,
                ffdim=ffdim,
                V=V,
                T=T,
                n_layers=n_layers,
                attn_class=attn_class,
                attn_args=attn_args
            ).to(device)
        
        num_params = count_parameters(model)
        formatted = format_parameter_count(num_params)
        print(f"  {name:40s} {num_params:>12,} params ({formatted:>8s})")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return num_params
    except Exception as e:
        print(f"  {name:40s} ERROR: {e}")
        return None

def process_runs_file(runs_file_path):
    """Process a single runs file"""
    runs_name = runs_file_path.stem
    
    try:
        # Import the runs module
        spec = importlib.util.spec_from_file_location(f"runs.{runs_name}", runs_file_path)
        runs_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runs_module)
        
        # Get models and T
        models = getattr(runs_module, 'models', [])
        T = getattr(runs_module, 'T', 512)
        
        if not models:
            print(f"\n{runs_name}: No models found")
            return
        
        # Determine if this is HEP or language modeling
        is_hep = 'hep' in runs_name.lower()
        
        print(f"\n{runs_name} ({'HEP' if is_hep else 'Language Modeling'}):")
        print(f"  T={T}, {len(models)} models")
        print("-" * 70)
        
        total_params = 0
        for name, attn_class, attn_args, n_layers in models:
            num_params = print_model_size(name, attn_class, attn_args, n_layers, is_hep, T)
            if num_params is not None:
                total_params += num_params
        
        print("-" * 70)
        print(f"  {'Total':40s} {total_params:>12,} params ({format_parameter_count(total_params):>8s})")
        
    except Exception as e:
        print(f"\n{runs_name}: ERROR loading - {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to process all runs files"""
    runs_dir = Path(__file__).parent / "runs"
    
    if not runs_dir.exists():
        print(f"Error: runs directory not found at {runs_dir}")
        return
    
    # Find all Python files in runs directory (except __init__.py)
    runs_files = sorted([f for f in runs_dir.glob("*.py") if f.name != "__init__.py"])
    
    if not runs_files:
        print("No runs files found")
        return
    
    print("=" * 70)
    print("Model Size Report")
    print("=" * 70)
    
    for runs_file in runs_files:
        process_runs_file(runs_file)
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()

