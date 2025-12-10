import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import argparse
import importlib
import random
from datetime import datetime
from sklearn.metrics import r2_score
import numpy as np

from transformer.TransformerRegressor import TransformerRegressor
from data.hep_data import create_hep_dataloaders

# Setup logging to file with current date/time
parser = argparse.ArgumentParser(description='Train HEP regression models')
parser.add_argument('--runs', type=str, required=True,
                    help='Name of the runs file to import (e.g., hep)')
args = parser.parse_args()
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = open(f'logs/train_hep_{timestamp}_{args.runs}.log', 'w')

def log(*args, **kwargs):
    """Write to log file instead of printing"""
    message = ' '.join(str(arg) for arg in args)
    if kwargs:
        message += ' ' + ' '.join(f'{k}={v}' for k, v in kwargs.items())
    log_file.write(message + '\n')
    print(message)
    log_file.flush()  # Ensure immediate write

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f'Using device: {device}')

# Hyperparameters
feature_dim = 6  # pT, eta, phi, mass, charge, pid
target_dim = 4  # px, py, pz, E (missing momentum)
dim = 256
heads = 8
ffdim = 4 * dim
batch_size = 32
lr = 1e-4  # Reduced learning rate for stability
weight_decay = 0.01
grad_clip = 1.0

# Base seed for reproducibility (can be overridden per model)
base_seed = 42

# Get sequence length from runs file
def load_runs(runs_name):
    """Load models and steps from a runs file"""
    try:
        runs_module = importlib.import_module(f'runs.{runs_name}')
        models = runs_module.models
        steps = getattr(runs_module, 'steps', 20000)  # Default to 20000 if not specified
        T = getattr(runs_module, 'T', 512)  # Default sequence length
        log(f'Loaded {len(models)} models from runs.{runs_name}')
        log(f'Number of training steps: {steps}')
        log(f'Sequence length T: {T}')
        return models, steps, T
    except ImportError as e:
        raise ImportError(f"Could not import runs.{runs_name}. Make sure the file exists in the runs/ directory. Error: {e}")
    except AttributeError as e:
        raise AttributeError(f"runs.{runs_name} does not have a 'models' attribute. Error: {e}")

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def train(name, attn_class, attn_args, train_loader, val_loader, T, n_layers, steps, runs_name, seed=None):
    log(f'\n----Training {name}----')
    if seed is not None:
        log(f'Using random seed: {seed}')
        set_seed(seed)
    
    # Clear CUDA cache before creating new model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create checkpoint directory for this model
    checkpoint_dir = f'models/{runs_name}/{name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    log(f'Checkpoints will be saved to {checkpoint_dir}')
    
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
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    step_idx = 0
    
    while step_idx < steps:
        for features, targets, masks in train_loader:
            if step_idx >= steps:
                break
                
            features = features.to(device)  # (B, T, feature_dim)
            targets = targets.to(device)  # (B, target_dim)
            masks = masks.to(device)  # (B, T)
            
            # Check for NaN/Inf in inputs
            if torch.isnan(features).any() or torch.isinf(features).any():
                log(f"WARNING: NaN/Inf detected in features at step {step_idx}")
                continue
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                log(f"WARNING: NaN/Inf detected in targets at step {step_idx}")
                continue
            
            # Check for sequences with no real tokens
            if masks.sum(dim=1).min() == 0:
                log(f"WARNING: Found sequence with all padding at step {step_idx}")
                continue
            
            optimizer.zero_grad()
            predictions, loss = model(features, targets, attn_mask=masks)
            
            # Check for NaN in predictions or loss
            if torch.isnan(loss) or torch.isinf(loss):
                log(f"WARNING: NaN/Inf loss at step {step_idx}")
                log(f"  Predictions stats: min={predictions.min().item():.4f}, max={predictions.max().item():.4f}, mean={predictions.mean().item():.4f}")
                log(f"  Targets stats: min={targets.min().item():.4f}, max={targets.max().item():.4f}, mean={targets.mean().item():.4f}")
                log(f"  Features stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
                # Skip this batch
                continue
            
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                log(f"WARNING: NaN/Inf gradients detected at step {step_idx}, skipping update")
                optimizer.zero_grad()
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            step_idx += 1

            if step_idx % 500 == 0:
                log(f"{name}: step {step_idx}/{steps} | loss {loss.item():.4f}")
                
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step_idx}.pt')
                torch.save({
                    'step': step_idx,
                    'model_state_dict': model.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                log(f"Saved checkpoint to {checkpoint_path}")
                
                # Periodically clear cache during training
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return model

def evaluate_regression(model, data_loader, device):
    """Evaluate regression model and return metrics."""
    model.eval()
    all_predictions = []
    all_targets = []
    losses = []
    
    with torch.no_grad():
        for features, targets, masks in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            predictions, loss = model(features, targets, attn_mask=masks)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            losses.append(loss.item())
    
    all_predictions = np.concatenate(all_predictions, axis=0)  # (N, target_dim)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, target_dim)
    avg_loss = np.mean(losses)
    
    # Compute metrics per dimension
    mae_per_dim = np.mean(np.abs(all_predictions - all_targets), axis=0)  # (target_dim,)
    rmse_per_dim = np.sqrt(np.mean((all_predictions - all_targets)**2, axis=0))  # (target_dim,)
    r2_per_dim = np.array([r2_score(all_targets[:, i], all_predictions[:, i]) for i in range(target_dim)])
    
    # Overall metrics
    mae_overall = np.mean(mae_per_dim)
    rmse_overall = np.sqrt(avg_loss)  # RMSE from MSE loss
    r2_overall = np.mean(r2_per_dim)
    
    return {
        'loss': avg_loss,
        'mae': mae_overall,
        'mae_per_dim': mae_per_dim,
        'rmse': rmse_overall,
        'rmse_per_dim': rmse_per_dim,
        'r2': r2_overall,
        'r2_per_dim': r2_per_dim,
    }

def run_all_models(runs_name):
    models, steps, T = load_runs(runs_name)
    log(f'Running {runs_name} with {len(models)} models')
    log('models:', models)
    log(f'Base seed: {base_seed}')

    results = {}
    for idx, (name, attn_class, attn_args, n_layers) in enumerate(models):
        # Use different seed for each model run
        # This ensures different random initialization and training randomness
        model_seed = base_seed + idx
        log(f'Model {idx+1}/{len(models)}: {name} will use seed {model_seed}')
        
        # Load dataset with this seed (different data generation/shuffling)
        train_loader, val_loader, test_loader, feature_stats = create_hep_dataloaders(
            n_train=10000,
            n_val=2000,
            n_test=2000,
            min_particles=50,
            max_particles=500,
            max_length=T,
            batch_size=batch_size,
            normalize=True,
            device=device,
            seed=model_seed
        )
        
        model = train(name, attn_class, attn_args, train_loader, val_loader, T, n_layers, steps, runs_name, seed=model_seed)
        
        # Evaluate on train and validation sets
        train_metrics = evaluate_regression(model, train_loader, device)
        val_metrics = evaluate_regression(model, val_loader, device)
        test_metrics = evaluate_regression(model, test_loader, device)
        
        log(f"\n{name} - Training Metrics:")
        log(f"  Loss: {train_metrics['loss']:.4f}")
        log(f"  MAE: {train_metrics['mae']:.4f}")
        log(f"  RMSE: {train_metrics['rmse']:.4f}")
        log(f"  R²: {train_metrics['r2']:.4f}")
        log(f"  MAE per dim: {train_metrics['mae_per_dim']}")
        log(f"  R² per dim: {train_metrics['r2_per_dim']}")
        
        log(f"\n{name} - Validation Metrics:")
        log(f"  Loss: {val_metrics['loss']:.4f}")
        log(f"  MAE: {val_metrics['mae']:.4f}")
        log(f"  RMSE: {val_metrics['rmse']:.4f}")
        log(f"  R²: {val_metrics['r2']:.4f}")
        log(f"  MAE per dim: {val_metrics['mae_per_dim']}")
        log(f"  R² per dim: {val_metrics['r2_per_dim']}")
        
        log(f"\n{name} - Test Metrics:")
        log(f"  Loss: {test_metrics['loss']:.4f}")
        log(f"  MAE: {test_metrics['mae']:.4f}")
        log(f"  RMSE: {test_metrics['rmse']:.4f}")
        log(f"  R²: {test_metrics['r2']:.4f}")
        log(f"  MAE per dim: {test_metrics['mae_per_dim']}")
        log(f"  R² per dim: {test_metrics['r2_per_dim']}")
        
        results[name] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        }
    
    return results

if __name__ == '__main__':
    log(f'Starting HEP training with runs.{args.runs}...')
    try:
        run_all_models(runs_name=args.runs)
        log('Training completed successfully!')
    except Exception as e:
        log(f'Error during training: {e}')
        import traceback
        log(traceback.format_exc())
        raise
    finally:
        log_file.close()

