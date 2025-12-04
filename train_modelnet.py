import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import argparse
import importlib
from datetime import datetime
import numpy as np

from transformer.TransformerClassifier import TransformerClassifier
from data.modelnet_data import create_modelnet_dataloaders_with_val, MODELNET40_CLASSES

# Setup logging to file with current date/time
parser = argparse.ArgumentParser(description='Train ModelNet40 classification models')
parser.add_argument('--runs', type=str, required=True,
                    help='Name of the runs file to import (e.g., modelnet)')
args = parser.parse_args()
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = open(f'logs/train_modelnet_{timestamp}_{args.runs}.log', 'w')

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
num_points = 1024  # Standard for point cloud benchmarks
use_normals = False  # Start with xyz only (3D features)
feature_dim = 6 if use_normals else 3
num_classes = 40
dim = 256
heads = 8
ffdim = 4 * dim
batch_size = 32
lr = 1e-4
weight_decay = 0.01
grad_clip = 1.0

# Load dataset
log("Loading ModelNet40 dataset...")
train_loader, val_loader, test_loader, dataset_stats = create_modelnet_dataloaders_with_val(
    root='data/ModelNet40',
    num_points=num_points,
    use_normals=use_normals,
    batch_size=batch_size,
    normalize=True,
    val_split=0.1,
    num_workers=0,
    seed=42
)
log(f"Dataset loaded: train={dataset_stats['train_size']}, val={dataset_stats['val_size']}, test={dataset_stats['test_size']}")
log(f"Feature dim: {dataset_stats['feature_dim']}, Num classes: {dataset_stats['num_classes']}")

# Get sequence length from runs file
def load_runs(runs_name):
    """Load models and steps from a runs file"""
    try:
        runs_module = importlib.import_module(f'runs.{runs_name}')
        models = runs_module.models
        steps = getattr(runs_module, 'steps', 20000)  # Default to 20000 if not specified
        T = getattr(runs_module, 'T', 1024)  # Default sequence length for point clouds
        log(f'Loaded {len(models)} models from runs.{runs_name}')
        log(f'Number of training steps: {steps}')
        log(f'Sequence length T: {T}')
        return models, steps, T
    except ImportError as e:
        raise ImportError(f"Could not import runs.{runs_name}. Make sure the file exists in the runs/ directory. Error: {e}")
    except AttributeError as e:
        raise AttributeError(f"runs.{runs_name} does not have a 'models' attribute. Error: {e}")

def train(name, attn_class, attn_args, train_loader, val_loader, T, n_layers, steps, runs_name):
    log(f'\n----Training {name}----')
    # Clear CUDA cache before creating new model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create checkpoint directory for this model
    checkpoint_dir = f'models/{runs_name}/{name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    log(f'Checkpoints will be saved to {checkpoint_dir}')
    
    model = TransformerClassifier(
        feature_dim=feature_dim,
        dim=dim,
        heads=heads,
        ffdim=ffdim,
        T=T,
        n_layers=n_layers,
        attn_class=attn_class,
        attn_args=attn_args,
        num_classes=num_classes,
        pooling='mean'
    ).to(device)
    
    # Log model parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'Model parameters: {n_params:,}')
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    step_idx = 0
    
    while step_idx < steps:
        for points, labels, masks in train_loader:
            if step_idx >= steps:
                break
                
            points = points.to(device)  # (B, num_points, feature_dim)
            labels = labels.to(device)  # (B,)
            masks = masks.to(device)    # (B, num_points)
            
            # Check for NaN/Inf in inputs
            if torch.isnan(points).any() or torch.isinf(points).any():
                log(f"WARNING: NaN/Inf detected in points at step {step_idx}")
                continue
            
            optimizer.zero_grad()
            logits, loss = model(points, labels, attn_mask=masks)
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                log(f"WARNING: NaN/Inf loss at step {step_idx}")
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
                # Compute training accuracy for this batch
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == labels).float().mean().item()
                
                log(f"{name}: step {step_idx}/{steps} | loss {loss.item():.4f} | acc {acc:.4f}")
                
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

def evaluate_classification(model, data_loader, device):
    """Evaluate classification model and return metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    losses = []
    
    with torch.no_grad():
        for points, labels, masks in data_loader:
            points = points.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            logits, loss = model(points, labels, attn_mask=masks)
            preds = logits.argmax(dim=-1)
            
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            losses.append(loss.item())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    avg_loss = np.mean(losses)
    
    # Overall accuracy
    accuracy = (all_predictions == all_labels).mean()
    
    # Per-class accuracy
    per_class_acc = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            class_acc = (all_predictions[mask] == all_labels[mask]).mean()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    per_class_acc = np.array(per_class_acc)
    mean_class_acc = per_class_acc.mean()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mean_class_accuracy': mean_class_acc,
        'per_class_accuracy': per_class_acc,
    }

def run_all_models(runs_name):
    models, steps, T = load_runs(runs_name)
    log(f'Running {runs_name} with {len(models)} models')
    log('models:', models)

    results = {}
    for name, attn_class, attn_args, n_layers in models:
        model = train(name, attn_class, attn_args, train_loader, val_loader, T, n_layers, steps, runs_name)
        
        # Evaluate on train, validation, and test sets
        train_metrics = evaluate_classification(model, train_loader, device)
        val_metrics = evaluate_classification(model, val_loader, device)
        test_metrics = evaluate_classification(model, test_loader, device)
        
        log(f"\n{name} - Training Metrics:")
        log(f"  Loss: {train_metrics['loss']:.4f}")
        log(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        log(f"  Mean Class Accuracy: {train_metrics['mean_class_accuracy']:.4f}")
        
        log(f"\n{name} - Validation Metrics:")
        log(f"  Loss: {val_metrics['loss']:.4f}")
        log(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        log(f"  Mean Class Accuracy: {val_metrics['mean_class_accuracy']:.4f}")
        
        log(f"\n{name} - Test Metrics:")
        log(f"  Loss: {test_metrics['loss']:.4f}")
        log(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        log(f"  Mean Class Accuracy: {test_metrics['mean_class_accuracy']:.4f}")
        
        # Log worst performing classes
        worst_classes = np.argsort(test_metrics['per_class_accuracy'])[:5]
        log(f"  Worst 5 classes:")
        for c in worst_classes:
            log(f"    {MODELNET40_CLASSES[c]}: {test_metrics['per_class_accuracy'][c]:.4f}")
        
        results[name] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        }
    
    return results

if __name__ == '__main__':
    log(f'Starting ModelNet40 training with runs.{args.runs}...')
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

