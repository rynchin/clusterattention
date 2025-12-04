"""
Training script for HEP jet tagging (classification) benchmark.
Uses the improved jet tagging dataset that requires learning particle relationships.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import argparse
import importlib
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

from transformer.TransformerClassifier import TransformerClassifier
from data.hep_data_v2 import create_jet_tagging_dataloaders

# Setup logging to file with current date/time
parser = argparse.ArgumentParser(description='Train HEP jet tagging (classification) models')
parser.add_argument('--runs', type=str, required=True,
                    help='Name of the runs file to import (e.g., hep_v2)')
args = parser.parse_args()
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = open(f'logs/train_hep_v2_{timestamp}_{args.runs}.log', 'w')

def log(*args, **kwargs):
    """Write to log file instead of printing"""
    message = ' '.join(str(arg) for arg in args)
    if kwargs:
        message += ' ' + ' '.join(f'{k}={v}' for k, v in kwargs.items())
    log_file.write(message + '\n')
    print(message)
    log_file.flush()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f'Using device: {device}')

# Hyperparameters
feature_dim = 6  # pT, eta, phi, mass, charge, pid
dim = 256
heads = 8
ffdim = 4 * dim
batch_size = 32
lr = 3e-4
weight_decay = 0.01
grad_clip = 1.0

# Load dataset
log("Loading jet tagging dataset...")
train_loader, val_loader, test_loader, stats = create_jet_tagging_dataloaders(
    n_train=10000,
    n_val=2000,
    n_test=2000,
    n_particles_per_jet=50,
    max_length=128,
    batch_size=batch_size,
    normalize=True,
    device='cpu',  # Data on CPU, move to device during training
    seed=42
)
log(f"Dataset loaded: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

# Log class balance
train_targets = torch.cat([t for _, t, _ in train_loader])
log(f"Training class balance: {train_targets.mean():.3f} (gluon fraction)")


def load_runs(runs_name):
    """Load models and steps from a runs file"""
    try:
        runs_module = importlib.import_module(f'runs.{runs_name}')
        models = runs_module.models
        steps = getattr(runs_module, 'steps', 20000)
        T = getattr(runs_module, 'T', 128)
        log(f'Loaded {len(models)} models from runs.{runs_name}')
        log(f'Number of training steps: {steps}')
        log(f'Sequence length T: {T}')
        return models, steps, T
    except ImportError as e:
        raise ImportError(f"Could not import runs.{runs_name}. Error: {e}")
    except AttributeError as e:
        raise AttributeError(f"runs.{runs_name} does not have a 'models' attribute. Error: {e}")


def train(name, attn_class, attn_args, train_loader, val_loader, T, n_layers, steps, runs_name):
    log(f'\n----Training {name}----')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
        num_classes=1,  # Binary classification
        pooling='mean'
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)

    model.train()
    step_idx = 0
    running_loss = 0.0
    running_acc = 0.0
    
    while step_idx < steps:
        for features, targets, masks in train_loader:
            if step_idx >= steps:
                break
                
            features = features.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            # Check for NaN/Inf
            if torch.isnan(features).any() or torch.isinf(features).any():
                log(f"WARNING: NaN/Inf in features at step {step_idx}")
                continue
            
            if masks.sum(dim=1).min() == 0:
                log(f"WARNING: Empty sequence at step {step_idx}")
                continue
            
            optimizer.zero_grad()
            logits, loss = model(features, targets, attn_mask=masks)
            
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
                log(f"WARNING: NaN gradients at step {step_idx}")
                optimizer.zero_grad()
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            
            # Compute accuracy
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == targets).float().mean().item()
            
            running_loss += loss.item()
            running_acc += acc
            step_idx += 1

            if step_idx % 500 == 0:
                avg_loss = running_loss / 500
                avg_acc = running_acc / 500
                current_lr = scheduler.get_last_lr()[0]
                log(f"{name}: step {step_idx}/{steps} | loss {avg_loss:.4f} | acc {avg_acc:.4f} | lr {current_lr:.2e}")
                running_loss = 0.0
                running_acc = 0.0
                
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step_idx}.pt')
                torch.save({
                    'step': step_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'acc': avg_acc,
                }, checkpoint_path)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return model


def evaluate_classification(model, data_loader, device):
    """Evaluate classification model and return metrics."""
    model.eval()
    all_logits = []
    all_targets = []
    losses = []
    
    with torch.no_grad():
        for features, targets, masks in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            logits, loss = model(features, targets, attn_mask=masks)
            
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            losses.append(loss.item())
    
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    all_preds = (all_probs > 0.5).astype(float)
    
    avg_loss = np.mean(losses)
    
    # Compute metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5  # If only one class present
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }


def run_all_models(runs_name):
    models, steps, T = load_runs(runs_name)
    log(f'Running {runs_name} with {len(models)} models')
    log('models:', models)

    results = {}
    for name, attn_class, attn_args, n_layers in models:
        model = train(name, attn_class, attn_args, train_loader, val_loader, T, n_layers, steps, runs_name)
        
        # Evaluate
        train_metrics = evaluate_classification(model, train_loader, device)
        val_metrics = evaluate_classification(model, val_loader, device)
        test_metrics = evaluate_classification(model, test_loader, device)
        
        log(f"\n{name} - Training Metrics:")
        log(f"  Loss: {train_metrics['loss']:.4f}")
        log(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        log(f"  Precision: {train_metrics['precision']:.4f}")
        log(f"  Recall: {train_metrics['recall']:.4f}")
        log(f"  F1: {train_metrics['f1']:.4f}")
        log(f"  AUC: {train_metrics['auc']:.4f}")
        
        log(f"\n{name} - Validation Metrics:")
        log(f"  Loss: {val_metrics['loss']:.4f}")
        log(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        log(f"  Precision: {val_metrics['precision']:.4f}")
        log(f"  Recall: {val_metrics['recall']:.4f}")
        log(f"  F1: {val_metrics['f1']:.4f}")
        log(f"  AUC: {val_metrics['auc']:.4f}")
        
        log(f"\n{name} - Test Metrics:")
        log(f"  Loss: {test_metrics['loss']:.4f}")
        log(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        log(f"  Precision: {test_metrics['precision']:.4f}")
        log(f"  Recall: {test_metrics['recall']:.4f}")
        log(f"  F1: {test_metrics['f1']:.4f}")
        log(f"  AUC: {test_metrics['auc']:.4f}")
        
        results[name] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        }
    
    # Summary table
    log("\n" + "=" * 60)
    log("SUMMARY - Test Metrics")
    log("=" * 60)
    log(f"{'Model':<30} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    log("-" * 60)
    for name in results:
        test = results[name]['test']
        log(f"{name:<30} {test['accuracy']:>10.4f} {test['f1']:>10.4f} {test['auc']:>10.4f}")
    
    return results


if __name__ == '__main__':
    log(f'Starting jet tagging training with runs.{args.runs}...')
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

