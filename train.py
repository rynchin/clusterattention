import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import argparse
import importlib
from datetime import datetime

from transformer.TransformerLM import TransformerLM

# Setup logging to file with current date/time
parser = argparse.ArgumentParser(description='Train attention models')
parser.add_argument('--runs', type=str, required=True,
                    help='Name of the runs file to import (e.g., causal, noncausal)')
args = parser.parse_args()
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = open(f'logs/train_{timestamp}_{args.runs}.log', 'w')

def log(*args, **kwargs):
    """Write to log file instead of printing"""
    message = ' '.join(str(arg) for arg in args)
    if kwargs:
        message += ' ' + ' '.join(f'{k}={v}' for k, v in kwargs.items())
    log_file.write(message + '\n')
    print(message)
    log_file.flush()  # Ensure immediate write

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Enwiki8 dataset ---
def load_enwik8(path="enwik8"):
    if path is None:
        path = os.environ.get("ENWIK8_PATH", "enwik8")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "rb") as f:
        data = f.read()
    data = torch.tensor(list(data), dtype=torch.long)
    return data

log("Loading enwik8...")
full_data = load_enwik8(None)
N = full_data.size(0)
log("Total bytes:", N)

train_data = full_data[:90_000_000] # 90M
val_data   = full_data[90_000_000:95_000_000] # 5M
test_data  = full_data[95_000_000:] # 5M

V = 256
T = 512
dim = 256
heads = 8
ffdim = 4 * dim
batch_size = 32

lr = 3e-4
weight_decay = 0.01
grad_clip = 1.0

def get_batch(source, batch_size, T, device):
    N = source.size(0)
    idx = torch.randint(0, N - T - 1, (batch_size,)) # random start indices
    x = torch.stack([source[i:i+T] for i in idx]) # B,T
    y = torch.stack([source[i+1:i+T+1] for i in idx]) # B,T
    return x.to(device), y.to(device) # B,T

def train(name, attn_class, attn_args, data, T, n_layers, steps, runs_name):
    log(f'\n----Training {name}----')
    # Clear CUDA cache before creating new model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create checkpoint directory for this model
    checkpoint_dir = f'models/{runs_name}/{name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    log(f'Checkpoints will be saved to {checkpoint_dir}')
    
    model = TransformerLM(dim=dim, heads=heads, ffdim=ffdim, V=V, T=T, n_layers=n_layers, attn_class=attn_class, attn_args=attn_args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for step_idx in range(1, steps + 1):
        x, y = get_batch(data, batch_size, T, device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

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

def evaluate_bpb(model, data, T, num_batches=200):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, batch_size, T, device)
            _, loss = model(x, y)
            losses.append(loss.item())
    avg_ce = sum(losses)/len(losses)
    bpb = avg_ce/math.log(2.0) # nats to bits
    return bpb

def load_runs(runs_name):
    """Load models and steps from a runs file"""
    try:
        runs_module = importlib.import_module(f'runs.{runs_name}')
        models = runs_module.models
        steps = getattr(runs_module, 'steps', 50000)  # Default to 50000 if not specified
        log(f'Loaded {len(models)} models from runs.{runs_name}')
        log(f'Number of training steps: {steps}')
        return models, steps
    except ImportError as e:
        raise ImportError(f"Could not import runs.{runs_name}. Make sure the file exists in the runs/ directory. Error: {e}")
    except AttributeError as e:
        raise AttributeError(f"runs.{runs_name} does not have a 'models' attribute. Error: {e}")

def run_all_models(runs_name):
    models, steps = load_runs(runs_name)
    log(f'Running {runs_name} with {len(models)} models')
    log('models:', models)

    results = {}
    for name, attn_class, attn_args, n_layers in models:
        model = train(name, attn_class, attn_args, train_data, T, n_layers, steps, runs_name)
        train_bpb = evaluate_bpb(model, train_data, T, num_batches=100)
        val_bpb = evaluate_bpb(model, val_data, T, num_batches=100)
        log(f"{name}: train bpb={train_bpb:.4f} | val bpb={val_bpb:.4f}")
        results[name] = {
            'train_bpb': float(train_bpb),
            'val_bpb': float(val_bpb),
        }
    return results

if __name__ == '__main__':
    log(f'Starting training with runs.{args.runs}...')
    try:
        run_all_models(runs_name=args.runs)
        log('Training completed successfully!')
    except Exception as e:
        log(f'Error during training: {e}')
        raise
    finally:
        log_file.close()