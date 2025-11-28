import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformer.TransformerLM import TransformerLM
from variants.MHA import MHA
from variants.LinearAttention import LinearAttention
from variants.ClusterAttention import ClusterAttention

device = 'cuda' if torch.cuda.is_available() else 'cpu'

V = 1000
T = 64
dim = 128
heads = 8
ffdim = 512
batch_size = 32
steps = 500

# --- bigram model ---
@torch.no_grad()
def bigram_model(V, device):
    # markov chain (order 1), returns probability matrix
    logits = torch.randn(V, V, device=device)
    probs = F.softmax(logits, dim=-1)
    return probs

bigram_probs = bigram_model(V, device)

def get_batch(batch_size, T, V, device):
    # random sample from bigram + teacher forcing
    data = torch.zeros(batch_size, T+1, dtype=torch.long, device=device)
    
    # initial
    data[:, 0] = torch.randint(0, V, (batch_size,), device=device)
    for t in range(T):
        prev = data[:, t] # B,
        probs = bigram_probs[prev]
        data[:, t+1] = torch.multinomial(probs, num_samples=1).squeeze(-1) # B,

    x = data[:, :-1]
    y = data[:, 1:]
    return x, y

def train(name, attn_class, attn_args):
    print(f'\n----Training {name}----')
    attn_module = attn_class(dim=dim, heads=heads, **attn_args).to(device)
    model = TransformerLM(dim=dim, heads=heads, ffdim=ffdim, V=V, T=T, attn_module=attn_module).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    for step in range(steps):
        x,y = get_batch(batch_size, T, V, device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        if (step+1) % 50 == 0:
            print(f'{name}: Step {step+1}/{steps} | Loss {loss.item():.4f}')
            
    return model

if __name__ == '__main__':
    models = [
        ('MHA', MHA, {}),
        ('LinearAttention', LinearAttention, {'eps': 1e-6}),
        ('ClusterAttention', ClusterAttention, {'cluster_scale': 1.0}),
    ]
    for name, attn_class, attn_args in models:
        train(name, attn_class, attn_args)