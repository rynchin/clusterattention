import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ffdim, V, T, attn: nn.Module):
        super().__init__()
    
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.d = dim // heads

        self.pos_emb = nn.Embedding(T, dim)
        self.tok_emb = nn.Embedding(V, dim)

        self.attn = attn  # expects (B,T,dim) -> (B,T,dim)
        
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        self.fc1 = nn.Linear(dim, ffdim)
        self.fc2 = nn.Linear(ffdim, dim)

    def forward(self, x):
        B, T = x.shape
        pos_idx = torch.arange(0, T, device=x.device) # T,
        h = self.pos_emb(pos_idx).unsqueeze(0) + self.tok_emb(x) # B,T,dim
        
        y = self.ln1(h) # layernorm

        out = self.attn(y)  # B,T,dim

        # residual
        h = h + out # positional encoding
        
        z = self.ln2(h)
        m = self.fc2(F.gelu(self.fc1(z)))
        
        h = h + m # residual
        return h
