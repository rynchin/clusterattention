import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, ffdim, attn_module: nn.Module):
        super().__init__()
        self.attn = attn_module  # expects (B,T,dim) -> (B,T,dim)
        
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        self.fc1 = nn.Linear(dim, ffdim)
        self.fc2 = nn.Linear(ffdim, dim)


    def forward(self, h):
        # h: (B,T,dim)
        y = self.ln1(h) # layernorm

        out = self.attn(y)  # B,T,dim

        # residual
        h = h + out # positional encoding
        
        z = self.ln2(h)
        m = self.fc2(F.gelu(self.fc1(z)))
        
        h = h + m # residual
        return h
