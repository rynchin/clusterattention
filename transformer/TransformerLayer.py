import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, ffdim, attn_module: nn.Module, dropout=0.0):
        super().__init__()
        self.attn = attn_module  # expects (B,T,dim) -> (B,T,dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        self.fc1 = nn.Linear(dim, ffdim)
        self.fc2 = nn.Linear(ffdim, dim)


    def forward(self, h, attn_mask=None):
        # h: (B,T,dim)
        # attn_mask: (B, T) boolean mask, True for real tokens, False for padding
        y = self.ln1(h) # layernorm

        out = self.attn(y, attn_mask=attn_mask)  # B,T,dim
        out = self.dropout(out)  # Apply dropout

        # residual
        h = h + out # positional encoding
        
        z = self.ln2(h)
        m = self.fc2(F.gelu(self.fc1(z)))
        m = self.dropout(m)  # Apply dropout
        
        h = h + m # residual
        return h
