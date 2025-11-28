import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.TransformerBlock import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(self, dim, heads, ffdim, V, T, attn_module: nn.Module):
        super().__init__()
        self.block = TransformerBlock(dim, heads, ffdim, V, T, attn=attn_module)
        self.lm_head = nn.Linear(dim, V)
    
    def forward(self, x, targets=None):
        # x: (B,T)
        h = self.block(x) # B,T,dim
        logits = self.lm_head(h) # B,T,V
        
        if targets is None:
            return logits
        
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) # (B*T,V) -> (B*T,)
        return logits, loss