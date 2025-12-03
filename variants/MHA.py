import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, dim, heads, causal=True):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.d = dim // heads
        self.causal = causal
        
        self.WQ = nn.Linear(dim, dim)
        self.WK = nn.Linear(dim, dim)
        self.WV = nn.Linear(dim, dim)
        self.WO = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        # B,T,dim
        # attn_mask: (B, T) boolean mask, True for real tokens, False for padding
        B,T,dim = x.shape
        Q = self.WQ(x).reshape(B,T,self.heads, self.d).transpose(1,2)
        K = self.WK(x).reshape(B,T,self.heads, self.d).transpose(1,2)
        V = self.WV(x).reshape(B,T,self.heads, self.d).transpose(1,2)
        
        logits = torch.einsum('bhtd,bhkd->bhtk', Q, K)/(self.d ** 0.5) # BHTT
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(torch.ones(T,T, dtype=torch.bool, device=x.device), diagonal=1)
            logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply attention mask (padding mask)
        if attn_mask is not None:
            # attn_mask: (B, T) -> expand to (B, H, T, T)
            # Mask out positions where either query or key is padding
            mask_q = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            mask_k = attn_mask.unsqueeze(1).unsqueeze(3)  # (B, 1, T, 1)
            mask = mask_q & mask_k  # (B, 1, T, T)
            mask = mask.expand(-1, self.heads, -1, -1)  # (B, H, T, T)
            logits = logits.masked_fill(~mask, float('-inf'))  # (B, H, T, T)
        
        score = torch.softmax(logits, dim = -1) # BHTT
        out = torch.einsum('bhtk,bhkd->bhtd', score, V)
        out = out.transpose(1,2).reshape(B,T,dim)
        
        out = self.WO(out)
        return out

if __name__ == '__main__':
    x = torch.randn(2,20,128)
    y = MHA(128, 8)
    print(y(x).shape)