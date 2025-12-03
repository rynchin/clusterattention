import torch
import torch.nn as nn
import torch.nn.functional as F

def elu_feature_map(x):
    return F.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, dim, heads, eps, causal=True):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.d = dim // heads
        self.eps = eps
        self.causal = causal
    
        self.WQ = nn.Linear(dim, dim)
        self.WK = nn.Linear(dim, dim)
        self.WV = nn.Linear(dim, dim)
        self.WO = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        # attn_mask: (B, T) boolean mask, True for real tokens, False for padding
        B,T,dim = x.shape
        # for elu R == D
        Q = elu_feature_map(self.WQ(x).reshape(B,T,self.heads, self.d).transpose(1,2)) # BHTR
        K = elu_feature_map(self.WK(x).reshape(B,T,self.heads, self.d).transpose(1,2)) # BHTR
        V = self.WV(x).reshape(B,T,self.heads, self.d).transpose(1,2) # BHTD
        
        # Apply attention mask: zero out padding positions
        if attn_mask is not None:
            # attn_mask: (B, T) -> (B, 1, T, 1) for broadcasting
            mask = attn_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, T, 1)
            K = K * mask  # Zero out padding keys
            V = V * mask  # Zero out padding values
        
        KV = torch.einsum('bhtr,bhtd->bhtrd', K, V) # outer product, BHTRD
        if self.causal:
            S = KV.cumsum(dim=2) # BHTRD
            Z = K.cumsum(dim=2).unsqueeze(-1) # BHTR1
        else:
            # For non-causal, sum over all positions and broadcast
            S_sum = KV.sum(dim=2, keepdim=True) # BH1RD
            Z_sum = K.sum(dim=2, keepdim=True).unsqueeze(-1) # BH1R1
            S = S_sum.expand(-1, -1, T, -1, -1) # BHTRD
            Z = Z_sum.expand(-1, -1, T, -1, -1) # BHTR1
        
        num = torch.einsum('bhtr,bhtrd->bhtd', Q, S) # BHTD
        den = torch.einsum('bhtr,bhtrd->bhtd', Q, Z) # BHT1
        out = num/(den + self.eps)
        return self.WO(out.transpose(1,2).reshape(B,T,dim))

if __name__ == '__main__':
    x = torch.randn(2,20,128)
    y = LinearAttention(128, 8, 1e-6, causal=False)
    print(y(x).shape)