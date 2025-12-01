import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(logits, tau=1.0, eps=1e-9):
    # logits: B,T,c
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps) # sample Gumbel noise
    y_soft = F.softmax((logits + gumbel) / tau, dim=-1) # BTc
    idx = y_soft.argmax(dim=-1, keepdim=True) # BT1
    y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0) # BTc
    y = y_hard.detach() - y_soft.detach() + y_soft # straight through trick
    return y, idx.squeeze(-1) # BTc, BT
    
class LearnedClusterAttention(nn.Module):
    def __init__(self, dim, heads, T, cluster_scale=1.0, tau=1.0):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.d = dim // heads
        self.cluster_scale = cluster_scale
        self.tau = tau
        self.T = T
    
        s = int(cluster_scale * T**0.5)
        s = max(1, min(s, T))
        self.num_clusters = (T + s - 1) // s
    
        self.WQ = nn.Linear(dim, dim)
        self.WK = nn.Linear(dim, dim)
        self.WV = nn.Linear(dim, dim)
        self.WO = nn.Linear(dim, dim)
        
        self.cluster_proj = nn.Linear(dim, self.num_clusters)

    def forward(self, x):
        B,T,dim = x.shape
        assert T == self.T
        
        # project
        Q = self.WQ(x).reshape(B,T,self.heads, self.d).transpose(1,2) # BHTD
        K = self.WK(x).reshape(B,T,self.heads, self.d).transpose(1,2)
        V = self.WV(x).reshape(B,T,self.heads, self.d).transpose(1,2)

        logits = self.cluster_proj(x) # BTc
        soft_assign, idx = gumbel_softmax(logits, self.tau) # BTc, BT
    
        R_soft = torch.einsum('btc,buc->btu', soft_assign, soft_assign)
        R_hard = (idx.unsqueeze(-1)==idx.unsqueeze(-2)).float() # BTT (same cluster assignments)
        R = R_hard.detach() - R_soft.detach() + R_soft # BTT
        
        ar = torch.arange(T, device=x.device)
        causal = (ar[None,:] <= ar[:,None]).float() # TT
        R = R * causal # BTT
        
        logits = torch.einsum('bhtd,bhkd->bhtk', Q, K)/(self.d ** 0.5) # BHTT
        logits = logits.masked_fill(R.unsqueeze(1)==0, float('-inf'))
        
        score = torch.softmax(logits, dim = -1) # BHTT
        out = torch.einsum('bhtk,bhkd->bhtd', score, V) # BHTD
        out = out.transpose(1,2).reshape(B,T,dim)
        return self.WO(out)

x = torch.randn(2,20,128)
y = LearnedClusterAttention(128, 8, 20)
print(y(x).shape)