import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(logits, tau=1.0, eps=1e-9):
    raise DeprecationWarning('Use ordinary softmax instead')
    # logits: B,T,C
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps) # sample Gumbel noise
    y_soft = F.softmax((logits + gumbel) / tau, dim=-1) # BTC
    idx = y_soft.argmax(dim=-1, keepdim=True) # BT1
    y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0) # BTc
    y = y_hard.detach() - y_soft.detach() + y_soft # straight through trick
    return y, idx.squeeze(-1) # BTC, BT
    
class LearnedClusterAttention(nn.Module):
    def __init__(self, dim, heads, T, cluster_scale=1.0, tau=1.0, causal=True, force_one_cluster=False):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.d = dim // heads
        self.cluster_scale = cluster_scale
        self.tau = tau
        self.T = T
        self.causal = causal
        self.force_one_cluster = force_one_cluster
    
        s = int(cluster_scale * T**0.5)
        s = max(1, min(s, T))
        self.num_clusters = (T + s - 1) // s
    
        self.WQ = nn.Linear(dim, dim)
        self.WK = nn.Linear(dim, dim)
        self.WV = nn.Linear(dim, dim)
        self.WO = nn.Linear(dim, dim)
        
        # Only create cluster projection if not forcing one cluster
        if not self.force_one_cluster:
            self.cluster_proj = nn.Linear(dim, self.num_clusters)

    def _token_attention(self, x, soft_assign=None, idx=None):
        """Token-level cluster-masked attention. If force_one_cluster=True, soft_assign and idx are ignored."""
        B,T,dim = x.shape
        
        # project
        Q = self.WQ(x).reshape(B,T,self.heads, self.d).transpose(1,2) # BHTD
        K = self.WK(x).reshape(B,T,self.heads, self.d).transpose(1,2)
        V = self.WV(x).reshape(B,T,self.heads, self.d).transpose(1,2)
    
        logits = torch.einsum('bhtd,bhkd->bhtk', Q, K)/(self.d ** 0.5) # BHTT
        
        # If forcing one cluster, skip cluster masking and just apply causal mask if needed
        if self.force_one_cluster:
            if self.causal:
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
                logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # Use cluster-based masking
            R_soft = torch.einsum('btc,buc->btu', soft_assign, soft_assign)
            R_hard = (idx.unsqueeze(-1)==idx.unsqueeze(-2)).float() # BTT (same cluster assignments)
            # straight through trick
            R = R_hard.detach() - R_soft.detach() + R_soft # BTT
            
            if self.causal:
                ar = torch.arange(T, device=x.device)
                causal_mask = (ar[None,:] <= ar[:,None]).float() # TT
                R = R * causal_mask # BTT
            
            logits = logits.masked_fill(R.unsqueeze(1)==0, float('-inf'))
        
        score = torch.softmax(logits, dim = -1) # BHTT
        out = torch.einsum('bhtk,bhkd->bhtd', score, V) # BHTD
        out = out.transpose(1,2).reshape(B,T,dim)
        return self.WO(out)

    def forward(self, x):
        B,T,dim = x.shape
        assert T == self.T
        
        # If forcing one cluster, skip cluster assignment and use ordinary attention
        if self.force_one_cluster:
            return self._token_attention(x, soft_assign=None, idx=None)
        
        # Otherwise, compute cluster assignments
        logits = self.cluster_proj(x) # BTC
        soft_assign = F.softmax(logits, dim=-1) # BTC
        idx = soft_assign.argmax(dim=-1) # BT
        
        return self._token_attention(x, soft_assign, idx)

if __name__ == '__main__':
    x = torch.randn(2,20,128)
    y = LearnedClusterAttention(128, 8, 20, force_one_cluster=True)
    print(y(x).shape)