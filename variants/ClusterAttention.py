import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterAttention(nn.Module):
    def __init__(self, dim, heads, cluster_scale=1.0):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.d = dim // heads
        self.cluster_scale = cluster_scale
        
        self.WQ = nn.Linear(dim, dim)
        self.WK = nn.Linear(dim, dim)
        self.WV = nn.Linear(dim, dim)
        self.WO = nn.Linear(dim, dim)
        
        self.cluster_proj = nn.Linear(dim, 1)

    def forward(self, x):
        B,T,dim = x.shape
        s = int(self.cluster_scale * (T ** 0.5))
        if s < 1:
            s = 1
        if s > T:
            s = T
            
        num_clusters = (T + s - 1) // s
        
        # project
        Q = self.WQ(x).reshape(B,T,self.heads, self.d).transpose(1,2) # BHTD
        K = self.WK(x).reshape(B,T,self.heads, self.d).transpose(1,2)
        V = self.WV(x).reshape(B,T,self.heads, self.d).transpose(1,2)

        scores = self.cluster_proj(x).squeeze(-1) # BT
        sorted_scores, idx = torch.sort(scores, dim = -1) # idx contains original indices
        
        # inverse index to unsort
        inv_idx = torch.zeros_like(idx) # BT
        arange = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        inv_idx.scatter_(1, idx, arange)
        
        # reorder along dim 2
        idx_shape = idx[:, None, :, None].expand(B, self.heads, T, self.d) # broadcast
        Qs = torch.gather(Q, 2, idx_shape) # BHTD
        Ks = torch.gather(K, 2, idx_shape)
        Vs = torch.gather(V, 2, idx_shape)
        
        out_s = torch.empty_like(Qs) # BHTD
        for i in range(num_clusters):
            start = i * s
            end = min(T, (i+1)*s)
            Qc = Qs[:, :, start:end, :]
            Kc = Ks[:, :, start:end, :]
            Vc = Vs[:, :, start:end, :]
            
            pos = idx[:, start:end]

            # causal wrt initial pos
            future = (pos[:,None,:] > pos[:,:,None]).bool() # Bcc
            causal_mask = future[:,None,:,:] # B1cc
            
            att = torch.einsum('bhtd,bhkd->bhtk', Qc, Kc)/(self.d ** 0.5) # BHcc
            att = att.masked_fill(causal_mask, float('-inf'))
            att = torch.softmax(att, dim = -1)
            out_c = torch.einsum('bhtk,bhkd->bhtd', att, Vc)
        
            out_s[:, :, start:end, :] = out_c
            
        # unsort along dim 2
        inv_idx_shape = inv_idx[:, None, :, None].expand(B, self.heads, T, self.d)
        out = torch.gather(out_s, 2, inv_idx_shape) # BHTD
        out = out.transpose(1,2).reshape(B,T,dim)
        return self.WO(out)

x = torch.randn(2,20,128)
y = ClusterAttention(128, 8)
print(y(x).shape)