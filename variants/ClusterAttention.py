import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterAttention(nn.Module):
    def __init__(self, dim, heads, cluster_scale=1.0, causal=True):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.d = dim // heads
        self.cluster_scale = cluster_scale
        self.causal = causal
        
        self.WQ = nn.Linear(dim, dim)
        self.WK = nn.Linear(dim, dim)
        self.WV = nn.Linear(dim, dim)
        self.WO = nn.Linear(dim, dim)
        
        self.cluster_proj = nn.Linear(dim, 1)

    def forward(self, x, attn_mask=None):
        # attn_mask: (B, T) boolean mask, True for real tokens, False for padding
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
        
        # Apply attention mask: set padding positions to very negative scores
        if attn_mask is not None:
            scores = scores * attn_mask.float() + (1 - attn_mask.float()) * (-1e9)
        
        sorted_scores, idx = torch.sort(scores, dim = -1) # idx (BT) contains original indices
        
        # inverse index to unsort
        inv_idx = torch.zeros_like(idx) # BT
        arange = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        inv_idx.scatter_(1, idx, arange)
        
        # reorder along token dimension according to idx
        idx_shape = idx[:, None, :, None].expand(B, self.heads, T, self.d) # broadcast to BHTD
        Qs = torch.gather(Q, 2, idx_shape) # BHTD
        Ks = torch.gather(K, 2, idx_shape)
        Vs = torch.gather(V, 2, idx_shape)
        
        # Reorder attention mask if provided
        if attn_mask is not None:
            mask_reordered = torch.gather(attn_mask.unsqueeze(1).expand(-1, self.heads, -1), 2, idx.unsqueeze(1).expand(-1, self.heads, -1))
        else:
            mask_reordered = None
        
        out_s = torch.empty_like(Qs) # BHTD
        for i in range(num_clusters):
            start = i * s
            end = min(T, (i+1)*s)
            Qc = Qs[:, :, start:end, :]
            Kc = Ks[:, :, start:end, :]
            Vc = Vs[:, :, start:end, :]
            
            pos = idx[:, start:end]

            att = torch.einsum('bhtd,bhkd->bhtk', Qc, Kc)/(self.d ** 0.5) # BHss
            
            if self.causal:
                # Compute causal mask wrt initial pos
                # s = end - start = cluster size
                future = (pos[:,None,:] > pos[:,:,None]).bool() # Bss
                causal_mask = future[:,None,:,:] # B1ss
                att = att.masked_fill(causal_mask, float('-inf'))
            
            # Apply attention mask for padding
            if mask_reordered is not None:
                mask_c = mask_reordered[:, :, start:end]  # (B, H, s)
                mask_2d = mask_c.unsqueeze(2) & mask_c.unsqueeze(3)  # (B, H, s, s)
                att = att.masked_fill(~mask_2d, float('-inf'))
            
            # Safety: if all positions are masked for a query, set uniform attention
            # This prevents NaN in softmax when all logits are -inf
            # Check if any valid (non-inf) position exists for each query
            has_valid = torch.isfinite(att).any(dim=-1, keepdim=True)  # (B, H, s, 1)
            # If no valid positions, set to uniform (zeros before softmax = uniform after softmax)
            att = torch.where(has_valid, att, torch.zeros_like(att))
            
            att = torch.softmax(att, dim = -1)
            
            # Check for NaN in attention scores (safety check)
            if torch.isnan(att).any():
                # Replace NaN with uniform distribution
                s_cluster = end - start
                att = torch.where(torch.isnan(att), torch.ones_like(att) / s_cluster, att)
            
            out_c = torch.einsum('bhtk,bhkd->bhtd', att, Vc)
        
            out_s[:, :, start:end, :] = out_c
            
        # unsort along dim 2
        inv_idx_shape = inv_idx[:, None, :, None].expand(B, self.heads, T, self.d)
        out = torch.gather(out_s, 2, inv_idx_shape) # BHTD
        out = out.transpose(1,2).reshape(B,T,dim)
        return self.WO(out)

if __name__ == '__main__': 
    x = torch.randn(2,20,128)
    y = ClusterAttention(128, 8)
    print(y(x).shape)