import torch
import torch.nn as nn
import torch.nn.functional as F

from .LearnedClusterAttention import LearnedClusterAttention

class SuperClusterAttention(LearnedClusterAttention):
    def __init__(self, dim, heads, T, cluster_scale=1.0, tau=1.0, causal=False):
        if causal:
            raise ValueError("SuperClusterAttention requires causal=False. Causal attention is not supported.")
        super().__init__(dim, heads, T, cluster_scale, tau, causal)
        
        # supernode projections
        self.WQ_s = nn.Linear(dim, dim)
        self.WK_s = nn.Linear(dim, dim)
        self.WV_s = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        # attn_mask: (B, T) boolean mask, True for real tokens, False for padding
        B,T,dim = x.shape
        assert T == self.T
        
        # cluster assignments
        logits = self.cluster_proj(x) # BTC
        
        # Apply attention mask to cluster assignments
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1).float()  # (B, T, 1)
            logits = logits * mask + (1 - mask) * (-1e9)
        
        soft_assign = F.softmax(logits / self.tau, dim=-1) # BTC
        idx = soft_assign.argmax(dim=-1) # BT
        cluster_probs = soft_assign.sum(dim=1, keepdim=False)  # BC
        
        # supernodes
        S = torch.einsum('btc,btd->bcd', soft_assign, x) # BCD
        S = S / (cluster_probs.unsqueeze(-1) + 1e-8)  # BCD, normalize by cluster probabilities
        Qs = self.WQ_s(S) #BCD
        Ks = self.WK_s(S) #BCD
        Vs = self.WV_s(S) #BCD

        logits_s = torch.einsum('bcd,bkd->bck', Qs, Ks)/(dim ** 0.5) #BCC
        
        # Safety: if all positions are masked for a query, set uniform attention
        # This prevents NaN in softmax when all logits are -inf
        # Check if any valid (non-inf) position exists for each query
        has_valid = torch.isfinite(logits_s).any(dim=-1, keepdim=True)  # (B, C, 1)
        # If no valid positions, set to uniform (zeros before softmax = uniform after softmax)
        logits_s = torch.where(has_valid, logits_s, torch.zeros_like(logits_s))
        
        score_s = torch.softmax(logits_s, dim = -1) #BCC
        
        # Check for NaN in attention scores (safety check)
        if torch.isnan(score_s).any():
            # Replace NaN with uniform distribution
            C = logits_s.shape[-1]
            score_s = torch.where(torch.isnan(score_s), torch.ones_like(score_s) / C, score_s)
        
        out_s = torch.einsum('bck,bkd->bcd', score_s, Vs) # BCD
        
        # broadcast supernode info to tokens
        info = torch.einsum("btc,bcd->btd", soft_assign, out_s) # BTD
        x_aug = x + info # BTD
        
        # Use parent class token attention with augmented input
        return self._token_attention(x_aug, soft_assign, idx, attn_mask=attn_mask)

if __name__ == '__main__':
    x = torch.randn(2,20,128)
    y = SuperClusterAttention(128, 8, 20)
    print(y(x).shape)