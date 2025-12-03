import torch
import torch.nn as nn
import torch.nn.functional as F

from .LearnedClusterAttention import LearnedClusterAttention, gumbel_softmax

class SuperClusterAttention(LearnedClusterAttention):
    def __init__(self, dim, heads, T, cluster_scale=1.0, tau=1.0, causal=False):
        if causal:
            raise ValueError("SuperClusterAttention requires causal=False. Causal attention is not supported.")
        super().__init__(dim, heads, T, cluster_scale, tau, causal)
        
        # supernode projections
        self.WQ_s = nn.Linear(dim, dim)
        self.WK_s = nn.Linear(dim, dim)
        self.WV_s = nn.Linear(dim, dim)

    def forward(self, x):
        B,T,dim = x.shape
        assert T == self.T
        
        # cluster assignments
        logits = self.cluster_proj(x) # BTC
        soft_assign, idx = gumbel_softmax(logits, self.tau) # BTC, BT
        cluster_probs = soft_assign.sum(dim=1, keepdim=False)  # BC
        
        # supernodes
        S = torch.einsum('btc,btd->bcd', soft_assign, x) # BCD
        S = S / (cluster_probs.unsqueeze(-1) + 1e-8)  # BCD, normalize by cluster probabilities
        Qs = self.WQ_s(S) #BCD
        Ks = self.WK_s(S) #BCD
        Vs = self.WV_s(S) #BCD

        logits_s = torch.einsum('bcd,bkd->bck', Qs, Ks)/(dim ** 0.5) #BCC
        score_s = torch.softmax(logits_s, dim = -1) #BCC
        out_s = torch.einsum('bck,bkd->bcd', score_s, Vs) # BCD
        
        # broadcast supernode info to tokens
        info = torch.einsum("btc,bcd->btd", soft_assign, out_s) # BTD
        x_aug = x + info # BTD
        
        # Use parent class token attention with augmented input
        return self._token_attention(x_aug, soft_assign, idx)

if __name__ == '__main__':
    x = torch.randn(2,20,128)
    y = SuperClusterAttention(128, 8, 20)
    print(y(x).shape)