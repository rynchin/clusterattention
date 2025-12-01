import torch
import torch.nn as nn
import torch.nn.functional as F

def phi(x):
    return F.elu(x) + 1

class FastCKA(nn.Module):
    def __init__(self, dim, heads, T, cluster_scale=1.0, tau=1.0, r=32, mix_rank=8):
        super().__init__()
        assert dim % heads == 0

        self.dim = dim
        self.heads = heads
        self.d = dim // heads
        self.T = T
        self.r = r
        self.tau = tau
        self.mix_rank = mix_rank

        s = int(cluster_scale * T**0.5)
        s = max(1, min(s, T))
        self.C = (T + s - 1) // s

        H = heads
        D = dim
        C = self.C
        k = mix_rank

        self.WQ = nn.Linear(D, H * r)
        self.WK = nn.Linear(D, H * r)
        self.WV = nn.Linear(D, D)
        self.WO = nn.Linear(D, D)

        self.cluster_proj = nn.Linear(D, C)  # cluster logits

        # low rank mixing params for cluster space
        self.MA = nn.Parameter(torch.randn(C, k) * 0.02)
        self.MB = nn.Parameter(torch.randn(C, k) * 0.02)

    def forward(self, x):
        B, T, D = x.shape
        assert T == self.T

        C, H, r, d = self.C, self.heads, self.r, self.d
        k = self.mix_rank

        logits = self.cluster_proj(x) # BTc
        assign = F.softmax(logits / self.tau, dim=-1) # BTc

        Q = phi(self.WQ(x).reshape(B, T, H, r)) # BTHr
        K = phi(self.WK(x).reshape(B, T, H, r))
        V = self.WV(x).reshape(B, T, H, d) # BTHd

        A = F.softplus(self.MA) # ck, positive
        Bmat = F.softplus(self.MB) # ck

        # collapse assignments into low-rank (reordered)
        A_bar = torch.einsum("btc,ck->btk", assign, A) # BTk
        B_bar = torch.einsum("btc,ck->btk", assign, Bmat) # BTk

        # token contributions in rank-k space
        token_K  = torch.einsum("btk,bthr->btkhr", A_bar, K) # BTkHr
        token_KV = torch.einsum("btk,bthr,bthd->btkhrd", A_bar, K, V) # BTkHrd

        # causal prefix sum
        tmp_K = torch.cumsum(token_K, dim=1) # BTkHr
        tmp_KV = torch.cumsum(token_KV, dim=1) # BTkHrd

        # project back to tokens
        Kf = torch.einsum("btkhr,btk->bthr", tmp_K, B_bar) # BTHr
        KVf = torch.einsum("btkhrd,btk->bthrd", tmp_KV, B_bar) # BTHrd

        # kernelized attention
        num = torch.einsum("bthr,bthrd->bthd", Q, KVf) # BTHd
        den = torch.einsum("bthr,bthr->bth", Q, Kf).unsqueeze(-1) # BTH1

        den = den.clamp(min=1e-4) # training stability fix

        h = num / den
        h = h.reshape(B, T, D) # BTD

        return self.WO(h)

x = torch.randn(2, 20, 128)
layer = FastCKA(dim=128, heads=8, T=20, r=32, mix_rank=8)
y = layer(x)
print(y.shape)