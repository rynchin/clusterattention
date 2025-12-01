import torch
import torch.nn as nn
import torch.nn.functional as F

def phi(x):
    return F.elu(x) + 1

class ClusterKernelAttention(nn.Module):
    def __init__(self, dim, heads, T, cluster_scale=1.0, tau=1.0, r=32, mix_rank=8):
        super().__init__()
        assert dim % heads == 0

        self.dim = dim
        self.heads = heads
        self.d = dim // heads
        self.T = T
        self.r = r # feature dim
        self.tau = tau
        self.mix_rank = mix_rank # low-rank mixing dim

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

        self.cluster_proj = nn.Linear(D, C) # cluster logits

        # low rank mixing params
        self.MA = nn.Parameter(torch.randn(C, k) * 0.02)
        self.MB = nn.Parameter(torch.randn(C, k) * 0.02)

    def forward(self, x):
        B, T, D = x.shape
        assert T == self.T

        C, H, r, d = self.C, self.heads, self.r, self.d

        logits = self.cluster_proj(x) # BTc
        assign = F.softmax(logits / self.tau, dim=-1) # BTc

        Q = phi(self.WQ(x).reshape(B, T, H, r)) # BTHr
        K = phi(self.WK(x).reshape(B, T, H, r))
        V = self.WV(x).reshape(B, T, H, d) # BTHd

        delta_K = torch.einsum("btc,bthr->btchr", assign, K) # BTcHr, per cluster per timestep phi(k) 
        delta_KV = torch.einsum("btc,bthr,bthd->btchrd", assign, K, V) # BTcHrd, per cluster per timestep phi(k) * v

        Kc = torch.cumsum(delta_K, dim=1) # cumulative sum for causal
        KVc = torch.cumsum(delta_KV, dim=1)

        A = F.softplus(self.MA) # ck, positive
        Bmat = F.softplus(self.MB) # ck

        # low-rank mixing O(Ck) instead of O(C^2)
        temp_K = torch.einsum("btjhr,jk->btkhr", Kc, A) # BTkHr, rank k compression
        temp_KV = torch.einsum("btjhrd,jk->btkhrd", KVc, A)

        mix_K = torch.einsum("btkhr,ck->btchr", temp_K, Bmat) # BTcHr
        mix_KV = torch.einsum("btkhrd,ck->btchrd", temp_KV, Bmat)

        # project mix cluster states to tokens
        Kf = torch.einsum("btc,btchr->bthr", assign, mix_K) # BTHr
        KVf = torch.einsum("btc,btchrd->bthrd", assign, mix_KV) # BTHrd

        # kernelized attention
        num = torch.einsum("bthr,bthrd->bthd", Q, KVf) # BTHd
        den = torch.einsum("bthr,bthr->bth", Q, Kf).unsqueeze(-1) # BTH1

        den = den.clamp(min=1e-4) # training stability fix

        h = num / den
        h = h.reshape(B, T, D) # BTD

        return self.WO(h)


x = torch.randn(2, 20, 128)
layer = ClusterKernelAttention(dim=128, heads=8, T=20, r=32, mix_rank=8)
y = layer(x)
print(y.shape)
