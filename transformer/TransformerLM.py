import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.TransformerLayer import TransformerLayer

class TransformerLM(nn.Module):
    def __init__(self, dim, heads, ffdim, V, T, n_layers, attn_class, attn_args):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.V = V
        self.T = T

        self.tok_emb = nn.Embedding(V, dim)
        self.pos_emb = nn.Embedding(T, dim)

        layers = []
        for _ in range(n_layers):
            attn_module = attn_class(dim=dim, heads=heads, **attn_args)
            layers.append(TransformerLayer(dim, heads, ffdim, attn_module))
        self.layers = nn.ModuleList(layers)

        self.ln_f = nn.LayerNorm(dim)

        self.lm_head = nn.Linear(dim, V, bias=False)
        self.lm_head.weight = self.tok_emb.weight # tied weights

    def forward(self, x, targets=None):
        B, T = x.shape

        pos_idx = torch.arange(0, T, device=x.device)  # (T,)
        h = self.pos_emb(pos_idx).unsqueeze(0) + self.tok_emb(x)  # (B,T,dim)

        for layer in self.layers:
            h = layer(h)  # (B,T,dim)

        h = self.ln_f(h)
        logits = self.lm_head(h)  # (B,T,V)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) # (B*T,V) -> (B*T,)
        return logits, loss
