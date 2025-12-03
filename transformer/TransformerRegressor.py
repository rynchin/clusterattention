import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.TransformerLayer import TransformerLayer


class TransformerRegressor(nn.Module):
    """Transformer model for regression on particle sequences."""
    
    def __init__(
        self,
        feature_dim: int,
        dim: int,
        heads: int,
        ffdim: int,
        T: int,
        n_layers: int,
        attn_class,
        attn_args: dict,
        target_dim: int = 4,
        pooling: str = 'mean'
    ):
        """
        Args:
            feature_dim: Input feature dimension per particle (e.g., 6 for pT, eta, phi, mass, charge, pid)
            dim: Model dimension
            heads: Number of attention heads
            ffdim: Feed-forward dimension
            T: Maximum sequence length
            n_layers: Number of transformer layers
            attn_class: Attention class to use
            attn_args: Arguments for attention class
            target_dim: Output dimension (e.g., 4 for missing momentum px, py, pz, E)
            pooling: Pooling method ('mean', 'pt_weighted', 'last')
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.feature_dim = feature_dim
        self.T = T
        self.target_dim = target_dim
        self.pooling = pooling

        # Project input features to model dimension
        self.feature_proj = nn.Linear(feature_dim, dim)
        
        # Positional encoding (learned)
        self.pos_emb = nn.Embedding(T, dim)

        # Transformer layers
        layers = []
        for _ in range(n_layers):
            attn_module = attn_class(dim=dim, heads=heads, **attn_args)
            layers.append(TransformerLayer(dim, heads, ffdim, attn_module))
        self.layers = nn.ModuleList(layers)

        # Final layer norm
        self.ln_f = nn.LayerNorm(dim)

        # Regression head
        self.reg_head = nn.Linear(dim, target_dim)

    def forward(self, x, targets=None, attn_mask=None):
        """
        Args:
            x: (B, T, feature_dim) particle features
            targets: (B, target_dim) regression targets (optional)
            attn_mask: (B, T) boolean mask, True for real particles, False for padding
        
        Returns:
            If targets is None: (B, target_dim) predictions
            If targets is not None: (B, target_dim) predictions, scalar loss
        """
        B, T_seq, _ = x.shape
        assert T_seq <= self.T, f"Sequence length {T_seq} exceeds maximum {self.T}"

        # Project features
        h = self.feature_proj(x)  # (B, T_seq, dim)
        
        # Add positional encoding
        pos_idx = torch.arange(0, T_seq, device=x.device)  # (T_seq,)
        h = h + self.pos_emb(pos_idx).unsqueeze(0)  # (B, T_seq, dim)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)  # (B, T_seq, dim)

        # Final layer norm
        h = self.ln_f(h)  # (B, T_seq, dim)

        # Pool sequence representation
        if self.pooling == 'mean':
            # Mean pooling with mask
            if attn_mask is not None:
                mask = attn_mask.unsqueeze(-1).float()  # (B, T_seq, 1)
                h_pooled = (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B, dim)
            else:
                h_pooled = h.mean(dim=1)  # (B, dim)
        elif self.pooling == 'pt_weighted':
            # Weight by pT (first feature)
            if attn_mask is not None:
                mask = attn_mask.unsqueeze(-1).float()  # (B, T_seq, 1)
                pt_weights = x[:, :, 0].unsqueeze(-1) * mask  # (B, T_seq, 1) - pT is first feature
                pt_weights = pt_weights / (pt_weights.sum(dim=1, keepdim=True) + 1e-8)
                h_pooled = (h * pt_weights).sum(dim=1)  # (B, dim)
            else:
                pt_weights = x[:, :, 0].unsqueeze(-1)  # (B, T_seq, 1)
                pt_weights = pt_weights / (pt_weights.sum(dim=1, keepdim=True) + 1e-8)
                h_pooled = (h * pt_weights).sum(dim=1)  # (B, dim)
        elif self.pooling == 'last':
            # Use last token (if masked, use last real token)
            if attn_mask is not None:
                # Find last real token for each sequence
                lengths = attn_mask.sum(dim=1).long() - 1  # (B,)
                lengths = torch.clamp(lengths, min=0)
                h_pooled = h[torch.arange(B, device=x.device), lengths]  # (B, dim)
            else:
                h_pooled = h[:, -1, :]  # (B, dim)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Regression head
        predictions = self.reg_head(h_pooled)  # (B, target_dim)

        if targets is None:
            return predictions

        # Compute loss (MSE)
        loss = F.mse_loss(predictions, targets)
        return predictions, loss


if __name__ == '__main__':
    # Test the model
    B, T, feature_dim = 4, 128, 6
    dim, heads, ffdim = 256, 8, 1024
    target_dim = 4
    
    from variants.MHA import MHA
    
    model = TransformerRegressor(
        feature_dim=feature_dim,
        dim=dim,
        heads=heads,
        ffdim=ffdim,
        T=T,
        n_layers=2,
        attn_class=MHA,
        attn_args={'causal': False},
        target_dim=target_dim,
        pooling='mean'
    )
    
    # Create dummy data
    x = torch.randn(B, T, feature_dim)
    targets = torch.randn(B, target_dim)
    attn_mask = torch.ones(B, T, dtype=torch.bool)
    attn_mask[:, 100:] = False  # Some padding
    
    predictions, loss = model(x, targets, attn_mask)
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Loss: {loss.item():.4f}")

