import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.TransformerLayer import TransformerLayer


class TransformerClassifier(nn.Module):
    """Transformer model for binary classification on particle sequences."""
    
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
        num_classes: int = 1,
        pooling: str = 'mean'
    ):
        """
        Args:
            feature_dim: Input feature dimension per particle
            dim: Model dimension
            heads: Number of attention heads
            ffdim: Feed-forward dimension
            T: Maximum sequence length
            n_layers: Number of transformer layers
            attn_class: Attention class to use
            attn_args: Arguments for attention class
            num_classes: Number of output classes (1 for binary with BCE, 2+ for multi-class)
            pooling: Pooling method ('mean', 'cls', 'max')
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.feature_dim = feature_dim
        self.T = T
        self.num_classes = num_classes
        self.pooling = pooling

        # Project input features to model dimension
        self.feature_proj = nn.Linear(feature_dim, dim)
        
        # Positional encoding (learned)
        self.pos_emb = nn.Embedding(T, dim)
        
        # Optional CLS token for classification
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Transformer layers
        layers = []
        for _ in range(n_layers):
            attn_module = attn_class(dim=dim, heads=heads, **attn_args)
            layers.append(TransformerLayer(dim, heads, ffdim, attn_module))
        self.layers = nn.ModuleList(layers)

        # Final layer norm
        self.ln_f = nn.LayerNorm(dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, num_classes)
        )

    def forward(self, x, targets=None, attn_mask=None):
        """
        Args:
            x: (B, T, feature_dim) particle features
            targets: (B,) classification targets (optional)
            attn_mask: (B, T) boolean mask, True for real particles, False for padding
        
        Returns:
            If targets is None: (B, num_classes) logits
            If targets is not None: (B, num_classes) logits, scalar loss
        """
        B, T_seq, _ = x.shape
        assert T_seq <= self.T, f"Sequence length {T_seq} exceeds maximum {self.T}"

        # Project features
        h = self.feature_proj(x)  # (B, T_seq, dim)
        
        # Add CLS token if using cls pooling
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
            h = torch.cat([cls_tokens, h], dim=1)  # (B, T_seq+1, dim)
            T_seq = T_seq + 1
            
            # Extend attention mask for CLS token
            if attn_mask is not None:
                cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
                attn_mask = torch.cat([cls_mask, attn_mask], dim=1)
        
        # Add positional encoding
        pos_idx = torch.arange(0, T_seq, device=x.device)
        h = h + self.pos_emb(pos_idx).unsqueeze(0)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)

        # Final layer norm
        h = self.ln_f(h)

        # Pool sequence representation
        if self.pooling == 'cls':
            h_pooled = h[:, 0, :]  # CLS token
        elif self.pooling == 'mean':
            if attn_mask is not None:
                mask = attn_mask.unsqueeze(-1).float()
                mask_sum = mask.sum(dim=1)
                mask_sum = torch.clamp(mask_sum, min=1.0)
                h_pooled = (h * mask).sum(dim=1) / mask_sum
            else:
                h_pooled = h.mean(dim=1)
        elif self.pooling == 'max':
            if attn_mask is not None:
                # Set padding positions to very negative for max pooling
                mask = attn_mask.unsqueeze(-1).float()
                h_masked = h * mask + (1 - mask) * (-1e9)
                h_pooled = h_masked.max(dim=1)[0]
            else:
                h_pooled = h.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Classification head
        logits = self.classifier(h_pooled)  # (B, num_classes)
        
        if self.num_classes == 1:
            logits = logits.squeeze(-1)  # (B,) for binary classification

        if targets is None:
            return logits

        # Compute loss
        if self.num_classes == 1:
            # Binary classification with BCE
            loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        else:
            # Multi-class classification with CE
            loss = F.cross_entropy(logits, targets.long())
        
        return logits, loss


if __name__ == '__main__':
    # Test the model
    B, T, feature_dim = 4, 64, 6
    dim, heads, ffdim = 128, 4, 512
    
    from variants.MHA import MHA
    
    model = TransformerClassifier(
        feature_dim=feature_dim,
        dim=dim,
        heads=heads,
        ffdim=ffdim,
        T=T,
        n_layers=2,
        attn_class=MHA,
        attn_args={'causal': False},
        num_classes=1,
        pooling='mean'
    )
    
    # Create dummy data
    x = torch.randn(B, T, feature_dim)
    targets = torch.randint(0, 2, (B,)).float()
    attn_mask = torch.ones(B, T, dtype=torch.bool)
    attn_mask[:, 50:] = False
    
    logits, loss = model(x, targets, attn_mask)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Predicted probabilities: {torch.sigmoid(logits)}")

