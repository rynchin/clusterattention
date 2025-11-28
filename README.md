# Cluster Attention

Small Transformer language model for comparing three attention mechanisms:

- **MHA**: standard causal softmax attention
- **LinearAttention**: causal kernelized attention
- **ClusterAttention**: block-based √N clustered attention

## Usage

Train all variants with:

```bash
python train.py
```

## Structure

- `transformer/` - Transformer blocks and language model
- `variants/` - Attention implementations (MHA, LinearAttention, ClusterAttention)
- `train.py` - Training script against Bigram model
