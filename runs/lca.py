"""Learned Cluster Attention model configurations"""

from variants.LearnedClusterAttention import LearnedClusterAttention

# Sequence length - should match train.py
T = 512

# Number of training steps
steps = 20000

# each model is a tuple (name, attn_class, attn_args, n_layers)
models = [
     ('LCA_l1s1', LearnedClusterAttention, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'causal': True, 'force_one_cluster': False}, 1),
]

