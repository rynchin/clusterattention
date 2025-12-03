"""Causal attention model configurations"""

from variants.MHA import MHA
from variants.LinearAttention import LinearAttention
from variants.ClusterKernelAttention import ClusterKernelAttention

# Sequence length - should match train.py
T = 512

# Number of training steps
steps = 50000

# each model is a tuple (name, attn_class, attn_args, n_layers)
models = [
    # 50k runs, causal
    ('CKA_l1_s1_nonmixing', ClusterKernelAttention, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': True, 'mixing': False}, 1),
]

