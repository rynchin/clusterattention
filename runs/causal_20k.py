"""Causal attention model configurations"""

from variants.MHA import MHA
from variants.LinearAttention import LinearAttention
from variants.FastCKA import FastCKA
from variants.LearnedClusterAttention import LearnedClusterAttention
from variants.ClusterAttention import ClusterAttention

# Sequence length - should match train.py
T = 512

# Number of training steps
steps = 20000

# each model is a tuple (name, attn_class, attn_args, n_layers)
models = [
    ('LinearAttention_l1', LinearAttention, {'eps': 1e-6, 'causal': True}, 1),
    ('FastCKA_l6_s1', FastCKA, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': True}, 6),
    ('FastCKA_l8_s1', FastCKA, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': True}, 8),
]

