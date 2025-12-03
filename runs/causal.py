"""Causal attention model configurations"""

from variants.MHA import MHA
from variants.LinearAttention import LinearAttention
from variants.FastCKA import FastCKA

# Sequence length - should match train.py
T = 512

# Number of training steps
steps = 50000

# each model is a tuple (name, attn_class, attn_args, n_layers)
models = [
    # 50k runs, causal
    ('LinearAttention_l1', LinearAttention, {'eps': 1e-6, 'causal': True}, 1),
    ('LinearAttention_l4', LinearAttention, {'eps': 1e-6, 'causal': True}, 4),
    ('LinearAttention_l8', LinearAttention, {'eps': 1e-6, 'causal': True}, 8),

    ('FastCKA_l1_s1', FastCKA, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': True}, 1),
    ('FastCKA_l4_s1', FastCKA, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': True}, 4),
    ('FastCKA_l8_s1', FastCKA, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': True}, 8),

    ('MHA_l1', MHA, {'causal': True}, 1),
    ('MHA_l4', MHA, {'causal': True}, 4),
    ('MHA_l8', MHA, {'causal': True}, 8),
]

