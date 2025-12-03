"""Non-causal attention model configurations"""

from variants.MHA import MHA
from variants.LinearAttention import LinearAttention
from variants.FastCKA import FastCKA
from variants.SuperClusterAttention import SuperClusterAttention

# Sequence length - should match train.py
T = 512

# Number of training steps
steps = 50000

# each model is a tuple (name, attn_class, attn_args, n_layers)
models = [
    # noncausal
    ('MHA', MHA, {'causal': False}, 2),
    ('LinearAttention', LinearAttention, {'eps': 1e-6, 'causal': False}, 2),
    ('FastCKA_l2_s1', FastCKA, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': False}, 2),
    ('SuperClusterAttention', SuperClusterAttention, {'T': T, 'cluster_scale': 4.0, 'tau': 1.0, 'causal': False}, 2),
]

