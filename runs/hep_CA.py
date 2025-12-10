"""Jet tagging (quark vs gluon) classification model configurations"""

from variants.MHA import MHA
from variants.LinearAttention import LinearAttention
from variants.FastCKA import FastCKA
from variants.LearnedClusterAttention import LearnedClusterAttention
from variants.ClusterAttention import ClusterAttention
from variants.SuperClusterAttention import SuperClusterAttention

# Sequence length - maximum number of particles per jet
T = 128

# Number of training steps
steps = 20000

# each model is a tuple (name, attn_class, attn_args, n_layers)
# Note: causal=False for event-level data (full event context available)
models = [
    # Cluster Attention
    ('ClusterAttention_l2_s1_1', ClusterAttention, {'cluster_scale': 1.0, 'causal': False}, 2),
    ('ClusterAttention_l2_s1_2', ClusterAttention, {'cluster_scale': 1.0, 'causal': False}, 2),
    ('ClusterAttention_l2_s1_3', ClusterAttention, {'cluster_scale': 1.0, 'causal': False}, 2),
    ('ClusterAttention_l2_s1_4', ClusterAttention, {'cluster_scale': 1.0, 'causal': False}, 2),
    ('ClusterAttention_l2_s1_5', ClusterAttention, {'cluster_scale': 1.0, 'causal': False}, 2),
    ('ClusterAttention_l2_s1_6', ClusterAttention, {'cluster_scale': 1.0, 'causal': False}, 2),
]


