"""Causal attention model configurations"""

from variants.MHA import MHA
from variants.LinearAttention import LinearAttention
from variants.FastCKA import FastCKA
from variants.LearnedClusterAttention import LearnedClusterAttention
from variants.ClusterAttention import ClusterAttention

# Sequence length - should match train.py
T = 512

# Number of training steps
steps = 50000

# each model is a tuple (name, attn_class, attn_args, n_layers)
models = [
    # 50k runs, causal
    ('LinearAttention_l2', LinearAttention, {'eps': 1e-6, 'causal': True}, 1),
    ('LinearAttention_l4', LinearAttention, {'eps': 1e-6, 'causal': True}, 4),
    ('LinearAttention_l8', LinearAttention, {'eps': 1e-6, 'causal': True}, 8),

    ('FastCKA_l2_s1', FastCKA, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': True}, 2),
    ('FastCKA_l4_s1', FastCKA, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': True}, 4),
    # ('FastCKA_l8_s1', FastCKA, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': True}, 8),

    ('MHA_l2', MHA, {'causal': True}, 2),
    ('MHA_l4', MHA, {'causal': True}, 4),
    # ('MHA_l8', MHA, {'causal': True}, 8),

     # # Learned Cluster Attention
    ('LCA_l2_s1', LearnedClusterAttention, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'causal': False, 'force_one_cluster': False}, 2),
    ('LCA_l4_s1', LearnedClusterAttention, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'causal': False, 'force_one_cluster': False}, 4),

    ('ClusterAttention_l2_s1', ClusterAttention, {'cluster_scale': 1.0, 'causal': False}, 2),
    ('ClusterAttention_l4_s1', ClusterAttention, {'cluster_scale': 1.0, 'causal': False}, 4),
]

