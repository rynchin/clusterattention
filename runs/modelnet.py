"""ModelNet40 point cloud classification model configurations"""

from variants.MHA import MHA
from variants.LinearAttention import LinearAttention
from variants.FastCKA import FastCKA
from variants.LearnedClusterAttention import LearnedClusterAttention
from variants.ClusterAttention import ClusterAttention
from variants.SuperClusterAttention import SuperClusterAttention

# Sequence length - number of points per point cloud
T = 1024

# Number of training steps
steps = 20000

# each model is a tuple (name, attn_class, attn_args, n_layers)
# Note: causal=False for point cloud data (no sequential ordering)
models = [
    # Standard multi-head attention (baseline)
    ('MHA_l2', MHA, {'causal': False}, 2),
    ('MHA_l4', MHA, {'causal': False}, 4),
    
    # Linear attention (efficient baseline)
    ('LinearAttention_l2', LinearAttention, {'eps': 1e-6, 'causal': False}, 2),
    ('LinearAttention_l4', LinearAttention, {'eps': 1e-6, 'causal': False}, 4),
    
    # Fast Cluster Kernel Attention
    # sqrt(1024) = 32 clusters
    ('FastCKA_l2_s1', FastCKA, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': False}, 2),
    ('FastCKA_l4_s1', FastCKA, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'r': 32, 'causal': False}, 4),
    
    # Learned Cluster Attention
    ('LCA_l2_s1', LearnedClusterAttention, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'causal': False, 'force_one_cluster': False}, 2),
    ('LCA_l4_s1', LearnedClusterAttention, {'T': T, 'cluster_scale': 1.0, 'tau': 1.0, 'causal': False, 'force_one_cluster': False}, 4),
]

