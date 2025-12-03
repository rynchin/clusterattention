# Introduction

Self-attention can be viewed as message passing on a complete directed graph with self-loops. With $n$ tokens you have $n^2$ edges and an $O(n^2)$ cost. Several existing methods reduce this cost by altering the graph structure or the attention kernel. For example:

* **Linear attention.** Replace the softmax kernel with a feature map $\phi(\cdot)$. Compute $(QK^\top)V$ as $Q(\phi(K)^\top V)$. Cost $O(n d^2)$ instead of $O(n^2 d)$. Quality depends on the kernel approximation.
* **Cluster attention.** Group tokens into clusters. Compute attention inside each cluster plus attention between cluster summaries. Cost depends on the number of clusters $C$. Apoorv et al. use locality-sensitive hashing to form clusters.
* **Sparse attention.** Replace the complete graph with a sparse adjacency pattern. Longformer, BigBird, and Sparse Transformers use fixed windows, dilations, or learned sparsity. Cost becomes $O(n \sqrt{n})$ or better depending on the schedule.

The blog develops a sequence of architectures built around clustering. We introduce two directions: (i) `ClusterKernelAttention`: hybrids that layer linearization on top of cluster structure, and `LearnedClusterAttention` and `SuperClusterAttention`: purely clustered models that rely on supernodes to summarize and route information. We apply these architectures to two domains: language modeling (enwik8) and physics regression (HEP events). 

---
# Proposed Architectures

## Idea 1: Two-Level Attention via Graph Partitioning

Interpret the transformer block as a GNN over the complete graph $G=(V,E)$. A compression strategy is to partition $V$ into $k$ cliques and add a new set of representatives (supernodes). Each supernode aggregates information from its clique and then communicates with other supernodes through a reduced complete graph. Choosing $k \approx \sqrt{n}$ to balance costs, yielding $O(n \sqrt{n})$ runtime.

### Partition step

Use a permutation-invariant rule to assign tokens to clusters. One simple mechanism:

1. Project each token to a scalar (s_i = w^\top x_i).
2. Sort tokens by (s_i).
3. Split the sorted sequence into (k=\sqrt{n}) contiguous blocks.

Cost: (O(n \log n)). Partition is differentiable through the projection but not through the sort unless you relax it. This can be treated as a hard grouping like routing in Mixture-of-Experts or the original Reformer.

### Supernode construction

Each clique produces one supernode. The simplest choice is a learned linear pool:

[
u_j = \sum_{i \in C_j} \alpha_{ij} x_i
]

with (\alpha) coming from intra-clique attention or uniform averaging. Number of supernodes: (k=\sqrt{n}). Cost: (O(n)).

### Intra-clique attention

Tokens in a clique attend only to their supernode instead of all other tokens. This replaces each local dense subgraph with a star shaped pattern mediated by the supernode. Cost: (O(n d)).

### Inter-clique (supernode) attention

Supernodes attend to one another through a complete graph. Size is (k) so the cost is (O(k^2 d)=O(n d)). This replaces the original (n^2) edges with (\sqrt{n}^2 = n) edges.

Total block cost:

[
O(n \log n) + O(n) + O(n) + O(n) = O(n \log n).
]

This gives a noncausal attention block with subquadratic cost and no approximation of the softmax kernel.

---

## Architectural Variants

Include variants to show the design space.

1. **Hard-clique version**
   Deterministic cluster assignment by sorting. Supernodes aggregate with uniform pooling or within-clique attention.

2. **Soft-cluster version**
   Replace the hard sort with soft routing (e.g. differentiable top-k or Sinkhorn-based assignment). Supernodes become mixtures of tokens. Cost increases but gradients flow reliably.

3. **Linearized supernode attention**
   Keep hard cliques but apply linear attention between supernodes. Reduces inter-clique cost to (O(n)) with a kernel map.

4. **Bidirectional architecture**
   Let tokens attend to both their supernode and a small fixed neighbor window. Adds local pattern similar to Longformer but retains global routing.

5. **Recurrent refinement**
   Iteratively recompute clusters after every block. This reduces partition error but costs more sorting steps.

6. **Causal adaptation**
   Replace the global sort with monotonic partitioning or causal chunking. Supernode communication becomes masked.

---

# Experiments
## Setup
Datasets:
enwik8: language modeling (causal/noncausal)
HEP events: regression (noncausal)
Metrics:
Language: bits-per-byte / perplexity
Regression: MAE, RMSE, R²
Baselines: MHA, LinearAttention
Variants tested: different layer counts, cluster scales
---

## Language modeling task
## Physics regression task


# Results
## Ablations

The minimal set of knobs:

1. **Number of clusters (k)**
   Study cost vs quality. Expect monotonic improvement toward MHA as (k\to n). Look for knees in the curve near (\sqrt{n}).

2. **Size of projection dimension for routing**
   Scalar vs multi-dimensional. Multi-dimensional routing avoids ties and improves diversity.

3. **Pooling type for supernodes**
   *Uniform.*
   *Attention pooling.*
   *Max pooling.*
   Measure the impact on information retention.

4. **Inter-clique attention type**
   Softmax vs linearized vs sparse. Linearized may preserve quality if (k) is already small.

5. **Use of token-token fallback edges**
   Add a small fraction of random or local edges. Compare quality to pure two-level routing.

6. **Causal vs noncausal**
   For language modeling evaluate the masked version.

7. **Parameter sharing**
   Share the projection that produces routing scores across layers vs learn it per layer.

---

If you want more detailed math, code examples, or figures, specify which section to expand.


# Conclusion


# References