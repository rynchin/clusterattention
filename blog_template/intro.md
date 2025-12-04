Title: Subquadratic self-attention with clustering

# Introduction

Self-attention can be viewed as message passing on a complete directed graph with self-loops. With $n$ tokens you have $n^2$ edges and an $O(n^2)$ cost. Several existing methods reduce this cost by altering the graph structure or the attention kernel. For example:

* **Performer: Linear attention.** Replace the softmax kernel with a feature map $\phi(\cdot)$. Compute $(QK^\top)V$ as $Q(\phi(K)^\top V)$. Cost $O(n d^2)$ instead of $O(n^2 d)$. Quality depends on the kernel approximation.
* **Cluster attention** [Rae & Razavi, 2020]. Group tokens into clusters. Compute attention inside each cluster plus attention between cluster summaries. Cost depends on the number of clusters $C$. Uses locality-sensitive hashing (LSH) to form clusters based on query-key similarity, achieving $O(n \log n)$ complexity.
* **Sparse attention** [Child et al., 2019]. Replace the complete graph with a sparse adjacency pattern. Sparse Transformers use fixed strided patterns or learned sparsity masks. Longformer and BigBird extend this with sliding windows and global tokens. Cost becomes $O(n \sqrt{n})$ to $O(n)$ or better depending on the schedule.
* **Reformer** [Kitaev et al., 2020]. Combines LSH-based clustering with reversible layers and chunked feed-forward networks. Uses LSH attention where queries and keys are hashed into buckets, and attention is computed only within buckets. Achieves $O(n \log n)$ complexity and reduces memory usage through reversible residual connections.

This blog develops a sequence of architectures built around clustering, motivated by graph neural networks. We introduce two directions: (i) `ClusterKernelAttention`, a hybrid that combines linear attention on top of cluster structure, and (ii) `SuperClusterAttention`: purely clustered models that rely on supernodes to summarize and route information between learned clusters.

# Proposed Architectures

## ClusterKernelAttention

`ClusterKernelAttention` is a hybrid architecture that combines the benefits of clustering with linear (kernelized) attention. The key insight is to use clustering to structure the computation, then apply efficient linear attention within and between clusters, avoiding the quadratic cost of standard softmax attention while maintaining expressiveness.

### Architecture overview

The method operates in three main stages:

1. **Soft cluster assignment**: Assign tokens to clusters using learned projections
2. **Kernelized attention within clusters**: Use linear attention with a feature map $\phi$ to compute cluster summaries
3. **Low-rank mixing between clusters**: Efficiently mix information across clusters using low-rank matrices

### Cluster assignment

Similar to `LearnedClusterAttention`, we project each token $x_i$ to cluster logits:

$$ \text{logits}_i = W_{\text{cluster}} x_i \in \mathbb{R}^C $$

where $C \approx \sqrt{n}$ is the number of clusters. We then compute soft assignments:

$$ \alpha_{ic} = \text{softmax}(\text{logits}_i / \tau)_c $$

where $\tau$ is a temperature parameter. This gives us a soft membership matrix $\alpha \in \mathbb{R}^{n \times C}$ where each token belongs partially to each cluster.

### Kernelized attention with clustering

Instead of computing full attention, we use a feature map $\phi(x) = \text{ELU}(x) + 1$ to enable linear attention. For each cluster $c$, we aggregate kernelized key-value pairs:

$$ K_c = \sum_{i=1}^n \alpha_{ic} \phi(K_i), \quad KV_c = \sum_{i=1}^n \alpha_{ic} \phi(K_i) \otimes V_i $$

where $K_i = W_K x_i$ and $V_i = W_V x_i$ are the key and value projections. For causal attention, we use prefix sums; for noncausal, we sum over all tokens.

### Low-rank cluster mixing

The key efficiency gain comes from mixing information between clusters. Naively, computing attention between all $C$ clusters would cost $O(C^2)$. Instead, we use low-rank mixing matrices $A, B \in \mathbb{R}^{C \times k}$ where $k \ll C$ (typically $k=8$):

First, we compress the $C$ cluster summaries into rank-$k$ space:
$$ \tilde{K}_\ell = \sum_{j=1}^C A_{j\ell} K_j \quad \text{for } \ell \in \{1, \ldots, k\} $$

Then we project back to cluster space:
$$ \tilde{K}_c = \sum_{\ell=1}^k B_{c\ell} \tilde{K}_\ell = \sum_{\ell=1}^k B_{c\ell} \sum_{j=1}^C A_{j\ell} K_j $$

This reduces the mixing cost from $O(C^2)$ to $O(Ck)$, where $k$ is the mixing rank (typically $k=8$ or $k=32$). The matrices $A$ and $B$ are learned parameters (constrained to be positive via softplus) initialized with small random values. This low-rank factorization allows clusters to communicate efficiently while maintaining expressiveness.

### Final attention computation

For each token $i$, we compute its output by:
1. Projecting the mixed cluster states back to token space using its cluster assignments
2. Applying kernelized attention: $\text{output}_i = Q_i \cdot \frac{\sum_c \alpha_{ic} KV_c}{\sum_c \alpha_{ic} K_c}$

### Computational complexity

The total cost is:
- Cluster assignment: $O(nC) = O(n\sqrt{n})$
- Per-cluster aggregation: $O(nr)$ where $r$ is the feature dimension (typically $r=32$)
- Low-rank mixing: $O(Ck)$ where $k \ll C$
- Final projection: $O(nC)$

**Total: $O(n\sqrt{n} + nr + Ck)$** which is $O(n\sqrt{n})$ when $r$ and $k$ are treated as constants.

### FastCKA variant

`FastCKA` is an optimized variant that reorders operations to fuse computations more efficiently, achieving the same asymptotic complexity but with better constant factors. The key difference is that it projects cluster assignments into the low-rank space earlier, reducing intermediate tensor sizes.

### Advantages

- **Subquadratic cost**: $O(n\sqrt{n})$ instead of $O(n^2)$
- **Differentiable**: Soft cluster assignments allow gradients to flow
- **Flexible**: Works for both causal and noncausal attention
- **No kernel approximation error**: Unlike pure linear attention, clustering preserves local structure

### Limitations

- Requires knowing sequence length $T$ at initialization (for cluster count $C$)
- Cluster assignments may not align perfectly with semantic boundaries
- Low-rank mixing may limit expressiveness for very complex inter-cluster interactions

## SuperClusterAttention

We interpret the transformer block as a GNN over the complete graph $G=(V,E)$. Partition $V$ into $C$ cliques and add a new set of representatives (supernodes). Each supernode aggregates information from its clique and then communicates with other supernodes through a reduced complete graph. Subsequent inter-clique attention for each clique allows information sharing between cliques. Choosing $C \approx \sqrt{n}$ to balance costs, yielding $O(n \sqrt{n})$ runtime, as we will show below.

### Partition step

In order for self-attention to be permutation equivariant, we must use a permutation-invariant rule to assign tokens to clusters. One $O(n \log n)$ sorting-based mechanism we came up with was:

1. Project each token to a scalar $s_i = w^\top x_i$.
2. Sort tokens by $s_i$.
3. Split the sorted sequence into $k=\sqrt{n}$ contiguous blocks.

However, the sorting and partitioning step is not differentiable, so we instead tried

1. Project each token to C dimensions, softmax, to get soft cluster assignments
2. Argmax to get hard cluster assignments.
3. Let `R` be the one hot cluster assignments. Use straight through trick: `R = R_hard.detach() - R_soft.detach() + R_soft`. This works because `R` is correlated with `R_soft`.

### Supernode construction

Each clique produces one supernode. The simplest choice is a learned linear pool:

$$ u_j = \sum_{i \in C_j} \alpha_{ij} x_i $$

with $\alpha$ coming from intra-clique attention.

### Intra-clique attention

Tokens in a clique attend only to their supernode instead of all other tokens. This replaces each local dense subgraph with a star shaped pattern mediated by the supernode. Cost: $O(C (N/C)^2) = O(N^2/C)$.

### Inter-clique (supernode) attention

Supernodes attend to one another through a complete graph. Size is $C$ so the cost is $O(C^2)$. TODO: mention Causal mask cannot be enforced with supernodes.

This yields a noncausal attention block with subquadratic cost and no approximation of the softmax kernel.
