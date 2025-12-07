Title: Subquadratic self-attention with clustering

# Introduction

A useful way to reframe self-attention is to interpret it as a graph neural network operating over the complete graph $G=(V,E)$ on $n=|V|$. Every token attends to every other token, which is exactly message passing on a complete graph with self-edges. This is quadratic in the number of tokens, as the number of messages passed is $O(E) = O(n^2)$. Several existing methods reduce this cost that can be categorized as altering the graph structure or the attention kernel. For example:

* **Linear attention.** Replace the softmax kernel with a feature map $\phi(\cdot)$ [2, 3]. Compute $(QK^\top)V$ as $Q(\phi(K)^\top V)$. Cost $O(n d^2)$ instead of $O(n^2 d)$. Quality depends on the kernel approximation.
* **Cluster attention** [6]. Group tokens into clusters. Compute attention inside each cluster plus attention between cluster summaries. Cost depends on the number of clusters $C$. Uses locality-sensitive hashing (LSH) to form clusters based on query-key similarity, achieving $O(n \log n)$ complexity. Reformer [4] combines LSH-based clustering with reversible layers and chunked feed-forward networks to reduce memory usage.
* **Sparse attention.** Replace the complete graph with a sparse adjacency pattern. Sparse Transformers [5] use fixed strided patterns or learned sparsity masks. Longformer [7] and BigBird [8] extend this with sliding windows and global tokens. Cost becomes $O(n \sqrt{n})$ to $O(n)$ or better depending on the schedule.

This blog develops a sequence of architectures centered on clustering, motivated by graph neural networks. Our work culminates in two main directions:
 
1. `SuperClusterAttention` an attention mechanism that restricts self-attention to within learned clusters and uses supernodes to route information between clusters.
2. `ClusterKernelAttention`, a hybrid that combines linear attention on top of cluster structure. We apply these architectures to three domains: language modeling (enwik8), physics regression (HEP events), and 3D object recognition (modelnet). 

We hypothesize that these proposed techniques are particularly effective when the underlying data exhibit latent cluster structure, making clustering an appropriate inductive bias.


# Proposed Architectures

## SuperClusterAttention

SuperClusterAttention replaces full self-attention with a two–stage process that first routes information through a small set of cluster “supernodes” and then performs local attention inside each cluster. The steps are:

0. Tokens are assigned to clusters
1. Each cluster produces a supernode, defined by a learned aggregation of its members. The $9$ supernodes attend to each other (a 
C×C
C×C complete graph). The resulting messages are then broadcast back to tokens using the same cluster weights
3. Each cluster runs self-attention within itself. 

Every token receives local information from its cluster and global information routed through its supernode, which is 

### Cluster Assignment

To remain permutation-equivariant with respect to token order, we must use a permutation-invariant rule to assign tokens to clusters. A simple approach based on sorting token scores gives contiguous blocks in $O(n \log⁡ n)$, but sorting is non-differentiable and breaks end-to-end training of the cluster projection.

1. Project each token to a scalar $s_i = w^\top x_i$.
2. Sort tokens by $s_i$.
3. Split the sorted sequence into $k=\sqrt{n}$ contiguous blocks.

However, the sorting and partitioning step is not differentiable, making backprop unable to reach the learned cluster embedding. Instead, we use the "straight through trick" from the literature:

1. Project each token to C dimensions, softmax, to get soft cluster assignments
2. Argmax to get hard cluster assignments.
3. Let `R` be the one hot cluster assignments. Use straight through trick: `R = R_hard.detach() - R_soft.detach() + R_soft`. This works because `R` is correlated with `R_soft`. TODO: cite

### Supernode construction

Each cluster produces one supernode. We define its feature vector as a learned linear pool over the cluster:

$$ u_j = \sum_{i \in C_j} \alpha_{ij} x_i $$

with $\alpha$ coming from the cluster assignment step.

### Remarks
* This technique is not compatible with causal masking.
* In our benchmarks, we define two ablations:
  +  `LearnedClusterAttention`, which is this idea but without supernodes (i.e. no inter-cluster communication).
  +  `ClusterAttention`, which uses our initial sort-based partition idea. The cluster assignment remains random each block.

## ClusterKernelAttention
**TODO**: Ryan
// leave empty for now


### FastCKA variant
// leave empty for now

### Remarks
// leave empty for now

# Experiments

We evaluate our cluster-based attention architectures on three diverse tasks: character-level language modeling, high-energy physics jet tagging, and 3D object recognition. These tasks differ in sequence length, structure, and supervision: language modeling uses long 1D byte sequences with causal dependencies, jet tagging uses short variable-length physics events with full-context classification, and ModelNet40 uses fixed-size 3D point clouds with geometric invariances.

## Language Modeling (enwik8)

We evaluate our architectures on character-level language modeling using the enwik8 dataset, which consists of the first 100 million bytes of Wikipedia. The task is to predict the next character given the previous context, making this a causal (autoregressive) prediction problem.

**Dataset details:**
- Training set: 90M bytes
- Validation set: 5M bytes  
- Test set: 5M bytes
- Vocabulary size: 256 (byte-level, one token per byte)
- Sequence length: $T = 512$ tokens per batch

**Model architecture:**
- Input: sequences of byte tokens $(x_1, \ldots, x_T)$
- Output: probability distribution over next token $P(x_{t+1} | x_{\leq t})$
- Loss: cross-entropy over vocabulary
- Metric: **bits-per-byte (bpb)** — lower is better, computed as $\text{bpb} = \text{CE} / \log(2)$ where CE is the cross-entropy loss

**Training:**
- Batch size: 32
- Learning rate: $3 \times 10^{-4}$
- Random sequence sampling: each batch samples random 512-token windows from the corpus
- Models trained for up to 50,000 steps (with some ablations at 20,000 steps)

**Architectures tested:**
- MHA: Multi-head attention baseline
- LinearAttention: Linear attention with feature map
- ClusterAttention: Hard clustering with sort-based partition
- LCA: Learned cluster attention with soft assignments
- CKA: ClusterKernelAttention (hybrid clustering + linear attention)
- FastCKA: Optimized variant of CKA

## High-Energy Physics Jet Tagging

We apply our architectures to a high-energy physics jet tagging task. Each event contains a variable-length sequence of detected particles, and the goal is to classify jets as either gluon-initiated or quark-initiated (binary classification).

**Dataset details:**
- Synthetic particle physics events with realistic distributions
- Variable sequence lengths: 50-500 particles per event
- Training: 50,000 events
- Validation: 10,000 events
- Test: 10,000 events
- Maximum sequence length: $T = 128$ (sequences are padded or truncated)

**Input features per particle:**
Each particle is represented by 6 features:
- $p_T$: transverse momentum (exponential distribution, GeV)
- $\eta$: pseudorapidity (uniform in $[-2.5, 2.5]$)
- $\phi$: azimuthal angle (uniform in $[0, 2\pi]$)
- mass: particle mass (mostly light particles: pions, kaons)
- charge: electric charge ($-1, 0, +1$)
- PID: particle ID encoding (0-5 for photon, electron, muon, pion, kaon, proton)

**Model architecture:**
- Input: $(B, T, 6)$ tensor of particle features (padded sequences)
- Attention mask: $(B, T)$ boolean mask indicating real particles vs padding
- Processing: transformer layers process the sequence with noncausal attention (full event context available)
- Pooling: mean pooling over sequence (masked) to get event-level representation
- Output: $(B, 1)$ binary classification logits
- Loss: binary cross-entropy

**Metrics:**
- **Accuracy**: Overall classification accuracy
- **F1**: F1 score (harmonic mean of precision and recall)
- **AUC**: Area under the ROC curve

**Training:**
- Batch size: 32
- Learning rate: $3 \times 10^{-4}$
- Weight decay: 0.1
- Dropout: 0.2
- Cosine annealing learning rate scheduler
- Models trained for 20,000 steps

## 3D Object Recognition (ModelNet40)

We evaluate our architectures on 3D object classification using the ModelNet40 dataset, which contains 40 categories of 3D objects represented as point clouds.

**Dataset details:**
- ModelNet40: 40 object categories
- Point clouds: 1024 points per object
- Features: 3D coordinates $(x, y, z)$
- Training/validation/test split: standard ModelNet40 splits

**Model architecture:**
- Input: $(B, 1024, 3)$ tensor of point coordinates
- Attention mask: $(B, 1024)$ boolean mask (all ones for fixed-size point clouds)
- Processing: transformer layers process the sequence with noncausal attention
- Pooling: mean pooling over sequence to get object-level representation
- Output: $(B, 40)$ classification logits
- Loss: cross-entropy over 40 classes

**Metrics:**
- **Accuracy**: Overall classification accuracy
- **Mean Class Accuracy**: Average per-class accuracy (handles class imbalance)

**Training:**
- Batch size: 32
- Learning rate: $1 \times 10^{-4}$
- Weight decay: 0.01
- Models trained for 20,000 steps

# Results

## Language Modeling Results

We evaluate our architectures on the enwik8 character-level language modeling task. All models are trained with causal masking, using the same hyperparameters (batch size 32, learning rate $3 \times 10^{-4}$).

### Main Results (50,000 steps)

Table 1 shows validation bits-per-byte (bpb) for different architectures across varying layer depths:

| Architecture | Layers | Val bpb | Notes |
|--------------|--------|---------|-------|
| MHA | 1 | 2.13 | Baseline |
| MHA | 4 | 1.56 | Baseline |
| MHA | 8 | 1.48 | Baseline (best) |
| LinearAttention | 1 | 2.30 | Linear kernel |
| LinearAttention | 4 | 1.74 | Linear kernel |
| LinearAttention | 8 | 1.61 | Linear kernel |
| CKA | 1 | 2.29 | Cluster + linear |
| CKA | 4 | 1.72 | Cluster + linear |
| CKA | 8 | 1.58 | Cluster + linear |


CKA shows minimal improvement over linear attention

### Cluster Scale Ablation (20,000 steps)

We investigate how the number of clusters of `ClusterAttention` affects performance by varying the cluster scale parameter $s$ where $C = s \cdot \sqrt{n}$. Results for 2-layer models:

| Cluster Scale | C (approx) | ClusterAttention Val bpb | LCA Val bpb | CKA Val bpb |
|---------------|------------|--------------------------|-------------|-------------|
| 1 | ~23 | 3.72 | 3.71 | 2.36 |
| 2 | ~45 | 3.59 | - | 2.39 |
| 4 | ~91 | 3.41 | 3.42 | 2.34 |
| 6 | ~136 | 3.24 | - | - |
| 8 | ~181 | 3.09 | - | 2.38 |
| 16 | ~362 | 2.54 | - | - |
| 32 | ~724 | 1.88 | - | - |
| 64 | ~1448 | 1.90 | - | - |


## High-Energy Physics Jet Tagging Results

We evaluate our architectures on the HEP jet tagging task. All models are trained for 20,000 steps with noncausal attention (full event context available). Table 2 shows test set performance metrics:

| Model | Layers | Test Loss | Test Acc | Test F1 | Test AUC |
|-------|--------|-----------|----------|---------|----------|
| ClusterAttention | 2 | 0.5788 | 0.6954 | 0.6763 | 0.7694 |
| SuperClusterAttention | 2 | 0.5810 | 0.6927 | 0.6659 | 0.7702 |
| LCA | 2 | 0.5826 | 0.6903 | 0.6678 | 0.7648 |
| MHA | 2 | 0.5884 | 0.6892 | 0.6697 | 0.7658 |
| SuperClusterAttention | 4 | 0.5817 | 0.6923 | 0.6692 | 0.7677 |
| ClusterAttention | 4 | 0.5849 | 0.6904 | 0.6680 | 0.7656 |
| MHA | 4 | 0.5904 | 0.6931 | 0.6755 | 0.7652 |
| LCA | 4 | 0.5849 | 0.6846 | 0.6633 | 0.7621 |
| LinearAttention | 2 | 0.6019 | 0.6742 | 0.6655 | 0.7384 |
| FastCKA | 2 | 0.6026 | 0.6740 | 0.6653 | 0.7384 |
| FastCKA | 4 | 0.6035 | 0.6751 | 0.6660 | 0.7381 |
| LinearAttention | 4 | 0.6040 | 0.6754 | 0.6663 | 0.7379 |

## ModelNet40 Results

We evaluate our architectures on the ModelNet40 3D object classification task. All models are trained for 20,000 steps with noncausal attention. Table 3 shows test set performance metrics:

| Model | Layers | Test Loss | Test Acc | Test MeanClassAcc |
|-------|--------|-----------|----------|-------------------|
| LinearAttention | 4 | 1.2072 | 0.7804 | 0.7355 |
| LinearAttention | 2 | 1.2941 | 0.7549 | 0.6702 |
| FastCKA | 2 | 1.3013 | 0.7342 | 0.6594 |
| FastCKA | 4 | 1.6009 | 0.7358 | 0.6794 |
| ClusterAttention | 4 | 1.7576 | 0.7451 | 0.6941 |
| ClusterAttention | 2 | 1.7780 | 0.7293 | 0.6666 |
| MHA | 4 | 2.5582 | 0.7261 | 0.6588 |
| SuperClusterAttention | 4 | 2.1732 | 0.7034 | 0.6383 |
| SuperClusterAttention | 2 | 2.1059 | 0.6868 | 0.6255 |
| LCA | 4 | 2.2155 | 0.7054 | 0.6391 |
| LCA | 2 | 2.6562 | 0.6827 | 0.6066 |
| MHA | 2 | 3.1728 | 0.6787 | 0.6150 |

# Conclusion
Hello let's talk about stuff.


# References

1. Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.
2. Katharopoulos, A., et al. "Transformers are RNNs: Fast autoregressive transformers with linear attention." ICML 2020.
3. Choromanski, K., et al. "Rethinking attention with performers." ICLR 2021.
4. Kitaev, N., Kaiser, Ł., & Levskaya, A. "Reformer: The efficient transformer." ICLR 2020.
5. Child, R., Gray, S., Radford, A., & Sutskever, I. "Generating long sequences with sparse transformers." arXiv:1904.10509, 2019.
6. Rae, J. W., & Razavi, A. "Do transformers need deep long-range memory?" arXiv:2007.04825, 2020.
7. Beltagy, I., Peters, M. E., & Cohan, A. "Longformer: The long-document transformer." arXiv:2004.05150, 2020.
8. Zaheer, M., et al. "Big bird: Transformers for longer sequences." NeurIPS 2020.
