Title: Subquadratic self-attention with clustering

# Introduction

A useful way to reframe self-attention is to interpret it as a graph neural network operating over the complete graph $G=(V,E)$ on $n=|V|$. Every token attends to every other token, which is exactly message passing on a complete graph with self-edges. This is quadratic in the number of tokens, as the number of messages passed is $O(E) = O(n^2)$. Several existing methods reduce this cost that can be categorized as altering the graph structure or the attention kernel. We summarize them below:


* **Linear attention.** Linear attention rewrites softmax attention using a positive feature map $\phi(\cdot)$ such that $(QK^\top)V$ becomes $\phi(Q)(\phi(K)^\top V)$ [1,2]. This reduces complexity to $O(n)$ but relies on the assumption that the softmax kernel can be well-approximated via an inner product. These approximations work well for some domains but can degrade performance when the kernel approximation is poor.
* **Cluster attention**. Cluster attention reduces computation by grouping tokens into clusters and operating primarily over cluster representatives. Reformer [3] uses LSH to bucket queries and keys and computes attention only within buckets, achieving $O(n \log n)$ complexity while assuming that hash buckets respect semantic locality. Fast Transformers with Clustered Attention [4] groups queries into $C$ clusters (e.g., via $k$-means), computes attention between cluster centroids and all keys, then broadcasts centroid outputs back to queries. This yields an $O(nC)$ method but relies on the assumption that the centroid provides a sufficiently informative summary of each cluster. 
* **Sparse attention.** Sparse Transformers [5] reduce attention cost by restricting each query to attend to only a small, predetermined subset of keys, for example fixed strided patterns or learned sparsity masks. Longformer [6] and BigBird [7] extend this with sliding windows and global tokens. They provide subquadratic cost but impose fixed sparsity patterns rather than learning them.


This blog develops a sequence of architectures centered on clustering. Our architectures build on linear and cluster attention as templates, but introduce different aggregation mechanisms (mixing clusters, supernodes) to obtain $O(n\sqrt{n})$ cost. Our work culminates in two main directions:
 
1. `SuperClusterAttention`: an attention mechanism that restricts self-attention to within learned clusters and uses supernodes to route information between clusters.
2. `ClusterKernelAttention`: which uses learned soft clusters and low-rank cluster mixing within linear attention to capture global context efficiently.

We apply these architectures to three domains: language modeling (enwik8), physics regression (HEP events), and 3D object recognition (ModelNet40). 

We treat clustering as a task-conditional inductive bias. We hypothesize that these techniques are most effective when the data exhibit stable latent group structure that benefits from explicit cross-cluster routing. Conversely, we expect limited gains or degradation when clustering signals are weak at short context lengths or when preserving fine-grained local geometry is critical.

# Proposed Architectures

## SuperClusterAttention (SCA)

SuperClusterAttention replaces full self-attention with a two–stage process that first routes information through a set of cluster “supernodes” and then performs local attention inside each cluster. The steps are:

1. Tokens are assigned to $C$ clusters.
2. Each cluster produces a supernode, defined by a learned aggregation of its members. The supernodes attend to each other (a $C×C$ complete graph). The resulting messages are then broadcast back to tokens using the same cluster weights
3. Each cluster runs self-attention within itself. 

Every token receives local information from its cluster and global information routed through its supernode, which creates a two-hop path (token -> supernode -> token) that replaces full token–token attention without paying the full $O(n^2)$ cost.

### Cluster Assignment

To remain permutation-equivariant with respect to token order, we must use a permutation-invariant rule to assign tokens to clusters. 

A simple permutation-invariant strategy is to project each token to a scalar, sort their scores, and then assign clusters by taking contiguous blocks of the sorted sequence. This produces deterministic non-overlapping clusters in $O(n \log n)$ time, but the sorting step is non-differentiable, so the assignments cannot be trained. We therefore treat this as a baseline (`RandomClusterAttention`) and replace it with differentiable soft assignments in our main models.

In our learned variants, we obtain differentiable cluster assignments using a straight-through estimator [8]:

1. Project each token into $C$ logits and apply a softmax to obtain soft cluster memberships `R_soft`.
2. Take an argmax over those logits to produce hard assignments `R_hard`.
3. Combine the two with a straight-through estimator:
   `R = R_hard.detach() - R_soft.detach() + R_soft`

This keeps clustering discrete in the forward pass while allowing gradients to flow through the soft assignments.


### Supernode construction
Each cluster produces a supernode that summarizes its tokens before participating in cluster-cluster attention. We define its feature vector as a learned linear pool over the cluster, with the softmax probabilities coming from `R_soft` in the cluster assignment step.

### LearnedClusterAttention and RandomClusterAttention Variants
- The supernode technique is not compatible with causal masking because it mixes information across all positions.
- For comparison, we include two ablations that do support causal masking:
  +  **LearnedClusterAttention (LCA)**, which keeps the learned cluster assignments but removes supernodes, preventing inter-cluster communication.
  +  **RandomClusterAttention (RCA)**, which uses the original sort-based cluster assignment procedure. Since this procedure is not differentiable, the assignments are effectively random.
- In all cases we treat each cluster as a fully connected subgraph (standard self-attention), though other intra-cluster graph structures can be explored and may be domain specific.

## ClusterKernelAttention (CKA)
`ClusterKernelAttention` replaces full token–token attention with kernel attention routed through soft clusters. Each token contributes its kernelized keys and values to the clusters it belongs to, clusters maintain running summaries, and tokens read back a cluster-conditioned context.

The procedure is as follows:
1. Tokens obtain soft cluster memberships through a learned projection and temperatured softmax.
2. Queries, keys, and values are mapped through a positive kernel feature map so linear attention applies.
3. Each cluster accumulates prefix sums of kernelized keys and key–value products (causal masking preserved).
4. Cluster states are mixed through a learned low-rank transformation so information flows across clusters without paying a full $C^2$ cost.
5. Tokens read back a mixed cluster summary using their soft assignments and form a standard linear-attention update.

For a more detailed mechanism and full code, refer to the Appendix and [GitHub repo](https://github.com/rynchin/clusterattention/blob/master/variants/ClusterKernelAttention.py).

### FastCKA variant
FastCKA implements the same update but reorganizes the algebra so that most operations happen in the reduced rank. This removes large intermediate cluster tensors and lowers matmul cost, resulting in an observed 2× training speed improvement while producing the same outputs.

### Remarks
* Soft assignments keep routing differentiable without straight-through. 
* The mixing rank $k$ controls how much cross-cluster information is exchanged.
* We utilize cluster mixing as sequences often contain latent groups or repeated local patterns that benefit from shared cluster summaries.

# Experiments

We test whether reduced complexity harms accuracy across domains with very different structure. We evaluate on three domains with different sequence statistics:

- **Language modeling (enwik8).** Long byte sequences with causal prediction. We report bits-per-byte under a shared training budget.
- **High-energy physics jets.** Variable-length particle sequences with full-context binary classification. We report Accuracy, F1, and AUC.
- **ModelNet40.** Fixed-size 3D point clouds with full-context multi-class classification. We report overall Accuracy and mean-class Accuracy.

All models use the same training schedules within each domain.

# Results

## Language Modeling Results

We evaluate our architectures on the enwik8 character-level language modeling task. All models are trained with causal masking, using the same hyperparameters (batch size 32, learning rate $3 \times 10^{-4}$).

### Main Results (50,000 steps)

We begin with a depth sweep on enwik8 to compare our clustered attention variants against standard baselines under a fixed configuration. We then isolate two orthogonal factors that may explain the gap: routing granularity (cluster scale) and cross-cluster communication capacity (mixing-rank dimension $k$).

Table 1 shows validation bits-per-byte (bpb) for different architectures across varying layer depths:

| Layers | MHA   | LinearAttention | RCA | LCA   | CKA   |
|-------:|------:|----------------:|-----------------:|------:|------:|
| 1      | 2.724 | 2.917 | 3.866 | 3.802 | 2.779 |
| 2      | 1.898 | 2.398 | 3.716 | 3.708 | 2.360 |
| 4      | 1.763 | 2.038 | 3.613 | 3.635 | 2.066 |
| 6      | 1.687 | 1.896 | 3.537 | 3.553 |   1.873   |
| 8      | 1.620 | 1.797 | 3.478 | 3.479 |   1.802   |


CKA performs similarly to linear attention across depths, while RCA and LCA perform poorly, despite modest gains from learned clustering over random clustering. With short enwik8 sequences and limited training, hard cluster partitioning likely blocks useful dependencies before stable cluster structure emerges. In this baseline setup, we also use a modest mixing-rank dimension ($k=8$), which may limit cross-cluster expressivity.

Note that SuperClusterAttention is not included here because it cannot be applied under causal masking, which is required for this language modeling setting.

### Cluster Scale Ablation (20,000 steps)

To investigate why RCA performs poorly, we vary the cluster scale parameter $s$ where $C = s \cdot \sqrt{n}$ and measure how cluster size influences performance. 

For 2-layer models:

![IMG_6249](https://hackmd.io/_uploads/ry3pXNHMWx.jpg)

Since cluster assignments here are random (not learned), even clusters roughly half the sequence length perform similarly to linear attention, showing that learned clustering is crucial if we want improvements beyond simple random partitioning.

### Cross-cluster Mixing Ablation (20,000 steps)

While cluster scale controls how coarse the routing is, CKA also introduces an independent source of capacity through low-rank cross-cluster mixing. We next vary the mixing-rank dimension $k$ to isolate its contribution.

![image](https://hackmd.io/_uploads/rk8u2lLG-x.png)

Increasing $k$ leads to consistently lower training loss by 20k steps, with the separation emerging most clearly in the mid-to-late training regime. This pattern suggests that richer cross-cluster interaction subspaces provide additional useful modeling capacity in this setting. Because this is a short-horizon, single-run ablation, we present the result as directional evidence rather than a definitive statement about scaling behavior.

To check whether this extra capacity is actually used, we tracked the norm and structure of the learned low-rank mixing matrices (MA/MB) during training.
![image](https://hackmd.io/_uploads/Skh0T-8z-g.png)
The MA/MB weights start tiny at 1k steps and grow to a similar scale as other attention weights by 20k, especially in layers 0–2. Their patterns also change shape rather than just getting scaled up.


## High-Energy Physics Jet Tagging Results

We evaluate our architectures on the HEP jet-tagging task using noncausal attention, which provides each event with full context. All models are trained for 20,000 steps under identical schedules. Table 2 reports test accuracy, F1, and AUC, sorted by accuracy.

![Screenshot 2025-12-09 at 3.14.50 PM](https://hackmd.io/_uploads/SkBBm-LM-e.png)


Caption: Colors indicate relative ranking within each metric (green best, then yellow, orange, red).

RandomClusterAttention ranks first in accuracy despite using random partitions, which indicates that HEP events contain strong latent grouping that even simple clustering can exploit. SuperClusterAttention is competitive, suggesting that routing global information across clusters aligns with the underlying jet substructure.

## ModelNet40 Results

We evaluate our architectures on the ModelNet40 3D object classification task using noncausal attention. All models are trained for 20 000 steps under identical settings. Table 3 reports test accuracy and mean-class accuracy, sorted by mean-class accuracy.

![Screenshot 2025-12-09 at 3.15.15 PM](https://hackmd.io/_uploads/r17wXW8fZe.png)


LinearAttention leads across metrics, while clustered variants lag. This implies that arbitrary clustering disrupts local geometric neighborhoods in point clouds, so 3D object recognition does not benefit from global cluster routing.

# Discussion
Among efficient attention methods focused on reducing quadratic cost, our results suggest that clustering offers a structurally different route than kernel-only or sparsity-only approaches. 

**SuperClusterAttention (SCA)** and its variants (LCA, RCA) perform very poorly on language modeling, indicating that clusters do not provide useful structure in that domain. SCA shows high performance on the HEP task, with comparatively poor performance from LCA. This contrast suggests that the relevant physics signal relies on information flow across clusters that is expressed only in SCA. On the contrary, we were intrigued by the high performance of RCA. Since it uses only random partitions, this suggests that some of the relevant signal is accessible even without learned clustering. We reran this ablation 5 times and note that the standard deviation of the accuracies is ~0.005, so the effect may be explained by stochastic variation rather than meaningful structure.
<!-- It also indicates that we do not yet fully understand which parts of the clustering mechanism matter most for HEP, and that further analysis is needed. -->

In addition, SCA and its ablations perform poorly on ModelNet40. Routing through a supernode and restricting attention to clusters may interfere with how the 3D structure is represented. Unlike the HEP task, where global mixing appears helpful, 3D recognition seems more sensitive to how local spatial cues are preserved. Further investigation is needed to understand the exact failure mode.

**ClusterKernelAttention (CKA)** matches or slightly outperforms linear attention in our language modeling runs, but is worse on some domain tasks like HEP. This method uses soft assignments and a low-rank cross-cluster mixing mechanism with mixing-rank dimension $k$ to keep routing differentiable and subquadratic at $O(n^{3/2})$. At the sequence lengths we test, the $O(n^2)$ bottleneck is not yet active and clustering signals appear too weak to reliably guide routing. However, our mixing-rank ablation shows that increasing $k$ yields lower training loss than linear attention, suggesting that cross-cluster mixing provides real capacity even in this short-context regime. Overall, CKA’s benefits appear modest and domain-dependent at this scale, motivating longer-context tests to determine whether these gains become more consistent with stronger clustering structure.

Taken together, these results suggest that future work on efficient attention may benefit from focusing not only on approximating the softmax kernel, but also on structuring communication paths that reflect latent organization of the data. Linear attention and Performer-style kernels already do an excellent job when every token interacts globally, but they do not encourage information flow through intermediate routes. Clustering also encourages a communication pattern that may become increasingly valuable as sequence lengths scale well beyond current limits.


<!-- // TODO elaborate on acronyms-->

# Conclusion
Our study indicates that reducing attention costs through clustering is feasible without collapsing performance at the scales we tested. SuperClusterAttention and ClusterKernelAttention approach the same goal with different mechanisms: one inserts explicit supernodes for global exchange, the other replaces dense token interactions with soft cluster routing combined with linear kernels and low-rank mixing. In both cases, information still travels globally, but the path is mediated by a much smaller set of cluster states.

That said, the overall picture is mixed, and our results should be interpreted cautiously. Performance varies sharply by domain, underscoring the need for domain-specific analyses before drawing broader conclusions.

Moving forward, we would like to see attention mechanisms that adapt the number of clusters, learn hierarchical cluster structure across layers, or incorporate priors that encourage meaningful routing without manual tuning, especially at larger sequence scales.

# References

1. Katharopoulos, A., et al. "Transformers are RNNs: Fast autoregressive transformers with linear attention." ICML 2020.
2. Choromanski, K., et al. "Rethinking attention with performers." ICLR 2021.
3. Kitaev, N., Kaiser, Ł., & Levskaya, A. "Reformer: The efficient transformer." ICLR 2020.
4. Vyas, A., Katharopoulos, A., & Fleuret, F. “Fast Transformers with Clustered Attention.” NeurIPS 2020.
5. Child, R., Gray, S., Radford, A., & Sutskever, I. "Generating long sequences with sparse transformers." arXiv:1904.10509, 2019.
6. Beltagy, I., Peters, M. E., & Cohan, A. "Longformer: The long-document transformer." arXiv:2004.05150, 2020.
7. Zaheer, M., et al. "Big bird: Transformers for longer sequences." NeurIPS 2020.
8. Courbariaux, M., Hubara, I., Soudry, D., El-Yaniv, R., & Bengio, Y. "Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to +1 or −1." arXiv:1602.02830, 2016.


# Appendix
## ClusterKernelAttention Derivation
ClusterKernelAttention replaces full token–token attention with kernel attention routed through soft clusters. Each token contributes its kernelized keys and values to the clusters it belongs to, clusters maintain running summaries, and tokens read back a cluster-conditioned context.

**Process.**
The procedure is as follows:
1. **Soft cluster assignment.**  
   Each token $x_i \in \mathbb{R}^d$ is projected to cluster logits 
   $$ z_i = W_c x_i \in \mathbb{R}^C $$
   and converted to soft assignments
$$
   a_{i,c} = \mathrm{softmax}\!\left(\frac{z_i}{\tau}\right)_c,\qquad \sum_{c=1}^C a_{i,c} = 1.
$$

2. **Kernel projection.**  
   Queries, keys, and values are computed per token,
$$ Q_i = \phi(W_Q x_i),\qquad
   K_i = \phi(W_K x_i),\qquad
   V_i = W_V x_i, $$
where $\phi(\cdot)$ is a positive feature map (we utilized $\phi(x)=\mathrm{ELU}(x)+1$).

3. **Cluster accumulation.**  
   Each cluster $c$ maintains running sums of kernelized keys and key–value products,
$$
   K_{c}^{(t)} = \sum_{i \le t} a_{i,c}K_i,\qquad
   S_{c}^{(t)} = \sum_{i \le t} a_{i,c}K_iV_i.
$$
   With causal masking these are prefix sums so position $t$ only sees positions $i \le t$.

4. **Cluster mixing.**  
   Cluster states are mixed through a learned low-rank transformation
$$
   M \approx AB^\top,\qquad A,B \in \mathbb{R}^{C \times k},
$$
   and we form
$$
   \tilde{K}_{c}^{(t)} = \sum_{c'=1}^C M_{c,c'}K_{c'}^{(t)},\qquad
   \tilde{S}_{c}^{(t)} = \sum_{c'=1}^C M_{c,c'}S_{c'}^{(t)}.
$$
   This moves information across clusters without a full $C^2$ cost.
   For $C \approx \sqrt{T}$, the dominant work scales as $T^{3/2}$, maintaining subquadratic attention.

5. **Token readout.**  
   Tokens read from the mixed cluster states using their soft assignments,
$$
   K_i^\ast = \sum_{c=1}^C a_{i,c}\tilde{K}_{c}^{(t)},\qquad
   S_i^\ast = \sum_{c=1}^C a_{i,c}\tilde{S}_{c}^{(t)},
$$
   and the final output is
$$
   h_i = \frac{Q_i^\top S_i^\ast}{Q_i^\top K_i^\ast}.
$$
   as similarly done in [Performer](https://arxiv.org/abs/2009.14794).

**Complexity.**  
Mixing avoids a full $C^2$ transformation. For $C \approx \sqrt{T}$, the dominant work scales as $T^{3/2}$ instead of $T^2$.
