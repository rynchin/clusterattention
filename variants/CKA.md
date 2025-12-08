# ClusterKernelAttention
ClusterKernelAttention replaces full token–token attention with kernel attention routed through soft clusters. Each token contributes its kernelized keys and values to the clusters it belongs to, clusters maintain running summaries, and tokens read back a cluster-conditioned context.

The procedure is as follows:
1. **Soft cluster assignment.**  
   Each token $x_i \in \mathbb{R}^d$ is projected to cluster logits
```math
   z_i = W_c x_i \in \mathbb{R}^C
```
   and converted to soft assignments
```math
   a_{i,c} = \mathrm{softmax}\!\left(\frac{z_i}{\tau}\right)_c,\qquad \sum_{c=1}^C a_{i,c} = 1.
```

2. **Kernel projection.**  
   Queries, keys, and values are computed per token,
```math
   Q_i = \phi(W_Q x_i),\qquad
   K_i = \phi(W_K x_i),\qquad
   V_i = W_V x_i,
```
   where $\phi(\cdot)$ is a positive feature map (we utilized $\phi(x)=\mathrm{ELU}(x)+1$).

3. **Cluster accumulation.**  
   Each cluster $c$ maintains running sums of kernelized keys and key–value products,
```math
   K_{c}^{(t)} = \sum_{i \le t} a_{i,c}K_i,\qquad
   S_{c}^{(t)} = \sum_{i \le t} a_{i,c}K_iV_i.
```
   With causal masking these are prefix sums so position $t$ only sees positions $i \le t$.

4. **Cluster mixing.**  
   Cluster states are mixed through a learned low-rank transformation
```math
   M \approx AB^\top,\qquad A,B \in \mathbb{R}^{C \times k},
```
   and we form
```math
   \tilde{K}_{c}^{(t)} = \sum_{c'=1}^C M_{c,c'}K_{c'}^{(t)},\qquad
   \tilde{S}_{c}^{(t)} = \sum_{c'=1}^C M_{c,c'}S_{c'}^{(t)}.
```
   This moves information across clusters without a full $C^2$ cost.
   For $C \approx \sqrt{T}$, the dominant work scales as $T^{3/2}$, maintaining subquadratic attention.

5. **Token readout.**  
   Tokens read from the mixed cluster states using their soft assignments,
```math
   K_i^\ast = \sum_{c=1}^C a_{i,c}\tilde{K}_{c}^{(t)},\qquad
   S_i^\ast = \sum_{c=1}^C a_{i,c}\tilde{S}_{c}^{(t)},
```
   and the final output is
```math
   h_i = \frac{Q_i^\top S_i^\ast}{Q_i^\top K_i^\ast}.
```
   as similarly done in [Performer](https://arxiv.org/abs/2009.14794).

**Complexity.**  
Mixing avoids a full $C^2$ transformation. For $C \approx \sqrt{T}$, the dominant work scales as $T^{3/2}$ instead of $T^2$.
