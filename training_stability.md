# Training Stability and Degeneracy Prevention
This document summarizes common failure modes encountered when training the rGCN-SCAE on multiplex graphs, along with the practical remedies and hyperparameter heuristics that are incorporated in this project.

## Cluster collapse / runaway imbalance
### Mechanism
Collapse arises when the L0-gating network learns to activate only a small subset of cluster units across nearly all nodes, or conversely, activates each node’s own unique gate (singleton clusters).
Because the model minimizes reconstruction error, it can achieve low loss by concentrating all traffic through one or two always-on gates (an information bottleneck collapse) rather than learning diverse, specialized subspaces.
### Symptoms
Most gates remain permanently off, or one/two gates dominate all activations; latent entropy is near zero.
### Fixes
- **Entropy floor (on gate activations):** Encourage diverse gate usage via
$$
\mathcal{L}_H = \max\big(0,\, H_{\text{min}} - H(p_g)\big),
\qquad p_g = \frac{1}{N} \sum_{i=1}^{N} \sigma(g_i)
$$,
where:
    - $\sigma(\cdot)$ denotes the sigmoid activation applied to the gate pre-activation $g_i$,
    - $N$ is the total number of nodes (or samples) in the current batch,
    - $p_g$ is the mean activation probability across all gates,
    - $H(p_g) = -\sum_k p_{g,k} \log p_{g,k}$ is the Shannon entropy over gate activations,
    - $H_{\min}$ is a small positive constant (e.g., 0.5-1.0 bits) specifying the minimum desired entropy level.
- **Dirichlet prior on gate frequencies:** Impose $p_g ∼ Dir(\alpha)$ to prefer moderate utilization; small $\alpha>1$ enforces near-uniform gate usage.
- **Temperature annealing:** Adjust the hard-concrete stretch parameters so early training is smoother (high temperature → broad gate activations) and gradually sharpen later.
- **Warm-up $\lambda$-schedule:** Begin with $\lambda=0$ (no sparsity) and increase over 20–40 epochs, preventing premature gate extinction.
- **Confidence-weighted negative sampling:** Reweight reconstruction negatives inversely by gate entropy so uncertain assignments contribute more to training and preserve diversity.

## Hard-concrete (L0) instability
### Mechanism
The L0 gates have discontinuous gradients and high variance early in training.
This leads to chattering gates (rapid on/off flips) and dead clusters (permanently closed).
### Fixes
- **$\lambda$ warm-up:** During training, the sparsity coefficient $\lambda$ is gradually increased from 0 to its target value $\lambda_{\max}$ over T warm-up epochs: $\lambda_t = \lambda_{\max} \frac{t}{T}, \qquad t \in [0,\, T]$.
This linear schedule prevents premature gate extinction by allowing the encoder and decoder to stabilize before sparsity pressure is applied.
After the warm-up phase, $\lambda_t$ is held constant at $\lambda_{\max}$.
- **Clamping:** Clamp the pre-gate activation values to \[-2, 2\] to avoid saturation.
- **Straight-through window:** Use wider continuous relaxation range $\in [\epsilon, 1+\epsilon]$
- **EMA gates:** Apply exponential moving average of gate activations for temporal smoothing.
- **Revival pass:** Re-enable dead clusters if their latent gradients exceed a threshold (gradient-based "resurrection").

## Over-smoothing / over-squashing in rGCN
### Mechanism
With deep message passing, node embeddings converge to similar vectors as information averages over many hops.
Gradients through long paths attenuate exponentially (over-squashing).
### Symptoms
Latent vectors become indistinguishable; long-range signals vanish.
### Fixes
- **Restrict depth** to 2–3 layers with **residual/JK connections** ["Representation Learning on Graphs with Jumping Knowledge Networks" by Xu et al. 2018](https://arxiv.org/abs/1806.03536).
- Inject **positional encodings** (i.e. Laplacian eigenvectors, k-core indices, Random-Walk SE) to preserve structural individuality.
- **DropEdge**: Randomly drop edges per epoch to decorrelate messages ["DropEdge: Towards Deep Graph Convolutional Networks on Node Classification" by Xu, Huang, Xu, Huang 2019](https://arxiv.org/abs/1907.10903).
- Apply **relation-wise normalization** to keep scales comparable across edge types.

## Hub and degree bias
### Mechanism
In a GCN, each node updates its embedding by averaging (or summing) messages from its neighbors.
If some nodes (hubs) have many more neighbors than others, they contribute disproportionately to the message-passing flow of information across the graph.
Over time, the model starts to learn embeddings that encode degree statistics rather than true semantic similarity.
### Symptoms
Clusters mirror degree more than semantics.
### Fixes
- **Degree-corrected normalization:** Scale incoming messages by $1 / \sqrt{d_i d_j}$ or to counteract hub dominance.
- **Degree-proportional edge dropout:** Randomly drop edges from high-degree nodes with probability proportional to $d_i / d_{\max}$ to reduce over-representation of hubs.
- **Degree-orthogonalization penalty:** Include degree as a covariate but penalize its correlation with latent features $L_{\text{deg}} = \mathrm{corr}^2(z\, \log d)$.

## Multiplex imbalance across relation types
### Mechanism
High-frequency relation types dominate gradients, drowning rare but semantically informative edges.
### Fixes
- **Per-relation sampling quotas** during batch construction.
- **Inverse-frequency loss reweighting:** $w_r = 1 / \sqrt(f_r)$.
- **Per-relation decoders:** Use separate decoder for each relation type via learnable temperature or scale parameters to equalize pre-gate activation variance.

## Negative sampling bias
### Mechanism
Random negatives are too easy; model learns trivial discrimination and clusters remain unshaped.
### Symptoms
Easy negatives inflate AUC but don’t shape clusters; or cross-type negatives dominate.
### Fixes
- **Stratify negatives** per relation and node-type pair.
- Add **hard negatives** using nearest-neighbor mining in latent space.
- **Maintain memory bank** of recent negatives within batch to preserve gradient diversity.
Because the model must batch by subgraph size to preserve size invariance, training examples cannot be randomly shuffled across the dataset.
As a result, sampled negative edges are repeatedly drawn from the same local neighborhood distribution in for batch i, reducing their diversity and therefore their usefulness for contrastive learning.
To increase negative sample diversity, a memory bank $\mathcal{M}$ of recent latent embeddings is maintained across iterations.
At iteration $t$, the negative pool is drawn from both the current batch $B_t$ and the stored embeddings.
Gradients are propagated only through the current batch, while the bank entries serve as fixed contrastive targets.
This preserves gradient diversity and stabilizes training without re-computing past examples.
