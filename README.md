# Automated Psychiatric Nosology via Partitioning of Knowledge Graph
## Table of Contents
- [Code Notes](#code-notes)
- [Background / Literature Review](#background--literature-review)
    - [Biomarkers of Psychopathology](#biomarkers-of-psychopathology)
    - [Transdiagnostic Dimensions](#transdiagnostic-dimensions)
    - [Automated Methods](#automated-methods)
    - [Conclusion](#conclusion)
- [Research Question](#research-question)
- [Hypothesis](#hypothesis)
- [Methods](#methods)
    - [Graph Creation](#graph-creation)
    - [Preventing Biased Alignment](#preventing-biased-alignment)
    - [Partitioning](#partitioning)
        - [Encoder Architecture](#encoder-architecture)
        - [Decoder Architecture](#decoder-architecture)
        - [Preventing Trivial / Degenerate Solutions](#preventing-trivial--degenerate-solutions)
        - [Adaptive Subgraph Sampling to Mitigate Overfitting](#adaptive-subgraph-sampling-to-mitigate-overfitting)
        - [Comparison](#comparison)
- [Abbreviations](#abbreviations)
- [References](#references)

## Code Notes
- Install project requirements via `pip3.10 install -r requirements.txt`.
- Run `huggingface-cli download tienda02/BioMedKG --repo-type=dataset --local-dir ./data` to download the main data for the knowledge graph.
- Run
```
mkdir -p data/hpo
curl -L -o data/hpo/hp.obo https://purl.obolibrary.org/obo/hp.obo
curl -L -o data/hpo/phenotype.hpoa https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa
curl -L -o data/hpo/genes_to_phenotype.txt https://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt
python3.10 prepare_hpo_csv.py data/hpo/hp.obo data/hpo/phenotype.hpoa genes_to_phenotype.txt data/hpo/
```
to prepare the data used for augmenting the graph to prevent degeneracy after removing the diagnosis nodes.
- MLflow is used for optional experiment tracking.
    - Enable tracking with MLflow by adding `--mlflow` (plus optional `--mlflow-tracking-uri`, `--mlflow-experiment`, `--mlflow-run-name`, and repeated `--mlflow-tag KEY=VALUE` flags) to `train_rgcn_scae.py`, which logs parameters, per-epoch metrics, and uploads the generated `partition.json` artifact.
    - Metric explanations:
        - **total_loss** = weighted sum of reconstruction, sparsity, entropy, Dirichlet, embedding-norm, KL, consistency, gate-entropy, and degree penalties reported below.
        - **recon_loss** averages the BCE losses for positive and sampled negative edges per graph.
        - **sparsity_loss**, **cluster_l0**, **inter_l0** track the hard-concrete L0 penalties that drive cluster/edge sparsity.
        - **entropy_loss** encourages per-graph assignment entropy to stay above `--assignment-entropy-floor`.
        - **dirichlet_loss** is the KL term that keeps mean cluster usage close to the Dirichlet prior.
        - **embedding_norm_loss** and **kld_loss** regularise latent vectors on a per-graph basis.
        - **degree_penalty** (with **degree_correlation_sq**) measures the decorrelation between latent norms and node degree.
        - **consistency_loss** / **consistency_overlap** are the memory-bank temporal consistency terms (0 when disabled).
        - **gate_entropy_bits** and **gate_entropy_loss** track how evenly decoder gates remain active.
        - **num_active_clusters** records the eval-mode gate count that matches the saved `partition.json`; **expected_active_clusters** is the summed HardConcrete L0 expectation.
        - **num_active_clusters_stochastic** retains the raw training-mode gate count if you need to debug EMA smoothing or gating noise.
        - **negative_confidence_weight** shows the entropy-driven reweighting applied when `--neg-entropy-scale > 0`.
        - **num_negatives** counts sampled negative edges that survived the per-graph cap.
        - **timing_sample_neg** and **timing_neg_logits** capture the wall-clock time (in seconds) spent sampling negatives and scoring them.
- The rGCN-SCAE trainer picks the latent cluster capacity automatically via `_default_cluster_capacity`, which grows sublinearly with node count (√N heuristic with a floor tied to relation count) to balance flexibility and memory usage.
- Training stops early when the stability metric you request stays within tolerance for a sliding window of epochs. Pass `--cluster-stability-window` (number of epochs), `--cluster-stability-tolerance` (absolute span), and optionally `--cluster-stability-relative-tolerance` when calling `train_rgcn_scae.py`; once the chosen `stability_metric` (defaults to `num_active_clusters`) varies less than both thresholds after `--min-epochs`, the run halts and records the stop epoch/reason in the history log.
- To run the hierarchical stochastic block model baseline you must install [graph tool](https://graph-tool.skewed.de) before calling [create_hSBM_partitions.py](./create_hSBM_partitions.py).
- Run `python3.10 -m pytest` from the repository root to execute the regression tests for the extraction pipeline and training utilities.

## Background / Literature Review
Psychiatric nosology has long been dominated by categorical systems such as the DSM and the ICD.
These frameworks define discrete diagnostic entities and draw strict boundaries between normal and pathological states.
However, their limitations are well established, including high comorbidity rates, arbitrary thresholds, and limited biological validity [1], [2].
Categorical standards such as DSM-5 impose rigid yes/no decisions regarding diagnosis [3], despite evidence from meta-analytic research indicating that most psychiatric disorders are better conceptualized as continuous spectra rather than binary categories [4].
These concerns have motivated the development of dimensional alternatives, notably the HiTOP [3] and the RDoC [5].
Each represents a unique effort to reconceptualize psychiatric nosology: HiTOP is symptom-driven and data-based, while RDoC is neuroscience-driven and theory-based.
This review examines evidence from biomarkers, transdiagnostic dimensions, and computational models.
Findings converge on the theme that psychiatric disorders are not discrete entities but reflect shared, transdiagnostic mechanisms with disorder-specific features sometimes co-occurring and thus are best represented dimensionally.

### Biomarkers of Psychopathology
Biomarker research increasingly demonstrates that biological abnormalities rarely remain unique to individual DSM categories.
Resting-state fMRI studies of large population cohorts have shown consistent disruptions in default mode, salience, and executive networks across depression, schizophrenia, and anxiety [6].
These findings were derived using graph-theoretic analyses of connectivity patterns, but despite their reproducibility across studies, they often lack disorder-specificity, limiting their diagnostic utility.
A meta-analysis pooling voxel-based morphometry data from more than 15,000 patients demonstrated gray matter reductions in the anterior cingulate cortex and insula across mood, anxiety, and psychotic disorders [7].
While the scale of this synthesis strengthens the claim of shared neurobiological substrates, heterogeneity in scanning methods and patient samples makes it difficult to identify causal pathways.
Rather than supporting disorder-specific neural substrates, these findings indicate shared biological bases.

Structural neuroimaging studies further suggest that biological variation aligns more closely with dimensional models than categorical diagnoses.
Meta-analysis shows that structural abnormalities cluster in ways consistent with HiTOP spectra, particularly internalizing and externalizing dimensions [8].
Evidence also supports the integration of multimodal approaches.
Reviews emphasize that relying on single modalities such as neuroimaging alone produces inconsistent results, while combining genetics, neuroimaging, peripheral biomarkers, and clinical measures offers greater potential for identifying robust transdiagnostic markers [9].
Collectively, biomarker research points to the conclusion that psychiatric disorders share overlapping biological substrates that map more naturally onto dimensional frameworks.

### Transdiagnostic Dimensions
Dimensional models provide alternative frameworks for capturing shared variance across disorders.
HiTOP is a transdiagnostic nosology that organizes psychopathology hierarchically, with broad spectra such as internalizing, externalizing, and thought disorder encompassing narrower syndromes [3].
A meta-analysis of 35 structural MRI studies involving over 12,000 participants found that abnormalities in cortical thickness and subcortical volume clustered in patterns consistent with HiTOP’s internalizing and externalizing spectra [8].
This supports the dimensional framework, though the cross-sectional nature of the data limits conclusions about developmental trajectories.
Functional neuroimaging studies have shown that individuals scoring high on internalizing dimensions exhibit hyperactivity in amygdala–hippocampal circuits across both mood and anxiety disorders, whereas externalizing spectra are associated with reduced prefrontal regulation of striatal reward pathways across disruptive behavior and substance use disorders [10].
These convergent findings provide neurobiological validation for dimensional constructs, although effect sizes remain modest.
Overlap across networks also persists, which is not a critical weakness for dimensional validity but does represent a limitation for clinical utility.

The RDoC initiative provides a complementary approach by focusing on functional domains of behavior and neural systems [5].
Domains such as cognition, negative valence, and arousal cut across traditional diagnoses and anchor psychopathology in specific neural circuits.
This approach differs from HiTOP by beginning with neuroscience constructs rather than symptom clustering, yet both converge on the principle that psychiatric syndromes are dimensional rather than categorical.

Meta-analytic research further supports the conclusion that dimensional continua provide a superior fit for psychiatric phenomena compared to categorical boundaries [4].
Overall, transdiagnostic models such as HiTOP and RDoC demonstrate greater validity than traditional nosologies, as they capture shared liability, reduce artifacts of comorbidity, and align more closely with underlying neurobiological processes.

### Automated Methods
In addition to biomarkers and dimensional frameworks, computational approaches offer formal models for psychiatric nosology.
Conceptual work in computational psychiatry has explored how diagnostic systems might be reframed in terms of latent processes and mathematical models.
For example, Bzdok and Meyer-Lindenberg propose a framework of “computational nosology and precision psychiatry” that emphasizes integrating behavioral, neural, and genetic data into quantitative models of psychopathology [12].
Although largely theoretical, such approaches highlight the potential for computational methods to reconfigure diagnostic standards.
At the same time, skepticism remains about the sufficiency of computational methods for redefining nosology.
Surveys of experts in computational psychiatry report concerns regarding circularity, where machine learning reproduces the categories it was trained on, lack of causal grounding in many statistical approaches, and neglect of subjective experience [13].
These critiques highlight the importance of ensuring that computational models do not merely replicate existing diagnostic systems but instead provide meaningful explanatory frameworks while also remaining true to patients’ lived experiences.

Beyond psychiatry, researchers in medicine have attempted to automate the construction of nosological frameworks.
Automated clustering methods have been applied to thousands of diseases using shared molecular features such as gene expression and protein interactions, yielding biologically coherent groupings that diverge from ICD classifications [14].
While this illustrates the feasibility of algorithmically derived nosologies, the reliance on molecular data alone neglects clinical and symptomatic dimensions critical for psychiatric classification.
Other approaches focus on ontology engineering at scale, including automated mapping of clinical vocabularies to biomedical ontologies [15] and the generation of large biomedical knowledge graphs that infer relationships among diseases, treatments, and biomarkers [16].
These methods demonstrate that automated classification is technically feasible, though their application in psychiatry remains limited.

Some early work has extended automation directly to the psychiatric nosology domain.
For example, graph-based clustering of polygenic risk variants has been used to identify subtypes of schizophrenia, suggesting a potential path toward biologically grounded psychiatric nosology [17].
Machine learning applied to electronic health records has also been used to detect latent subtypes of depression and other disorders, pointing to the possibility of automated reclassification based on large-scale clinical data [18].
Although these attempts are preliminary, they indicate that psychiatry may follow trends in other medical domains by adopting automated, data-driven methods to complement conceptual and dimensional frameworks.

### Conclusion
Across biomarkers, dimensional models, and computational frameworks, a consistent theme emerges: psychiatric disorders share biological and symptomatic substrates that transcend categorical nosologies.
Evidence from neuroimaging and meta-analyses demonstrates that common abnormalities appear across multiple disorders.
Dimensional models such as HiTOP and RDoC capture this overlap through hierarchical spectra and neuroscience-based domains.
Computational approaches illustrate the potential for integrating categories and dimensions into mechanistic models, though critiques caution against over-reliance on machine learning.

A clear gap remains.
Despite converging evidence that categorical systems are insufficient, there is currently no unified diagnostic standard that integrates biomarkers, dimensional constructs, and computational frameworks.
Existing studies tend to focus on isolated modalities or conceptual frameworks, underscoring the need for integrative approaches that are biologically valid, clinically useful, and transdiagnostic in scope.
This highlights the need for continued research into models that combine these strands into a biologically valid, parsimonious, and widely applicable classification system.

## Research Question
Can a proof-of-concept transdiagnostic dimensional psychiatric nosology be developed in an automated way by mining the scientific literature into a multiplex knowledge graph and partitioning it using information-theoretic methods, and how does its structure compare with HiTOP and RDoC in terms of parsimony, stability, and alignment?

## Hypothesis
It can be developed in this way and the resulting nosology will roughly have parsimony (as measured by minimum description length) within 95% or better of, stability (as measured by bootstrapped variation of information and bootstrapped adjusted rand index) >= 85% of and alignment (as measured by normalized mutual information and adjusted rand index) >= 75% of HiTOP and RDoC.
The parsimony will be so close to or better than HiTOP and RDoC due to the use of information-theoretic algorithms in the partitioning to optimize for compression.
The stability will be high due to the large volume of papers expected to be parsed.
The alignment will be reasonably high but not exact due to the vastly different methods in producing the final output (RDoC is mechanistically focused while HiTOP is focused on symptoms and this novel method will be automated).

## Methods
By mining the scientific literature into a multiplex graph and partitioning it with information-theoretic methods, this project draws inspiration from generative modeling’s emphasis on latent structure while also addressing the critiques of purely data-driven ML.
Unlike many ML approaches that risk reproducing existing DSM or RDoC categories (by training directly on them), this method removes those labels during graph construction.
Any observed alignment that later emerges with HiTOP or RDoC therefore reflects genuine structural similarity rather than trivial lexical overlap, ensuring a more independent test of whether automated nosology converges with established frameworks.

### Knowledge Graph
The knowledge graph was created from a subset of the BioMedKG dataset [19], which can be downloaded with: `huggingface-cli download tienda02/BioMedKG --repo-type=dataset --local-dir ./data`.
This data is then pared down to only the psychiatrically relevant nodes and edges by the [`create_graph.py`](./create_graph.py]) script.
First, loading of the disease, drug, protein, and DNA modality tables happens and the hybrid `PsychiatricRelevanceScorer` is invoked, which fuses ontology membership, learned group labels, psychiatric drug neighborhoods, and cosine similarity to psychiatric prototype text snippets into a continuous relevance score.
The high-confidence combinations still pass even if a single dimension underperforms, while low-scoring nodes are excluded.

Only disease vertices meeting these criteria become seed indices for the downstream graph walk.
Any edge whose source or target index appears in the psychiatric seed set is retained, optionally intersected with a whitelist of relation labels.
After collection, each edge is checked against the relation-role constraints prepared during initialization so that invalid edges are removed (i.e. "drug_disease" edges must actually bind drug-like nodes to disease-like nodes).
For every surviving edge, the extractor reconstructs a unique node table, joining in the disease/drug/protein/DNA/text attributes, and exporting JSON blobs for downstream consumption.
Psychiatric scoring outputs (psy_score, psy_evidence, boolean flags, and the final is_psychiatric decision) are carried through so that later models can weight nodes by clinical relevance.
Nodes lacking these columns receive neutral defaults to keep the table schema consistent.

The filtered node/edge frames are projected into a multiplex networkx.MultiDiGraph, preserving node metadata and relation labels; reverse edges are optionally added when symmetrical analysis is required.
During weighted projection each undirected edge receives a prior determined by its relation label and is modulated by the mean psychiatric score of its incident nodes, suppressing ties to weakly psychiatric neighbors while never dropping them outright.
Finally, the pipeline streams the psychiatric slice of PrimeKG into Parquet tables plus directed and weighted GraphML files, yielding artifacts whose every node and edge has survived both the semantic filters and the psychiatric relevance scoring requirements described above.

### Preventing Biased Alignment
Because the alignment metrics used to compare the emergent nosology with established frameworks (HiTOP and RDoC) can be artificially inflated if the same vocabulary appears in both the input graph and the target taxonomies, nodes corresponding to existing nosological systems are explicitly removed from the graph before partitioning.
They are left in the original graph in order to make alignment calculations easier.
This preserves semantically coherent nodes (symptoms, biomarkers, treatments, etc.) while keeping diagnostic labels available for later alignment checks and ensuring that the resulting graph structure emerges independently of existing nosological vocabularies.
This entity-level masking substantially reduces over-masking and yields more biologically meaningful connectivity patterns versus a simple token-level method.
Only after the final partitioning is complete will alignment metrics such as normalized mutual information and adjusted Rand index be computed against HiTOP and RDoC categories.
This ensures that any observed alignment reflects genuine structural similarities rather than trivial lexical overlap, preventing a biased alignment metric.

### Partitioning
Two complementary strategies were tested for discovering mesoscale structure in the multiplex psychopathology graph. First, a hierarchical stochastic block model (hSBM) [20], which provides a probabilistic baseline that infers discrete clusters by maximizing the likelihood of observed edge densities across multiple resolution levels.
This family of models gives interpretable, DSM-like partitions together with principled estimates of uncertainty, but inherits hSBM’s familiar computational burdens—quadratic scaling in the number of vertices and a rigid parametric form for block interactions.
This was used as a baseline.
The other model architecture tested was a **Relational Graph Convolutional Network Self-Compressing Autoencoder (RGCN-SCAE)**.
This model was trained exclusively on the full graph to derive a single global partitioning that captures the complete relational structure.
It learns a compact latent partition of the full knowledge graph while preserving relation-specific structure through low-rank factorization and hard-concrete gating.

#### Encoder Architecture
The encoder is an RGCN that couples a Deep Sets attribute encoder [21] with a stacked relational GCN that produces a temperature-controlled assignment matrix of shape |Nodes| $\times$ |Clusters|, forming the probabilistic basis for downstream partitioning via differentiable hard-concrete ($L_0$) gates [22].
Each node is first encoded by a DeepSets-style node attribute encoder, implementing the Zaheer et al. $\rho$/$\phi$ formulation, that is permutation-invariant.
This component integrates arbitrarily structured metadata—text learned embeddings, biomarkers and ontology terms.
It can expand its vocabulary online, allowing new attributes to be assimilated without retraining existing embeddings.
The resulting descriptor is concatenated with:
A learned node-type embedding, encoding categorical identity in a compact, regularized form and graph-positional encodings derived from standardized Laplacian eigenvectors, normalized per subgraph to maintain numerical stability when batch sizes or graph orders vary.
The fused vectors are processed through a relation-aware additive message-passing stack of graph convolutional layers using basis decomposition (a parameter sharing technique) across relation types.
Each layer is followed by GraphNorm (default, for unbalanced batches) or LayerNorm, a ReLU activation, and optional dropout.
Because the encoder operates directly on the supplied edge_index, it naturally accommodates cyclic connectivity and multiplex relation types without special handling.
During training the encoder applies degree-proportional edge dropout, symmetric degree normalization and inverse-frequency relation reweighting.
Together these steps keep gradients stable on multiplex graphs whose degree distributions and relation frequencies span orders of magnitude.
The additive message-passing uses an RGCN stack with basis decomposition per relation and per-layer GraphNorm (or LayerNorm on request).
After the stack, node embeddings can be augmented with a lightweight virtual-node context that injects graph-level statistics without scaling with |Vertices|.
Let $h_i$ denote the resulting embedding for node $i$ and $P = [p_1, \dots, p_K]$ the prototype bank maintained via an exponential moving average.
Embeddings are $L_2$-normalized, $\hat{h}_i = h_i / \lVert h_i \rVert_2$, and scored against prototypes with a temperature-controlled softmax
$$s_{ik} = \frac{\hat{h}_i^{\top} p_k}{\tau_a} + \log g_k,$$
where $\tau_a$ is the current assignment temperature and $g_k$ is the sample from the cluster hard-concrete gate.
Sinkhorn balancing applies $T$ rounds of alternating row/column normalization with entropic regularization $\varepsilon$ to $\exp(s)$, yielding a doubly-stochastic assignment matrix $Q$ that approximately satisfies $Q\mathbf{1} = \mathbf{1}$ and $Q^\top \mathbf{1} = (n/K)\,\mathbf{1}$.
The hard-concrete gates follow Louizos et al.'s [22] reparameterization with expected sparsity $\mathbb{E}[\mathrm{L}_0] = \sigma\left(\log \alpha - \tau_g \log \frac{-\gamma}{1+\zeta}\right)$, where $\alpha$ is the gate logit and $(\gamma, \zeta)$ the stretch parameters; annealing $\tau_g$ prevents premature collapse.
A momentum memory bank anchors embeddings for repeated node IDs across sampled subgraphs, and the encoder records the batch-index vector $b \in \{0,\dots,B-1\}^N$ that the decoder and degree-orthogonality penalty consume downstream.
The assignment matrix $Q$ and auxiliary tensors are then ready for $L_0$ sparsification.

#### Decoder Architecture
The decoder maps the Sinkhorn-balanced cluster assignments back to multiplex edge logits through a low-rank relational energy model gated by the hard-concrete masks.
Relation-specific inter-cluster affinities are produced via a low-rank factorization: every relation learns coefficients over a shared cluster basis, bringing the parameter count down to $\mathcal{O}(R \cdot C \cdot rank_r)$ instead of $\mathcal{O}(R \cdot C^2)$ (with $rank_r \ll C$).
A learnable absent-edge bias per relation absorbs the background sparsity so the decoder can focus its capacity on informative deviations.

For each mini-batch the decoder streams positive edges and sampled negatives in configurable chunks, multiplying source assignments by the gated relation weights and accumulating reconstruction losses.
This, combined with gradient checkpointing can be used to avoid exhausting CPU/GPU memory.
Relation frequencies reweight both positive and negative terms so that rare relation types retain influence, while optional type-restricted negative sampling and per-graph budgets keep contrastive updates aligned with ontology constraints.
Entropy-aware reweighting of negatives, inverse-frequency scaling, and chunked evaluation preserve size invariance and stability even on the largest subgraphs.
The relation-specific logits follow $\ell_{ijr} = a_i^{\top} W_r a_j + b_r,$ with assignment vectors $a_i = Q_{i\cdot}$ and low-rank weight matrices $W_r = \sigma(F_r B) \odot G_r$, where $F_r \in \mathbb{R}^{C\times d}$, $B \in \mathbb{R}^{d\times C}$, and $G_r$ is the current hard-concrete gate sample.
Binary cross-entropy losses accumulate over observed edges $(i,j,r) \in E$ and sampled negatives $(i',j',r)$.
Negative-sample losses are additionally scaled by a gate-entropy confidence factor $w_{\text{neg}} = 1 + \lambda \big(\max(H_g, H_{\min})^{-1} - 1\big)$, where $H_g = -\sum_k \pi_k \log_2 \pi_k$ is the gate entropy in bits computed from gate probabilities $\pi$ and $H_{\min}$ is the floor used in code.
This keeps confident, low-entropy gates focused on harder negatives without destabilizing early training.

#### Preventing Trivial / Degenerate Solutions
Five primary forms of degenerate or trivial solutions are guarded against in this model: uniform embeddings, local collapse, global collapse, decoder memorization, and latent drift.
Absent-edge modelling via type-aware negative sampling penalises trivial partitions even when node embeddings look uniform.
The decoder contrasts observed edges against sampled non-edges inside each subgraph while operating on Sinkhorn-balanced assignment vectors $a_i$ rather than raw encoder features.
For every relation r the decoder forms a low-rank, hard-gated interaction matrix $W_r$ (built from gated relation factors) and applies an absent-edge bias $b_r$.
The reconstruction loss adds the average positive-edge binary cross-entropy and a ratio-corrected negative term for every graph g in the batch:
    $$
    \mathcal{L}_{\text{recon}} \;=\; \frac{1}{G} \sum_{g=1}^{G} \Bigg[
        \frac{1}{|E_g|} \sum_{(i,j)\in E_g}
        \mathrm{BCE}\!\Big(\sigma(a_i^\top W_{r_{ij}} a_j + b_{r_{ij}}),\,1\Big)
        \;+\;
        \frac{|E_g|}{|\tilde{E}_g|} \sum_{(i',j')\in \tilde{E}_g}
        \mathrm{BCE}\!\Big(\sigma(a_{i'}^\top W_{r_{i'j'}} a_{j'} + b_{r_{i'j'}}),\,0\Big)
    \Bigg],
    $$
    where $\tilde{E}_g$ contains the sampled negatives for graph g and G is the number of graphs in the mini-batch.

| Symbol | Meaning |
| --- | --- |
| G                          | Number of graphs (mini-batch components) evaluated in the loss. |
| E_g                        | Set of observed edges for graph g. |
| $\tilde{E}_g$              | Negative samples generated for graph g under the current negative-sampling policy. |
| $|E_g|$                    | Cardinality of E_g. |
| $|\tilde{E}_g|$            | Cardinality of $\tilde{E}_g$. |
| $a_i \in \Delta^{K-1}$     | Sinkhorn-balanced cluster assignment for node i (probability vector over K clusters). |
| $r{ij}$                    | Relation index of edge (i, j); selects decoder parameters for that relation. |
| $W{r}$                     | Low-rank, hard-gated relation weight matrix for relation r. |
| $b_r$                      | Learnable absent-edge bias for relation r. |
| $\sigma(\cdot)$            | Logistic sigmoid that converts logits to probabilities. |
| $\mathrm{BCE}(\hat{y}, y)$ | Binary cross-entropy between probability $\hat{y}$ and label y. |

Intuitively, the first term rewards high similarity for real edges while the second penalizes the model when it predicts high similarity for random, non-existent connections.

However, this is not enough to prevent all situations of structure collapse.
For example, a situation could arise where local structure collapses but global does not.
Negative sampling does not penalize this because such coarse partitions can still separate positives from random negatives effectively.
Entropy regularization on the cluster assignment matrix counteracts this tendency by enforcing a floor on per-graph assignment entropy.
For each graph g, it computes $H_g = -\frac{1}{|V_g|} \sum_{i \in g} \sum_{k=1}^{K} p_{ik} \log p_{ik}$ and applies a hinge,
$$
\mathcal{L}_{\text{entropy}} = \lambda_H \cdot \frac{1}{G} \sum_{g=1}^{G} \max\!\bigl(0,\, H_{\text{floor}} - H_g\bigr),
$$
so the term activates only when entropy dips below the target floor $H_{\text{floor}}$.

$$
\mathcal{L}_H = -\frac{1}{N} \sum_{i=1}^{N}\sum_{k=1}^{K} p_{ik}\,\log p_{ik}
$$

| Symbol | Meaning |
| --- | --- |
| $H_g$              | Mean assignment entropy for graph g, $-\frac{1}{|V_g|}\sum_{i\in g}\sum_{k} p_{ik}\log p_{ik}$. |
| $|V_g|$            | Number of nodes belonging to graph g. |
| $H_{\text{floor}}$ | Target minimum entropy per graph (defaults to $\log K$ if unset). |
| $\lambda_H$        | Weight assigned to the entropy penalty. |

However, the entropy regularization only handles local degeneracy because it acts locally.
The model could still produce a few large "meta-clusters" that reconstruct edges well but lack finer internal structure — all symptoms in one, all treatments in another, etc.
To prevent this, the model matches the mean assignment usage against a Dirichlet-inspired prior stored as a categorical baseline $\pi$. Let $u_k = \frac{\sum_i p_{ik}}{\sum_{k'} \sum_i p_{ik'}}$ be the empirical cluster usage.
It minimizes the KL divergence
$$
\mathcal{L}_{\text{Dirichlet}} = \lambda_{\text{Dir}} \sum_{k=1}^{K} u_k \log \frac{u_k}{\pi_k},
$$
where $\pi_k$ is the normalised prior mass derived from the user-specified Dirichlet concentration parameters.

| Symbol | Meaning |
| --- | --- |
| $u_k$                  | Empirical mean cluster usage, normalised so $\sum_k u_k = 1$. |
| $\pi_k$                | Prior cluster usage probability derived from the user-specified Dirichlet concentrations (normalised once). |
| $\lambda_{\text{Dir}}$ | Weight applied to the KL penalty term. |

Even with these local and global regularizers, the decoder can still exploit trivial optima by memorizing adjacency patterns rather than learning meaningful structure.
This can occur when embeddings or decoder weights grow unbounded, allowing perfect reconstruction without informative latent geometry.
To prevent this *reconstruction-dominant collapse*, the mean squared embedding norm per graph so that large batches do not dominate:

$$
\mathcal{L}_{\text{norm}} = \lambda_z \cdot \frac{1}{G} \sum_{g=1}^{G} \left( \frac{1}{|V_g|} \sum_{i \in g} \|z_i\|_2^2 \right).
$$

| Symbol | Meaning |
| --- | --- |
| $\lambda_z$ | Scaling factor for the latent L2 penalty. |
| $z_i$       | Encoder embedding for node i before Sinkhorn balancing. |

Finally, to stop latent drift and keep graph-level statistics well behaved, regularization is added of the first two moments of each graph’s embedding distribution toward a zero-mean, unit-variance Gaussian:

$$
\mathcal{L}_{\text{mom}} = \frac{\lambda_{\text{KLD}}}{2G} \sum_{g=1}^{G} \sum_{d=1}^{D} \left( \mu_{g,d}^2 +
\sigma_{g,d}^2 - \log \sigma_{g,d}^2 - 1 \right),
$$

| Symbol | Meaning |
| --- | --- |
| $D$                    | Latent dimensionality of the encoder outputs. |
| $\mu_{g,d}$            | Empirical mean of latent dimension d over nodes in graph g. |
| $\sigma_{g,d}^2$       | Empirical variance of latent dimension d over nodes in graph g (clamped to stay positive). |
| $\lambda_{\text{KLD}}$ | Weight of the moment-based regulariser. |

#### Stability and Regularization
Training rGCN-SCAE models on multiplex graphs can exhibit several characteristic failure modes.
Detailed mitigation strategies and hyperparameter guidelines are provided in [`training_stability.md`](training_stability.md).

#### Adaptive Subgraph Sampling for Stability Testing
A practical limitation of this approach is that it is trained on a single, fixed multiplex graph.
Without a distribution of graphs, the model risks overfitting to idiosyncratic topological patterns rather than learning generalizable relational principles.
To check the effect of training on only the full knowledge graph, a dataset was be created with different subgraphs sampled via node-hopping from randomly chosen seed nodes with the hop radius determined adaptively from local connectivity metrics such as node degree, clustering coefficient, and k-core index in order to preserve local connectivity and type proportions.
For each seed s, a composite score $g(s) = z(c_s) + z(\kappa_s) - z(\deg_s)$ to select a hop radius $r(s) = \mathrm{clamp}_{[1,3]}(1 + \alpha g(s))$, encouraging larger neighborhoods in sparse or weakly clustered regions and smaller ones near dense hubs.

| Symbol                          | Meaning                                                                       | Notes                                                                                                                                                                                      |
| ------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| $s$                             | The **seed node** from which the local subgraph (ego-net) is grown.           | Chosen randomly, optionally stratified by node degree or type.                                                                                                                             |
| $c_s$                           | The **local clustering coefficient** of node s.                               | Measures how interconnected $s$’s neighbors are. In undirected projection form:$c_s = \frac{2T_s}{k_s(k_s-1)}$, where $T_s$ = number of triangles through s.                         |
| $\kappa_s$                      | The **k-core index** (or **core number**) of node s.                          | Largest integer k such that s belongs to the k-core subgraph (all nodes with degree ≥ k). Reflects local structural “embeddedness.”                                                  |
| $\deg_s$                        | The **degree** of node s.                                                     | Number of edges incident on s (can be total degree or weighted by relation type in multiplex setting).                                                                                   |
| $z(\cdot)$                      | The **z-score normalization** operator.                                       | For any scalar node-level metric $(x_s):(z(x_s) = \frac{x_s - \mu_x}{\sigma_x})$, where $\mu_x$ and $\sigma_x$ are the mean and standard deviation of x across all nodes in the graph. |
| $g(s)$                          | The **composite connectivity score** used to adaptively choose hop radius.    | Higher $g(s)$ → larger $r(s)$; encourages exploring sparser regions more deeply.                                                                                                           |
| $r(s)$                          | The **hop radius** (1–3) used to define the ego-net subgraph around seed s.   | Determined by scaling $g(s)$ via $\alpha$ and clamping to the range [1,3]).                                                                                                                |

Each resulting subgraph consisted of the induced r-ball around the seed nodes, preserving local connectivity patterns and approximate node-type proportions without enforcing a fixed target size.
This connectivity-aware hopping strategy generates a controlled distribution of partially overlapping ego-net subgraphs that collectively cover the full multiplex network, ensuring exposure to diverse local structures while maintaining coherence across samples.

Parameters of the rGCN-SCAE were shared across subgraphs to maintain a single shared partitioning in the latent space.
Since node attributes (text-derived embeddings, types, biomarkers, etc.) are stable across subgraphs, meaningful structure can still be derived despite the lack of full context in each training example.
The final partitioning is derived from running the full knowledge graph through the trained rGCN-SCAE.
This procedure reframes training as an information-theoretic compression task applied repeatedly to partially overlapping realizations of the same knowledge manifold, allowing estimation of replication reliability and consensus structure while reducing overfitting to any single instantiation.

#### Comparison
Together, hSBM offers a likelihood-grounded categorical perspective, while rGCN-SCAE furnishes a continuous latent manifold amenable to downstream regression or spectrum analysis.
The two approaches are treated as triangulating evidence: concordant structure across them increases confidence in emergent transdiagnostic clusters, whereas divergences highlight fronts for qualitative review.

| Aspect                        | Recurrent Graph Convoplutional Network Self-Compressing Autoencoder (rGCN-SCAE)                                                                                                | Hierarchical Stochastic Block Model (hSBM)                                                                                  |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Representation**            | Learns *continuous latent embeddings* for nodes and relations; nonlinear, differentiable, expressive.                                        | Assigns *discrete cluster memberships* via probabilistic inference on connectivity patterns.                              |
| **Objective**                 | Minimizes reconstruction loss → learns information-optimal embeddings that compress the multiplex graph while preserving semantic structure. | Maximizes likelihood under a generative model → partitions graph to best explain edge densities between groups.             |
| **Adaptivity**                | Learns directly from heterogeneous, weighted, typed edges and can incorporate node attributes, features, and higher-order dependencies.      | Operates purely on adjacency structure (and possibly metadata) assuming a fixed parametric form (block interaction matrix). |
| **Scalability & Flexibility** | Scales gracefully to large multiplex graphs via minibatch training and GPU acceleration; can integrate multiple modalities in future.                  | Computationally expensive for deep hierarchies; inference is typically O(N²), with N = #nodes in graph, and difficult to extend across modalities.     |
| **Output**                    | Produces a *latent manifold* where distances encode both structural and semantic similarity — enabling *continuous transdiagnostic spectra*. | Produces discrete, possibly hierarchical clusters — enforcing categorical partitions reminiscent of DSM-like divisions.     |

## Abbreviations
- DSM = Diagnostic and Statistical Manual of Mental Disorders
- HiTOP = Hierarchical Taxonomy of Psychopathology
- hSBM = Hierarchical Stochastic Block Model
- ICD = International Classification of Diseases
- RDoC = Research Domain Criteria
- RGCN = Relational Graph Convolutional Network
- SCAE = Self-Compressing Auto-Encoder

## References
1. R. Kotov et al., “The Hierarchical Taxonomy of Psychopathology (HiTOP): A dimensional alternative to traditional nosologies,” J. Abnorm. Psychol., vol. 126, no. 4, pp. 454–477, 2017.
1. N. Haslam, M. J. McGrath, W. Viechtbauer, and P. Kuppens, “Dimensions over categories: A meta-analysis of taxometric research,” Psychol. Med., vol. 50, no. 9, pp. 1418–1432, 2020.
1. B. N. Cuthbert and T. R. Insel, “Toward the future of psychiatric diagnosis: The seven pillars of RDoC,” BMC Med., vol. 11, no. 126, pp. 1–8, 2013.
1. R. Kapadia, R. R. Parikh, and A. G. Vahia, “Limitations of classification systems in psychiatry: Why DSM and ICD should be more dimensional, contextual, and culturally sensitive,” Dialogues Clin. Neurosci., vol. 22, no. 1, pp. 15–23, 2020.
1. A. Aftab and E. Ryznar, “Conceptual and historical evolution of psychiatric nosology,” Int. Rev. Psychiatry, vol. 33, no. 5, pp. 425–433, 2021.
1. L. Parkes, T. D. Satterthwaite, and D. S. Bassett, “Towards precise resting-state fMRI biomarkers in psychiatry: An overview of progress and open challenges,” arXiv preprint arXiv:2006.04728, 2020.
1. M. Goodkind et al., “Identification of a common neurobiological substrate for mental illness,” JAMA Psychiatry, vol. 72, no. 4, pp. 305–315, 2015.
1. W. R. Ringwald, H. R. Snyder, M. C. Keller, et al., “Meta-analysis of structural evidence for the Hierarchical Taxonomy of Psychopathology (HiTOP) model,” Psychol. Med., pp. 1–13, 2023.
1. D. Hirjak, R. C. Wolf, et al., “Linking clinical and biological insights to advance transdiagnostic psychiatry,” Biol. Psychiatry: Global Open Science, vol. 5, no. 1, pp. 14–27, 2025.
1. C. G. DeYoung, K. G. Patrick, et al., “The Hierarchical Taxonomy of Psychopathology (HiTOP) and the neurobiology of mental illness,” Front. Psychiatry, vol. 15, 2024, Art. no. 11529694.
1. A. Caspi and T. E. Moffitt, “All for one and one for all: Mental disorders in one dimension,” Am. J. Psychiatry, vol. 175, no. 9, pp. 831–844, 2018.
1. D. Bzdok and A. Meyer-Lindenberg, “Computational nosology and precision psychiatry,” Biol. Psychiatry, vol. 82, no. 6, pp. 421–430, 2017.
1. G. Starke, L. De Clercq, and F. Schülein, “Computational psychiatry: What do experts think?,” Biol. Psychiatry Global Open Sci., vol. 3, no. 3, pp. 305–313, 2023.
1. G. Zhou et al., “Classifying diseases by using biological features to identify potential nosological models,” Sci. Rep., vol. 11, no. 1, Art. no. 21613, 2021.
1. R. P. Roussel et al., “OMOP2OBO: Ontologizing health systems data at scale,” npj Digit. Med., vol. 6, no. 1, Art. no. 55, 2023.
1. L. Wang et al., “BIOS: An algorithmically generated biomedical knowledge graph,” arXiv preprint arXiv:2203.09975, 2022.
1. W. Wei et al., “NetMoST: A network-based machine learning approach for subtyping schizophrenia using polygenic SNP allele biomarkers,” arXiv preprint arXiv:2305.07005, 2023.
1. D. Drysdale et al., “Resting-state connectivity biomarkers define neurophysiological subtypes of depression,” Nat. Med., vol. 23, pp. 28–38, 2017.
1. S. Gao, K. Yu, Y. Yang, S. Yu, C. Shi, X. Wang, N. Tang, and H. Zhu, “Large language model powered knowledge graph construction for mental health exploration,” Nature Communications, vol. 16, no. 1, Art. no. 7526, 2025, doi: 10.1038/s41467-025-62781-z.
1. T. M. Sweet, A. C. Thomas, and B. W. Junker, “Hierarchical mixed membership stochastic blockmodels for multiple networks and experimental interventions,” in Handbook of Mixed Membership Models and Their Applications, E. Airoldi, D. Blei, E. Erosheva, and S. Fienberg, Eds. Boca Raton, FL, USA: Chapman & Hall/CRC Press, 2014, pp. 463–488.
1. M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, and A. Smola, “Deep Sets,” in Proc. 31st Conf. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 3391–3401.
1. C. Louizos, M. Welling, and D. P. Kingma, “Learning Sparse Neural Networks through L₀ Regularization,” arXiv preprint arXiv:1712.01312, 2017, presented at ICLR 2018.
