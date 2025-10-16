# Automated Psychiatric Nosology via Partitioning of Multiplex Graph Generated from Mining Scientific Papers for Findings
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
- This project uses python 3.10
- Must install [graph tool](https://graph-tool.skewed.de) in order to run the [create_hSBM_partitions.py](./create_hSBM_partitions.py) script.
- In order to use this for something other than psychiatry, must change `"concepts.id:C61535369",` filter part in [openalex_client.py](./openalex_client.py)

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

### Graph Creation
Multiplex Graph Design
- Node Types: extractor currently emits `Symptom`, `Diagnosis`, `Biomarker`, `Treatment`, and `Measure`
- Structural Relation Types (directed where appropriate): `treats`, `predicts`, `biomarker_for`, `measure_of`
- Evidence Qualifiers: each structural edge stores the NLI evidence label (`supports`, `contradicts`, `replicates`, `null_reported`) plus scores, margins, and an embedded `ClaimDescriptor`
- Orientation Logic: for every node pair, test both subject→object and object→subject hypotheses; if neither satisfies the role constraints, an evidence-backed fallback (e.g., Treatment→Diagnosis as `treats`) is applied so clinically plausible edges persist instead of isolating nodes
- Export Schema: edge qualifiers are flattened as `qual_*` attributes (e.g., `qual_nli_score`, `qual_claim`, `qual_evidence_label`) so GraphML/HTML tooltips surface the underlying evidence without manual JSON parsing

1. Specified queries ("bipolar disorder", "major depressive disorder", "schizoaffective disorder", "anxiety disorder", "PTSD", "OCD", "OCPD") are split into buckets with an extra fetch buffer evenly spread amongst each of the queries in case an error occurs during download or processing
1. Fetch text from OpenAlex (falling back to PMC) for top M most cited and N most recent papers for each query, all of which are restricted to have been published in the past decade
1. Pull out only results and discussion section
1. Extraction process (via [Biomedical Stanza](https://stanfordnlp.github.io/stanza/available_biomed_models.html) and [Hugging Face](https://huggingface.co/pritamdeka/PubMedBERT-MNLI-MedNLI))
    1. Tokenization
    1. POS
    1. NER for Nodes - NER exclusion terms
    1. Lemmatization of found NER terms to ensure canonicalization of nodes
    1. Entailment (only Hugging Face part) for relations

### Preventing Biased Alignment
Because the alignment metrics used to compare the emergent nosology with established frameworks (HiTOP and RDoC) can be artificially inflated if the same vocabulary appears in both the input graph and the target taxonomies, nodes corresponding to existing nosological systems are explicitly removed from the graph before partitioning.
They are left in the original graph in order to make alignment calculations easier.
This preserves semantically coherent nodes (symptoms, biomarkers, treatments, etc.) while keeping diagnostic labels available for later alignment checks and ensuring that the resulting graph structure emerges independently of existing nosological vocabularies.
This entity-level masking substantially reduces over-masking and yields more biologically meaningful connectivity patterns versus a simple token-level method.
Only after the final partitioning is complete will alignment metrics such as normalized mutual information and adjusted Rand index be computed against HiTOP and RDoC categories.
This ensures that any observed alignment reflects genuine structural similarities rather than trivial lexical overlap, preventing a biased alignment metric.

### Partitioning
Two complementary strategies were tested for discovering mesoscale structure in the multiplex psychopathology graph. First, a hierarchical stochastic block model (hSBM) [19], which provides a probabilistic baseline that infers discrete clusters by maximizing the likelihood of observed edge densities across multiple resolution levels.
This family of models gives interpretable, DSM-like partitions together with principled estimates of uncertainty, but inherits hSBM’s familiar computational burdens—quadratic scaling in the number of vertices and a rigid parametric form for block interactions.
This was used as a baseline.

Second, a **recurrent Graph Convolutional Network Self-Compressing Autoencoder (rGCN-SCAE)** tailored to the heterogeneous, typed graph.
The encoder is a recurrent GCN that outputs a #Nodes x #Clusters latent assignment matrix.
This latent space is partitioned by Louizos et al.’s hard-concrete (L0) gates [20].
Cluster gates drive automatic selection of the surviving latent clusters.
Relation-specific inter-cluster gates and matrices learn which cluster-to-cluster connections matter for each edge type, allowing the model to treat, for example, “symptom ↔ diagnosis” edges differently from “treatment ↔ biomarker” edges.
For each relation type, a distinct decoder is trained to reconstruct edges of that type from the shared latent embeddings.
This design allows relation-specific geometries to be learned while maintaining a common latent space.
A single decoder conditioned on relation embeddings would instead force all relation types to share one generative function, often leading to interference and loss of type specificity.
Because the encoder operates directly on the supplied `edge_index`, the model supports cyclic connectivity and multiplex relation types without special handling.
The decoder mirrors this flexibility, enabling reconstruction of directed feedback motifs that are pervasive in psychiatric knowledge graphs.

#### Encoder Architecture
The encoder is designed to fuse heterogeneous node evidence into a size-robust latent representation suitable for subsequent sparsification.
Each node is first endowed with a permutation-invariant semantic descriptor obtained from a DeepSete node attribute encoder, which implements the Zaheer et al. $\phi/\rho$ formulation.
This component admits arbitrarily structured metadata and expands its vocabulary online, ensuring that newly mined attributes can be assimilated without retraining embeddings from scratch.

The semantic descriptors are concatenated with two additional signals: (i) a learned node-type embedding that preserves categorical identity while remaining compact enough to avoid over-parameterization, and (ii) graph-aware positional embeddings derived from standardized Laplacian eigenvectors when available.
The latter are normalized per graph to prevent variance inflation when mini-batches mix subgraphs of widely differing order.

Relation-aware message passing is effected through a stack of graph convolutional layers equipped with basis decomposition across relation types.
Each layer is followed by either `GraphNorm` (the default choice, given its empirical stability on highly unbalanced batch compositions) or `LayerNorm`, a ReLU nonlinearity, and optional dropout.
The stack yields node embeddings that remain well-conditioned across sampling regimes, and the module returns both the embeddings and auxiliary bookkeeping tensors (e.g., batch indices) that downstream Sinkhorn balancing and gate sampling procedures consume to form temperature-controlled cluster assignment matrices.

#### Decoder Architecture
To improve expressivity while maintaining the strict size- and permutation-invariance required for subgraph-level training, the rGCN-SCAE decoder was designed using a **Deep Sets–style relational energy function** [21], [22].
Unlike bilinear or message-passing decoders, which respectively underfit or introduce graph-size dependence, this formulation learns an invariant nonlinear mapping between pairs of latent embeddings and their relation type.

Mathematically, for any relation type r and node pair (i,j):

$$
\hat{A}^{(r)}_{ij} =
\sigma\!\Big(
  W_o \,
  \rho\!\Big(
     \sum_{x \in \{\,z_i,\,z_j,\,e_r\,\}}
     \phi(x)
  \Big)
\Big),
$$

where:
- $(z_i, z_j \in \mathbb{R}^d)$ are latent node embeddings produced by the encoder,
- $e_r$ is the learned embedding of relation type r,
- $\phi(\cdot)$ projects each element into a shared latent space,
- the summation (or mean) enforces permutation-invariance over the set $\{z_i,z_j,e_r\}$,
- $\rho(\cdot)$ is a small multilayer perceptron mapping the pooled vector to a scalar edge energy,
- $W_o$ is a linear projection converting that energy to a logit, and
- $\sigma(\cdot)$ is the sigmoid function mapping logits to edge probabilities.

This construction ensures identical edge likelihoods regardless of node ordering and independence from batch size or subgraph composition.
Implementation-wise, the decoder consists of two shared MLPs with hidden width 128–256 and ReLU activations.
A relation embedding table provides $e_r$ for each edge type.
Because the entire function operates on latent vectors rather than adjacency statistics, the reconstruction loss remains strictly size-invariant.
This Deep Sets–based architecture preserves efficiency and stability for contrastive training while providing sufficient capacity to model relation-specific nonlinearities and higher-order dependencies.

#### Preventing Trivial / Degenerate Solutions
Five primary forms of degenerate or trivial solutions are guarded against in this model: uniform embeddings, local collapse, global collapse, decoder memorization, and latent drift.
Absent-edge modeling via negative sampling penalizes trivial solutions such as uniform embeddings that would otherwise predict every node pair as connected.
The decoder explicitly contrasts observed edges with sampled non-edges drawn within each subgraph, forcing the model to assign distinct embeddings that separate real from spurious connections.
That means clusters that lump everything together will get penalized because they will incorrectly predict connections between random node pairs that are not actually linked.
Mathematically, this is an additional loss term for each relation type:

$$
\mathcal{L}
= -\sum_{(i,j)\in E} \log \sigma\big(z_i^\top W z_j\big) - \sum_{(i',j')\notin E} \log \big[1-\sigma\big(z_{i'}^\top W z_{j'}\big)\big]
$$

| Symbol                                      | Meaning                                                                                                                                                                                                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| $E$                                       | The set of observed positive edges in the multiplex graph. Each edge (i,j) indicates a real relation between node (i) and node (j).                                                                                                                      |
| $(i', j') \notin E$                       | A set of *absent* edges (negative samples) — pairs of nodes that are *not* connected in the observed graph. These are drawn randomly (or stratified by node type) during training to contrast with positives.                                              |
| $z_i \in \mathbb{R}^d$                    | Latent embedding vector for node (i) produced by the encoder (the rGCN-SCAE). Dimension (d) is the latent feature size.                                                                                                                                      |
| $W \in \mathbb{R}^{d \times d}$           | Learnable relation (or global decoder) weight matrix that maps latent interactions to a scalar edge score. In a multiplex graph, there may be one such matrix per relation type (W_r).                                                                       |
| $\sigma(\cdot)$                           | Logistic sigmoid function, $\sigma(x) = 1/(1 + e^{-x})$, mapping inner products to edge probabilities.                                                                                                                                                       |
| $\log \sigma(z_i^\top W z_j)$             | Log-probability that the observed edge (i,j) is correctly reconstructed.                                                                                                                                                                                   |
| $\log [1 - \sigma(z_{i'}^\top W z_{j'})]$ | Log-probability that a *non-edge* is correctly predicted as absent.                                                                                                                                                                                          |
| $\mathcal{L}$                             | Total reconstruction loss — the binary cross-entropy objective over both observed and sampled absent edges. Minimizing $(\mathcal{L})$ trains embeddings $(z_i)$ so that connected nodes have high dot products, while unconnected ones have low similarity. |

Intuitively, the first term rewards high similarity for real edges while the second penalizes the model when it predicts high similarity for random, non-existent connections.

However, this is not enough to prevent all situations of structure collapse.
For example, a situation could arise where local structure collapses but global does not.
The model could produce a few large "meta-clusters" that reconstruct edges well but lack finer internal structure — all symptoms in one, all treatments in another, etc.
Negative sampling does not penalize this because such coarse partitions can still separate positives from random negatives effectively.
Entropy regularization on the cluster assignment matrix counteracts this tendency by discouraging prematurely confident, low-entropy assignments.
It maintains dispersion across latent clusters by keeping node-to-cluster probabilities sufficiently soft during early training, ensuring that multiple clusters remain active and receive gradient signal until the structure is well-formed.
Mathematically, this loss term is:

$$
\mathcal{L}_H = -\frac{1}{N} \sum_{i=1}^{N}\sum_{k=1}^{K} p_{ik}\,\log p_{ik}
$$

| Symbol     | Meaning                                                                                                     |
| ---------- | ----------------------------------------------------------------------------------------------------------- |
| $K$        | Maximum number of latent clusters.                                                                          |
| $N$        | Number of nodes (data points).                                                                              |
| $p_{ik}$   | Soft assignment probability of node ( i ) to cluster ( k ).                                                 |

However, the entropy regularization only handles local degeneracy because it acts locally.
For global degeneracy, a Dirichlet prior loss term is used because it constrains the aggregate cluster sizes across all nodes, discouraging the emergence of a few dominant mega-clusters while allowing moderate, data-driven variation in cluster sizes.
This global prior complements the local entropy term by balancing overall cluster utilization, ensuring that each cluster contributes meaningfully to the reconstruction objective without enforcing strict uniformity.
The equation for this loss term is:

$$
\mathcal{L}_{\text{Dirichlet}}
= -\sum_{k=1}^{K} (\alpha_k - 1)\,
  \log\left(
    \frac{1}{N}\sum_{i=1}^{N} p_{ik}
  \right)
$$

| Symbol     | Meaning                                                                                                     |
| ---------- | ----------------------------------------------------------------------------------------------------------- |
| $K$        | Maximum number of latent clusters.                                                                          |
| $N$        | Number of nodes (data points).                                                                              |
| $p_{ik}$   | Soft assignment probability of node ( i ) to cluster ( k ).                                                 |
| $\alpha_k$ | Concentration parameter of the Dirichlet prior; values ( < 1 ) encourage balanced but non-uniform clusters. |

Even with these local and global regularizers, the decoder can still exploit trivial optima by memorizing adjacency patterns rather than learning meaningful structure.
This can occur when embeddings or decoder weights grow unbounded, allowing perfect reconstruction without informative latent geometry.
To prevent this *reconstruction-dominant collapse*, an embedding norm regularization term is added to constrain latent magnitude and encourage geometric smoothness:

$$
\mathcal{L}_{\text{norm}} = \frac{\lambda_z}{N}\sum_{i=1}^{N} \|z_i\|_2^2
$$

| Symbol        | Meaning                                                                                 |
| ------------- | --------------------------------------------------------------------------------------- |
| $\lambda_z$   | Weight controlling the strength of embedding norm regularization.                       |
| $\|z_i\|_2^2$ | Squared L2 norm of the embedding vector for node $(i)$, penalizing large magnitudes.    |

Finally, the latent posterior distribution could much drift or the latent space might collapse into a narrow region.
To prevent this, a simple Kullback–Leibler divergence loss term is added:

$$
\mathcal{L}_{\text{KLD}}
= D_{\mathrm{KL}}\big(q(z_i) \,\|\, p(z)\big)
$$

| Symbol                  | Meaning                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------ |
| $q(z_i)$                | Approximate posterior distribution of latent embedding for node $(i)$.               |
| $p(z)$                  | Prior distribution, typically $\mathcal{N}(0,I)$.                                    |
| $D_{\mathrm{KL}}(q\|p)$ | Kullback–Leibler divergence measuring how far the posterior deviates from the prior. |

#### Stability and Regularization
Training rGCN-SCAE models on multiplex graphs can exhibit several characteristic failure modes.
Detailed mitigation strategies and hyperparameter guidelines are provided in [`training_stability.md`](training_stability.md).

#### Adaptive Subgraph Sampling to Mitigate Overfitting
A practical limitation of the basic rGCN-SCAE approach is that it would be trained on a single, fixed multiplex graph.
Without a distribution of graphs, the model risks overfitting to idiosyncratic topological patterns rather than learning generalizable relational principles.
To mitigate this, a dataset was be created with different subgraphs sampled via node-hopping from randomly chosen seed nodes with the hop radius determined adaptively from local connectivity metrics such as node degree, clustering coefficient, and k-core index in order to preserve local connectivity and type proportions.
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
- rGCN = Recurrent Graph Convolutional Network
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
1. T. M. Sweet, A. C. Thomas, and B. W. Junker, “Hierarchical mixed membership stochastic blockmodels for multiple networks and experimental interventions,” in Handbook of Mixed Membership Models and Their Applications, E. Airoldi, D. Blei, E. Erosheva, and S. Fienberg, Eds. Boca Raton, FL, USA: Chapman & Hall/CRC Press, 2014, pp. 463–488.
1. C. Louizos, M. Welling, and D. P. Kingma, “Learning Sparse Neural Networks through L₀ Regularization,” arXiv preprint arXiv:1712.01312, 2017, presented at ICLR 2018.
1. M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, and A. Smola, “Deep Sets,” in Proc. 31st Conf. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 3391–3401.
1. C. Feinauer and C. Lucibello, “Reconstruction of Pairwise Interactions Using Energy-Based Models,” in Proc. Mach. Learn. Res., vol. 145, 2022, pp. 1–17.
