# Automated Psychiatric Nosology
A novel method for automating nosology creation (alternative to DSM, HiTOP and RDoC) based on mining findings from scientific papers into a multiplex graph then partitioning in a way that minimizes the number of bits required to represent the partitioning to create the dimensional classification scheme.

Multiplex Graph Design
- Node Types: symptoms, diagnoses, treatments, metrics and biomarkers
- Edge Types: support for / against, prediction, co-occurrence, etc.
- Edge property for the number of relevant papers

## Background
Categorical classification standards for psychiatry such as DSM-5 (the current taxonomy for psychiatric nosology in the U.S.) attempt to impose strict yes/no boundaries on normal vs pathological (Kotov et al. 2017) despite evidence that shows most psychiatric diagnoses exist as a continuous spectrum of behaviors (Haslam et al. 2020). Attempts have been made to remedy this in the form of dimensional diagnostic standards like HiTOP (Kotov et al. 2017) and RDoC (Cuthbert, Insel 2013). These represent two unique paths to improved philosophies on psychiatric nosology but this new technique would be a third. HiTOP is derived from clustering symptoms and diagnoses only, which provides a strong data-driven approach but misses important aspects of nosology and RDoC derives from expert consensus on neuroscience, which is manually intensive. Using graph partitioning and information theory for psychiatric nosology is intrinsically data-driven and allows the incorporation of many different ways of thinking.

## Research Question
Can a proof-of-concept dimensional psychiatric nosology be developed in an automated way by mining the scientific literature into a multiplex knowledge graph and partitioning it using information-theoretic methods, and how does its structure compare with HiTOP and RDoC in terms of parsimony, stability, and alignment?

## Hypothesis
It can be developed in this way and the resulting nosology will roughly have parsimony (as measured by minimum description length) within 95% or better of, stability (as measured by bootstrapped variation of information and bootstrapped adjusted rand index) >= 85% of and alignment (as measured by normalized mutual information and adjusted rand index) >= 75% of HiTOP and RDoC. The parsimony will be so close to or better than HiTOP and RDoC due to the use of information-theoretic algorithms in the partitioning to optimize for compression. The stability will be high due to the large volume of papers expected to be parsed. The alignment will be reasonably high but not exact due to the vastly different methods in producing the final output (RDoC is mechanistically focused while HiTOP is focused on symptoms and this novel method will be automated).

## Preventing Biased Alignment
Because the alignment metrics used to compare the emergent nosology with established frameworks (HiTOP and RDoC) can be artificially inflated if the same vocabulary appears in both the input data and the target taxonomies, terms are explicitly removed from the existing nosological systems before graph construction. A lexicon of DSM, ICD, RDoC and HiTOP terms is used to mask those tokens from the text prior to keyphrase extraction and edge formation. Nodes are then defined purely by content nouns and predicates extracted from scientific findings, and relation types are induced without any pre‑defined nosology labels. Only after the final partitioning is complete will the alignment metrics be computed such as normalized mutual information and adjusted rand index against HiTOP and RDoC categories. This ensures that any observed alignment reflects genuine structural similarities rather than trivial lexical overlap, preventing a biased alignment metric.

## Unanswered Questions
- Which partitioning algorithms to test

## References
- Haslam, N., McGrath, M. J., Viechtbauer, W., & Kuppens, P. (2020). Dimensions over categories: a meta-analysis of taxometric research. Psychological medicine, 50(9), 1418–1432. https://doi.org/10.1017/S003329172000183X
- Kotov, R., Krueger, R. F., Watson, D., Achenbach, T. M., Althoff, R. R., Bagby, R. M., Brown, T. A., Carpenter, W. T., Caspi, A., Clark, L. A., Eaton, N. R., Forbes, M. K., Forbush, K. T., Goldberg, D., Hasin, D., Hyman, S. E., Ivanova, M. Y., Lynam, D. R., Markon, K., … Zimmerman, M. (2017). The hierarchical taxonomy of psychopathology (HiTOP): A dimensional alternative to traditional nosologies. Journal of Abnormal Psychology, 126(4), 454–477. https://doi.org/10.1037/abn0000258
- Cuthbert, B. N., & Insel, T. R. (2013). Toward the future of psychiatric diagnosis: The seven pillars of RDoC. BMC Medicine, 11, 126. https://doi.org/10.1186/1741-7015-11-126