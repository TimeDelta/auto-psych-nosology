# Physiology graph adapter and relation handling

The adapter imports a frozen `mechanistic-pathway-learning` graph as the biological
foundation of the nosology experiment. It consumes `nodes.parquet`, `edges.parquet`
and `relation_types.json`. No dependency on the other repository's Python package
or its current symptom set is required. New node types, relation types and source
columns remain available, including fields added by the complex-membership fix.

## Initial relation representation

Keep the source relation and sign separately. The default `typed-signed` model
encoding combines them into categorical R-GCN relations:

| Source relation | Recorded sign | Model relation |
| --- | --- | --- |
| `regulates_transcription_of` | +1 | `regulates_transcription_of::sign=positive` |
| `regulates_transcription_of` | -1 | `regulates_transcription_of::sign=negative` |
| `binds` | 0 | `binds::sign=unsigned` |
| Any declared relation | missing | `<relation>::sign=unknown` |

Positive and negative records remain separate even when they join the same pair.
The source's signs have relation-specific meanings: +1 on a substrate edge does
not assert that increasing the substrate causes a symptom. Zero is unsigned,
not a claim of no effect. Missing signs are never converted to positive or zero.
Signs must be -1, 0, +1 or missing; affinity or magnitude belongs in another column.

The loader consumes `model_relation` when present, then the legacy aliases
`predicate`, `relation` or `relation_type`. Conflicting aliases fail explicitly.
`predicate` can differ from the raw relation only with an explicit `model_relation`.
Wholly untyped legacy edges retain the generic `rel` category. Relation indices
are sorted by name and recorded in the adapter's relation map. Historical
checkpoints built with discarded relation types or earlier relation indexing
must be retrained; changed indices cannot be substituted into saved weights.

## Options for the experiment

| Option | What it preserves or models | Use |
| --- | --- | --- |
| Categorical relation plus sign, current R-GCN | Distinct relation/sign categories with learned transforms | Recommended first implementation; default adapter mode |
| Typed relation only | Direction and relation, with sign retained for audit | `--relation-mode typed`; tests what the sign distinction adds |
| Factorized relation/sign message passing | Relation transform plus explicit signed effect, magnitude or context gates | Candidate next encoder; requires relation-specific semantics and tests |
| Edge-conditioned message passing | Learned transform from continuous edge features | Candidate when usable stoichiometry, affinity or evidence features are present |

[R-GCN](https://arxiv.org/abs/1703.06103) learns relation-specific transforms and
supports parameter sharing through basis decomposition. PyG's
[RGCNConv interface](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGCNConv.html)
accepts `edge_type`; it does not accept per-edge sign or confidence arguments.
Categorical negative relations therefore do not enforce negative propagation.
The existing encoder also does not use `edge_weight` in its convolution calls.
Putting confidence into that field alone would not weight its messages.

An explicit signed encoder should distinguish regulation from substrate
consumption, product formation, transport and membership. It should keep
unknown signs separate and allow contradictory evidence rather than averaging
it into a single confident edge. A relation/sign factorization can share
parameters across signs; multiplying every relation by its sign is unsuitable
for unsigned binding, and evidence scores are not calibrated causal strengths.

[NNConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.NNConv.html)
provides an edge-conditioned transform, but adds parameters and does not supply
biochemical semantics. The [Signed GCN paper](https://arxiv.org/abs/1808.06354)
uses balance theory. Its assumptions about signed social relations do not, by
themselves, justify that architecture for biochemical activation and inhibition.
These are design judgments to evaluate, not established performance rankings.

Direction is preserved exactly. The adapter adds no reverse edges. If an encoder
needs inverse edges for information exchange, construct them as explicitly
marked inverse model relations; they are computational edges, not additional
biochemical evidence. Keep them out of evidence counts and causal-path readings.

## Usage and experiment pins

From the nosology repository, with a separate checkout of the physiology project:

```bash
python adapt_physiology_graph.py \
  ../mechanistic-pathway-learning/data/releases/v0.4/graph_full \
  data/physiology_v04
```

Replace the source path with the new released graph, including its gene/protein
split and corrected membership semantics, when available. This old release is
a compatibility fixture, not the selected input for the new experiment.

The first build prints `source_snapshot_sha256`. Freeze that value in the
experiment configuration and require it for subsequent builds:

```bash
python adapt_physiology_graph.py SOURCE_GRAPH_DIRECTORY data/physiology \
  --expected-snapshot-sha256 SNAPSHOT_SHA256
python train_rgcn_scae.py data/physiology.graphml --help
```

An adjacent release `MANIFEST.json` is discovered automatically, or supplied with
`--source-manifest`. Each imported file must match its release checksum. A
snapshot pin hashes the filenames and full-file checksums, including the source
graph summary if present. The output manifest retains the release metadata and
checksum, so a producing commit is recorded when the release supplies one.
Sources are checked again before publishing staged outputs. A failed pin,
manifest or structural check leaves earlier output artifacts untouched.

Artifacts:

- `.graphml`: directed multigraph, original node identifiers, raw edge relations,
  signs, source fields and explicit model relations.
- `.nodes.parquet` and `.rels.parquet`: byte-identical copies of the original
  source tables, not the legacy extractor's table schema.
- `.relations.json`: deterministic model vocabulary, raw relations, observed
  sign categories and edge-row counts.
- `.quality.json`: connectivity, type/relation distributions and sign coverage.
- `.manifest.json`: source snapshot, release metadata and output checksums.

Duplicate node identifiers, missing endpoints, undeclared relations and explicit
disease or nosology node types fail validation. Distinct compartment copies,
parallel source rows and isolates are retained. Edge-row counts are not counts
of independent papers. GraphML has scalar attributes; nested values are encoded
as JSON, while copied Parquet files preserve the exact source bytes.

This is a graph adapter, not evidence assembly or a mechanistic encoder port.
The current trainer does not automatically use every preserved node attribute,
complex gate or context field. The new evidence release should be imported as a
separate versioned layer: require valid perturbation-node mappings, symptom
coverage, source/reference deduplication and unchanged held-out evaluation
boundaries. Diagnosis identifiers can remain in evidence provenance and leakage
groups without becoming graph features or targets.

The new evidence specification already expands the older symptom crosswalk.
Its adequacy should be assessed on the table frozen for this experiment rather
than the old v0.4 table. An increased number of rows does not by itself establish
independent support for each symptom or resolve construct validity.

The defective original KG is not a required benchmark. More useful comparisons
hold the new graph and evidence fixed and test relation/sign encoding, layer
ablations, resampling stability and type-appropriate rewiring controls.
