# Status

Current state: foundation scaffold plus coordinator data-plan envelope,
materialized data/view handle smokes, target alignment smoke and minimal Arrow
identity/target/feature export.

Implemented:

- Rust workspace with core, facade, C ABI and CLI crates;
- identifier, axis, representation, source and schema types;
- data view, presence mask and data-plan structs;
- schema validation and deterministic fingerprinting;
- adapter registry and deterministic path solver;
- model-input data-plan planner with fixtures;
- sample alignment planner for `inner`, `left` and `outer` multi-source
  policies, plus planner-visible `Align` steps before multi-source joins;
- sample relation validation for groups, repetitions and augmentation origins;
- sample relation fingerprints;
- coordinator data-plan envelope export with schema, plan and relation
  fingerprints;
- explicit coordinator data-plan envelope schema version with unsupported
  version refusal;
- published JSON Schema artifact for coordinator data-plan envelopes, with a
  unit smoke that keeps its declared version aligned to the Rust contract;
- stdlib shared-contract validation script plus CI checkout of `dag-ml` so
  schema copies and coordinator fixtures cannot drift silently;
- shared conformance-pack manifest with canonical schema/fixture digests, C ABI
  requirements and required cross-repo checks, kept JSON-identical with
  `dag-ml`;
- conversion from `SampleRelationTable` to DAG-ML coordinator relation records;
- coordinator materialization request and handle-record contracts for validated
  data handles;
- in-memory coordinator handle arena that materializes envelopes into opaque
  handle records with run/node/phase/variant/fold/fingerprint traceability;
- materialized data handles scope coordinator relations to the request
  `source_ids`, so child views cannot expose observations from sources that
  were not materialized;
- identity-filtered view handles from `DataView`, preserving repeated
  observations while allowing sample/source/augmentation selection and honoring
  explicit requested sample order;
- sample-level target value alignment for view handles with deterministic
  de-duplication across repeated observations in view sample order;
- observation-level feature table alignment for view handles, preserving
  repeated observations, applying `DataView.columns` and keeping view sample
  order;
- executable observation-level feature fusion for aligned multi-source feature
  blocks, including namespaced columns, repetition-preserving reference rows,
  singleton-source broadcast, deterministic outer synthetic rows and explicit
  refusal of ambiguous repeated non-reference sources;
- feature exports now validate that the feature table representation matches
  the materialized data-plan output representation for the parent view handle;
- CLI `materialize-envelope` smoke command;
- Arrow C Data ABI structs, release helpers and coordinator identity-table
  export from validated envelopes;
- Arrow C Data numeric target-table export through materialized view handles;
- Arrow C Data numeric feature-table export through materialized view handles;
- Arrow C Data numeric feature-fusion export for aligned multi-source feature
  blocks;
- JSON and ABI-owned f64 numeric feature-collation exports for row-major tensor
  conformance;
- Rust-owned in-memory provider vtable with materialize, make-view,
  view-identity, target, feature, release and destroy callbacks;
- in-memory provider feature tables are converted once at provider creation
  into typed numeric buffers, so provider exports no longer re-parse JSON values
  on every `feature_arrow` call;
- typed numeric feature buffers now live in `dag-ml-data-core` as reusable
  column-major `NumericFeatureBuffer` contracts with projection tests, rather
  than being private C ABI fixture logic;
- typed row-major `NumericFeatureMatrixF64` input with optional validity masks
  converts directly to column-major buffers without per-cell
  `serde_json::Value` parsing on the numeric conformance path;
- typed numeric feature buffers are grouped behind a core
  `NumericFeatureBufferStore` with deterministic manifests, row/feature/value
  counts, estimated value bytes and stable buffer fingerprints;
- materialized data handles now receive deterministic `NumericFeatureBufferBinding`
  records for the feature buffers whose representation and observation coverage
  match that handle's scoped coordinator relations;
- the store and binding lifecycle now live behind the reusable core
  `NumericFeatureBufferArena`, so production providers can share the same
  manifest, bind, project and release contract instead of reimplementing it in
  the C ABI layer;
- provider `feature_arrow` accepts JSON fusion selectors and routes
  source-filtered provider-owned feature buffers through the core feature-fusion
  kernel;
- provider feature, fusion and collation exports validate that the selected
  buffer is bound to the parent data handle and requested source scope before
  exporting Arrow, JSON tensors or ABI-owned tensors;
- executable numeric late-collation kernel for feature blocks and ragged numeric
  rows, producing row-major tensor blocks with deterministic padding,
  truncation, presence masks and value-validity masks;
- provider-backed feature collation over in-memory provider typed buffers,
  including fusion selectors, returning deterministic JSON `NumericTensorBlock`
  or ABI-owned `DagMlDataTensorF64` output without changing the provider vtable
  layout;
- provider-owned feature-buffer manifests are exported as JSON through the C ABI
  for binding conformance and lifecycle validation;
- provider construction accepts typed f64 feature matrices through the C ABI as
  the preferred conformance path for numeric feature buffers;
- provider construction also accepts borrowed C `DagMlDataFeatureMatrixF64View`
  descriptors, copying them into Rust-owned feature buffers during the
  constructor call so bindings can avoid JSON numeric value transport;
- provider construction also accepts borrowed C
  `DagMlDataFeatureMatrixF64ColumnarView` descriptors with per-column f64
  slices and optional per-column validity bitmaps, mirroring production
  columnar layouts (Arrow IPC, Parquet, NumPy column ndarrays) and avoiding
  the row-major transpose copy paid by the row-major borrowed view path;
- coordinator and provider-backed feature collation now also return owned
  row-major `DagMlDataTensorF32` blocks beside the existing
  `DagMlDataTensorF64` exports. The collation kernel still operates in f64 to
  preserve canonical numeric semantics; values are cast to f32 at the ABI
  boundary and rejected with `ValidationError` if any padded value, finite
  input or padding fallback does not round-trip into a finite f32 (overflow
  to infinity, or non-finite input). The C ABI exposes
  `DAG_ML_DATA_TENSOR_F32_ABI_VERSION`, `dagmldata_tensor_f32_free` and the
  matching `coordinator`/`inmemory_provider` collation entry points;
- `DataView` now carries an optional `branch_view` field shaped as
  `CoordinatorBranchView` (`view_id`, `branch_id`, `mode`, `selector`,
  `allow_overlap`, `metadata`), mirroring `dag-ml`'s `BranchViewPlan` wire
  contract. `make_view` validates the mode↔selector field agreement and
  natively executes `by_source` branch views as an additional intersection
  over the existing source filter. `separation` mode passes through (the
  selector annotates branch identity and the host scheduler owns
  non-overlap), while `by_metadata`, `by_tag` and `by_filter` modes pass
  validation but are rejected for in-memory execution with a clear
  `requires host-side filtering` error so production providers can route
  them to native filter backends without breaking ABI compatibility;
- data-handle-scoped feature-buffer bindings are exported as JSON through the C
  ABI and become invalid when the parent data handle is released;
- provider vtable release conformance, including parent data-handle release
  invalidating child view handles;
- C header and linked C runtime smokes for the provider vtable and Arrow
  signatures;
- `dag_ml_data.h` and `dag_ml.h` share a guarded data-provider vtable ABI
  version macro and are compiled together in both include orders when the
  sibling checkout is present;
- Python `ctypes` smoke for the same provider lifecycle;
- reusable stdlib-only Python provider wrapper and CLI smoke in
  `examples/python`;
- C ABI schema fingerprint entry point;
- example schema fixture;
- CI workflow.

Not implemented yet:

- production runtime data providers backed by non-fixture buffer arenas;
- production Arrow feature-buffer provider implementation beyond the current
  in-memory typed numeric buffer arena conformance;
- production provider arenas for fused feature exports and production tensor
  buffer export beyond the in-memory `DagMlDataTensorF64` conformance;
- fitted adapter serialization;
- nirs4all connector.

Next recommended task:

Publish a standalone `coordinator_branch_view.schema.json` (mirroring the
`branch_view_plan` shape currently inlined in `dag-ml`'s
`campaign_spec.schema.json` `$defs`) plus a cross-repo conformance check
through `scripts/validate_contracts.py`, then add a fitted adapter
serialization contract (`AdapterStateRef`) so refit replay can carry
host-stored adapter binaries the same way artifact references already do for
models.
