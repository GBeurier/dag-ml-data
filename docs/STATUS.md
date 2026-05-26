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
- conversion from `SampleRelationTable` to DAG-ML coordinator relation records;
- coordinator materialization request and handle-record contracts for validated
  data handles;
- in-memory coordinator handle arena that materializes envelopes into opaque
  handle records with run/node/phase/variant/fold/fingerprint traceability;
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
- provider `feature_arrow` accepts JSON fusion selectors and routes
  source-filtered provider-owned feature buffers through the core feature-fusion
  kernel;
- executable numeric late-collation kernel for feature blocks and ragged numeric
  rows, producing row-major tensor blocks with deterministic padding,
  truncation, presence masks and value-validity masks;
- provider-backed feature collation over in-memory provider typed buffers,
  including fusion selectors, returning deterministic JSON `NumericTensorBlock`
  or ABI-owned `DagMlDataTensorF64` output without changing the provider vtable
  layout;
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

- production runtime data providers with real buffer/view lifecycles;
- production Arrow feature-buffer provider implementation beyond the current
  in-memory typed numeric buffer conformance;
- production provider lifecycles for fused feature exports and production tensor
  buffer export beyond the in-memory `DagMlDataTensorF64` conformance;
- fitted adapter serialization;
- nirs4all connector.

Next recommended task:

Attach real production feature-buffer lifecycles for single-source, fused and
collated tensor exports, then extend the tensor ABI beyond f64 row-major blocks
as needed.
