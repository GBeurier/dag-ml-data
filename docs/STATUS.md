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
- sample relation validation for groups, repetitions and augmentation origins;
- sample relation fingerprints;
- coordinator data-plan envelope export with schema, plan and relation
  fingerprints;
- explicit coordinator data-plan envelope schema version with unsupported
  version refusal;
- published JSON Schema artifact for coordinator data-plan envelopes, with a
  unit smoke that keeps its declared version aligned to the Rust contract;
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
- feature exports now validate that the feature table representation matches
  the materialized data-plan output representation for the parent view handle;
- CLI `materialize-envelope` smoke command;
- Arrow C Data ABI structs, release helpers and coordinator identity-table
  export from validated envelopes;
- Arrow C Data numeric target-table export through materialized view handles;
- Arrow C Data numeric feature-table export through materialized view handles;
- Rust-owned in-memory provider vtable with materialize, make-view,
  view-identity, target, feature, release and destroy callbacks;
- provider vtable release conformance, including parent data-handle release
  invalidating child view handles;
- C header and linked C runtime smokes for the provider vtable and Arrow
  signatures;
- Python `ctypes` smoke for the same provider lifecycle;
- reusable stdlib-only Python provider wrapper and CLI smoke in
  `examples/python`;
- C ABI schema fingerprint entry point;
- example schema fixture;
- CI workflow.

Not implemented yet:

- production runtime data providers with real buffer/view lifecycles;
- full production Arrow feature-buffer provider implementation beyond JSON
  conformance smokes;
- alignment and fusion execution;
- fitted adapter serialization;
- nirs4all connector.

Next recommended task:

Add native provider conformance against the in-memory provider vtable, then
replace JSON feature-table smokes with real provider-owned Arrow buffer
lifecycles.
