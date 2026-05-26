# Status

Current state: foundation scaffold plus coordinator data-plan envelope,
materialized data/view handle smokes, target alignment smoke and minimal Arrow
identity export.

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
- conversion from `SampleRelationTable` to DAG-ML coordinator relation records;
- coordinator materialization request and handle-record contracts for validated
  data handles;
- in-memory coordinator handle arena that materializes envelopes into opaque
  handle records with run/node/phase/variant/fold/fingerprint traceability;
- identity-filtered view handles from `DataView`, preserving repeated
  observations while allowing sample/source/augmentation selection;
- sample-level target value alignment for view handles with deterministic
  de-duplication across repeated observations;
- CLI `materialize-envelope` smoke command;
- Arrow C Data ABI structs, release helpers and coordinator identity-table
  export from validated envelopes;
- Arrow C Data numeric target-table export through materialized view handles;
- Rust-owned in-memory provider vtable with materialize, make-view,
  view-identity, target, release and destroy callbacks;
- C header and linked C runtime smokes for the provider vtable and Arrow
  signatures;
- Python `ctypes` smoke for the same provider lifecycle;
- C ABI schema fingerprint entry point;
- example schema fixture;
- CI workflow.

Not implemented yet:

- production runtime data providers with real buffer/view lifecycles;
- full Arrow feature-buffer provider implementation beyond identity/target
  conformance smokes;
- alignment and fusion execution;
- fitted adapter serialization;
- nirs4all connector.

Next recommended task:

Add native provider conformance and a reusable Python provider package against
the in-memory provider vtable, then attach feature-buffer lifecycles.
