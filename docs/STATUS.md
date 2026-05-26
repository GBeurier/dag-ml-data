# Status

Current state: foundation scaffold plus coordinator data-plan envelope and
first handle materialization smoke.

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
- CLI `materialize-envelope` smoke command;
- C ABI schema fingerprint entry point;
- example schema fixture;
- CI workflow.

Not implemented yet:

- production runtime data providers with real buffer/view lifecycles;
- Arrow identity export;
- alignment and fusion execution;
- fitted adapter serialization;
- nirs4all connector.

Next recommended task:

Connect the handle arena to a buffer-backed provider and expose identity tables
through Arrow C Data Interface smoke tests.
