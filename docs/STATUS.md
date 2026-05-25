# Status

Current state: foundation scaffold plus coordinator data-plan envelope.

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
- C ABI schema fingerprint entry point;
- example schema fixture;
- CI workflow.

Not implemented yet:

- runtime data providers;
- Arrow identity export;
- alignment and fusion execution;
- fitted adapter serialization;
- nirs4all connector.

Next recommended task:

Implement a runtime data provider mock that materializes a validated
coordinator envelope into opaque handles for `dag-ml` controller tasks.
