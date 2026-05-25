# Status

Current state: foundation scaffold.

Implemented:

- Rust workspace with core, facade, C ABI and CLI crates;
- identifier, axis, representation, source and schema types;
- data view, presence mask and data-plan structs;
- schema validation and deterministic fingerprinting;
- C ABI schema fingerprint entry point;
- example schema fixture;
- CI workflow.

Not implemented yet:

- adapter registry and path solver;
- runtime data providers;
- Arrow identity export;
- alignment and fusion execution;
- fitted adapter serialization;
- nirs4all connector.

Next recommended task:

Implement `AdapterRegistry` and a path solver with fixtures derived from
`docs/design/source/ml_data_specification_v1.md` sections 5-7.
