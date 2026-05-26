# Shared Contracts

This directory contains wire-contract artifacts shared with `dag-ml`.
`dag-ml-data` is the producer for the current coordinator data-plan envelope:
it describes schemas, relations, data plans and fingerprints without owning
folds, OOF prediction blocks, model selection or replay decisions.

## Coordinator Data Plan Envelope v1

Schema: `coordinator_data_plan_envelope.schema.json`

Runtime type produced here: `CoordinatorDataPlanEnvelope`

Consumer type in `dag-ml`: `ExternalDataPlanEnvelope`

The envelope binds a data plan to stable schema, plan and relation
fingerprints. It may carry coordinator relation records for sample, target,
group, origin, source and augmentation identity. The JSON Schema documents the
portable shape of that envelope; Rust validation enforces the stronger semantic
rules owned by this crate, and `dag-ml` applies campaign-specific OOF/leakage
checks after consuming it.

Short-term policy: both repositories keep a copy of the v1 schema and test that
the published artifact declares the Rust-supported version. When development
moves into a monorepo, this file should become a single generated or shared
contract artifact used by both crates.
