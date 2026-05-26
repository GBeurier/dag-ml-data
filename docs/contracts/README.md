# Shared Contracts

This directory contains wire-contract artifacts shared with `dag-ml`.
`dag-ml-data` is the producer for the current coordinator data-plan envelope:
it describes schemas, relations, data plans and fingerprints without owning
folds, OOF prediction blocks, model selection or replay decisions.

## Coordinator Data Plan Envelope v1

Schema: `coordinator_data_plan_envelope.schema.json`

Canonical fixture: `examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json`

Runtime type produced here: `CoordinatorDataPlanEnvelope`

Consumer type in `dag-ml`: `ExternalDataPlanEnvelope`

The envelope binds a data plan to stable schema, plan and relation
fingerprints. It may carry coordinator relation records for sample, target,
group, origin, source and augmentation identity. The JSON Schema documents the
portable shape of that envelope; Rust validation enforces the stronger semantic
rules owned by this crate, and `dag-ml` applies campaign-specific OOF/leakage
checks after consuming it.

Short-term policy: both repositories keep a JSON-identical conformance fixture
for this envelope plus a copy of the v1 schema, and test that the published
artifact declares the Rust-supported version. `scripts/validate_contracts.py`
compares the fixture and schema copies when `DAG_ML_REPO` points to a sibling
checkout, and CI checks out that peer explicitly. When development moves into a
monorepo, this file should become a single generated or shared contract artifact
used by both crates.

## Feature Fusion Selector v1

Schema: `feature_fusion_selector.schema.json`

Canonical fixture: `examples/fixtures/oof_campaign/feature_fusion_selector_nir_chem.json`

Runtime shape consumed by the in-memory provider `feature_arrow` hook:
`{ schema_version, feature_set_id, sources, alignment, policy? }`, where each
source maps a `source_id` to a provider-owned `feature_set_id` and optional
column subset. This selector keeps the vtable ABI stable while making
multi-source feature fusion explicit and conformance-testable.
