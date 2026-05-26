# C ABI

The data ABI lets a host runtime keep buffers and fitted data adapters in its own
memory while exposing deterministic descriptors and identity tables to the core.

## Current Scaffold

`crates/dag-ml-data-capi/include/dag_ml_data.h` exposes:

- version and string-free helpers;
- `dagmldata_schema_fingerprint_json`;
- Arrow C Data `ArrowArray` and `ArrowSchema` structs plus release helpers;
- `dagmldata_coordinator_identity_arrow_json` for identity-table smoke tests
  from a validated coordinator envelope;
- `dagmldata_coordinator_target_arrow_json` for numeric target-table smoke tests
  from a validated envelope, materialization request, `DataView` and target
  table;
- `dagmldata_coordinator_feature_arrow_json` for numeric observation-level
  feature-table smoke tests from the same coordinator/view contracts;
- `dagmldata_inmemory_provider_new_json` for a Rust-owned provider vtable that
  materializes data handles, creates view handles, exports view identity, exports
  numeric targets and supports release/destroy callbacks;
- `dagmldata_inmemory_provider_new_with_features_json` for the same provider
  plus JSON feature tables used by binding conformance tests;
- `DagMlDataVTable` with materialize/view/identity/target/feature/release hooks.

The coordinator envelope wire shape is versioned as
`CoordinatorDataPlanEnvelope` v1 and published at
`docs/contracts/coordinator_data_plan_envelope.schema.json`. Runtime validation
continues to check the stronger semantic contract: schema/data-plan/relation
fingerprints, identity consistency and materialization-request compatibility.

## Ownership Rules

| Object | Owner | Release path |
|---|---|---|
| Materialized data handle | Host | `DagMlDataVTable.release` |
| View handle | Host | `DagMlDataVTable.release` |
| Fitted adapter handle | Host | future fitted-adapter release hook |
| Rust error/fingerprint string | Rust allocation returned through ABI | `dagmldata_string_free` |
| Arrow arrays/schemas returned by Rust helpers | Rust allocation returned through ABI | `dagmldata_arrow_array_free`, `dagmldata_arrow_schema_free` |
| Arrow arrays produced by host vtables | Producer of the Arrow array | Arrow C Data Interface release callback |
| Rust-owned in-memory provider vtable | Rust allocation behind `user_data` | `DagMlDataVTable.destroy` or `dagmldata_inmemory_provider_destroy` |

## Coordinator Identity Export

`dagmldata_coordinator_identity_arrow_json` is a narrow smoke helper, not the
final provider implementation. It validates a `CoordinatorDataPlanEnvelope` and
exports one Arrow struct row per coordinator relation with:

- `observation_id`, `sample_id`, `target_id`, `group_id`;
- `origin_sample_id`, `source_id`, `is_augmented`.

This is enough for ABI consumers to verify sample/repetition/group/augmentation
identity transfer before full buffer-backed provider lifecycles exist.

`dagmldata_coordinator_target_arrow_json` extends the smoke path to
sample-level targets. It materializes the envelope, creates a `DataView`, aligns
target values to the selected samples and emits `sample_id`, `target_id` and
numeric `value` columns. Repeated observations are intentionally de-duplicated
to one target value per sample.

`dagmldata_coordinator_feature_arrow_json` is intentionally observation-level.
It materializes the same envelope/view, preserves repeated observations, applies
`DataView.columns`, and emits `observation_id`, `sample_id` plus one numeric
column per selected feature. This keeps the target aggregation rule separate
from feature row identity.

## In-Memory Provider VTable

The in-memory provider is the current ABI conformance target. It accepts one
validated coordinator envelope plus optional sample-level target tables and
observation-level feature tables, then implements:

- `materialize`: validates a coordinator materialization request and returns an
  opaque data handle;
- `make_view`: applies a `DataView` to a data handle and returns an opaque view
  handle;
- `view_identity`: returns the filtered relation table as Arrow C Data;
- `target_arrow`: returns sample-level numeric targets aligned to the view;
- `feature_arrow`: returns observation-level numeric features aligned to the
  view and filtered by `DataView.columns`;
- `release` and `destroy`: release handles and provider state.

The conformance provider still receives small JSON fixture feature tables at
construction time, but it converts them once into typed numeric buffers owned by
the provider state. `feature_arrow` exports are then view projections over those
owned buffers, not per-call JSON numeric parsing. Full provider implementations
will use the same vtable shape while keeping production data buffers host-owned.

`tests/c_header_smoke.rs` has two C checks: a header syntax smoke with
`cc -fsyntax-only`, and a linked C program that loads the Rust `cdylib`, creates
the provider vtable, materializes a view, exports identity, target and feature
Arrow arrays, then releases all handles.

`tests/python_ctypes_smoke.rs` performs the same provider lifecycle from Python
using only `ctypes`. It also runs `examples/python/provider_smoke.py`, which uses
`examples/python/dag_ml_data_provider.py` as a small reusable wrapper around the
current provider vtable. This is intentionally not the final Python package API:
it is the binding-friendly conformance target for materialize, view creation,
identity export, target export, feature export, release and destroy.

## ABI Roadmap

1. Freeze byte/string/status conventions.
2. Add C smoke test for schema fingerprinting.
3. Add path-solving and data-plan validation over canonical JSON.
4. Add native host provider conformance against the current
   identity/target/feature behavior.
5. Replace in-memory typed fixture buffers with production feature-buffer
   lifecycles.
