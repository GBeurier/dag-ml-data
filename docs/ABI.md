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
- `DagMlDataVTable` with materialize/view/identity/target/release hooks.

## Ownership Rules

| Object | Owner | Release path |
|---|---|---|
| Materialized data handle | Host | `DagMlDataVTable.release` |
| View handle | Host | `DagMlDataVTable.release` |
| Fitted adapter handle | Host | future fitted-adapter release hook |
| Rust error/fingerprint string | Rust allocation returned through ABI | `dagmldata_string_free` |
| Arrow arrays/schemas returned by Rust helpers | Rust allocation returned through ABI | `dagmldata_arrow_array_free`, `dagmldata_arrow_schema_free` |
| Arrow arrays produced by host vtables | Producer of the Arrow array | Arrow C Data Interface release callback |

## Coordinator Identity Export

`dagmldata_coordinator_identity_arrow_json` is a narrow smoke helper, not the
final provider implementation. It validates a `CoordinatorDataPlanEnvelope` and
exports one Arrow struct row per coordinator relation with:

- `observation_id`, `sample_id`, `target_id`, `group_id`;
- `origin_sample_id`, `source_id`, `is_augmented`.

This is enough for ABI consumers to verify sample/repetition/group/augmentation
identity transfer before full buffer-backed provider lifecycles exist.

## ABI Roadmap

1. Freeze byte/string/status conventions.
2. Add C smoke test for schema fingerprinting.
3. Add path-solving and data-plan validation over canonical JSON.
4. Add provider-vtable identity and target Arrow conformance tests for Python
   and native host data providers.
