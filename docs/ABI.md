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
- `dagmldata_inmemory_provider_new_json` for a Rust-owned provider vtable that
  materializes data handles, creates view handles, exports view identity, exports
  numeric targets and supports release/destroy callbacks;
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

## In-Memory Provider VTable

The in-memory provider is the current ABI conformance target. It accepts one
validated coordinator envelope plus optional sample-level target tables, then
implements:

- `materialize`: validates a coordinator materialization request and returns an
  opaque data handle;
- `make_view`: applies a `DataView` to a data handle and returns an opaque view
  handle;
- `view_identity`: returns the filtered relation table as Arrow C Data;
- `target_arrow`: returns sample-level numeric targets aligned to the view;
- `release` and `destroy`: release handles and provider state.

It still does not own heavy feature buffers. Full provider implementations will
use the same vtable shape while keeping data buffers host-owned.

`tests/c_header_smoke.rs` has two C checks: a header syntax smoke with
`cc -fsyntax-only`, and a linked C program that loads the Rust `cdylib`, creates
the provider vtable, materializes a view, exports identity/target Arrow arrays
and releases all handles.

`tests/python_ctypes_smoke.rs` performs the same provider lifecycle from Python
using only `ctypes`. It also runs `examples/python/provider_smoke.py`, which uses
`examples/python/dag_ml_data_provider.py` as a small reusable wrapper around the
current provider vtable. This is intentionally not the final Python package API:
it is the binding-friendly conformance target for materialize, view creation,
identity export, target export, release and destroy.

## ABI Roadmap

1. Freeze byte/string/status conventions.
2. Add C smoke test for schema fingerprinting.
3. Add path-solving and data-plan validation over canonical JSON.
4. Add native host provider conformance and extend the Python provider example
   against feature-buffer lifecycles once they exist.
