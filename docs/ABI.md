# C ABI

The data ABI lets a host runtime keep buffers and fitted data adapters in its own
memory while exposing deterministic descriptors and identity tables to the core.

## Current Scaffold

`crates/dag-ml-data-capi/include/dag_ml_data.h` exposes:

- version, string-free and tensor-free helpers;
- `dagmldata_schema_fingerprint_json`;
- Arrow C Data `ArrowArray` and `ArrowSchema` structs plus release helpers;
- `DagMlDataTensorF64`, an owned row-major f64 tensor descriptor with identity,
  shape, values, optional masks and feature names;
- `DAG_ML_DATA_TENSOR_F64_ABI_VERSION`, the C-visible ABI version expected in
  each f64 tensor descriptor;
- `dagmldata_coordinator_identity_arrow_json` for identity-table smoke tests
  from a validated coordinator envelope;
- `dagmldata_coordinator_target_arrow_json` for numeric target-table smoke tests
  from a validated envelope, materialization request, `DataView` and target
  table;
- `dagmldata_coordinator_feature_arrow_json` for numeric observation-level
  feature-table smoke tests from the same coordinator/view contracts;
- `dagmldata_coordinator_feature_fusion_arrow_json` for numeric multi-source
  fused feature-table smoke tests over already materialized coordinator feature
  blocks;
- `dagmldata_coordinator_feature_collation_json` for JSON row-major tensor
  collation smoke tests over coordinator feature blocks;
- `dagmldata_coordinator_feature_collation_tensor_f64_json` for ABI-owned
  row-major f64 tensor export over coordinator feature blocks;
- `dagmldata_inmemory_provider_new_json` for a Rust-owned provider vtable that
  materializes data handles, creates view handles, exports view identity, exports
  numeric targets and supports release/destroy callbacks;
- `dagmldata_inmemory_provider_new_with_features_json` for the same provider
  plus JSON feature tables used by binding conformance tests;
- `dagmldata_inmemory_provider_feature_buffer_manifest_json` for deterministic
  JSON manifests of provider-owned numeric feature buffers;
- `dagmldata_inmemory_provider_feature_collation_json` for JSON row-major
  tensor collation from feature buffers owned by the in-memory provider;
- `dagmldata_inmemory_provider_feature_collation_tensor_f64_json` for ABI-owned
  row-major f64 tensor export from provider-owned feature buffers;
- `DagMlDataVTable` with materialize/view/identity/target/feature/release hooks.
  The `feature_arrow` hook accepts either a plain feature-set id or a JSON
  feature-fusion selector. The vtable uses the shared
  `DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION` macro and guarded
  `DagMlDataVTable` definition so `dag_ml_data.h` and `dag_ml.h` can be
  included together by bindings.

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
| Rust-owned f64 tensor descriptor and nested arrays | Rust allocation returned through ABI | `dagmldata_tensor_f64_free` |
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

`dagmldata_coordinator_feature_fusion_arrow_json` exercises the pure Rust
feature-fusion kernel through the C ABI. It accepts a request shaped as
`{ feature_set_id, sources, alignment, policy? }`, where each source contains a
`source_id` and a `CoordinatorFeatureBlock`. The exported Arrow table preserves
reference-source repeated observations, broadcasts singleton rows from secondary
sources, namespaces fused feature columns by default and refuses incoherent
presence masks or ambiguous repeated secondary sources.

`dagmldata_coordinator_feature_collation_json` exercises the pure Rust numeric
late-collation kernel through the C ABI. It accepts `{ feature_block, policy? }`
and returns a JSON `NumericTensorBlock` with observation/sample identity,
row-major shape and values, optional presence mask and optional value-validity
mask. It is a conformance helper, not a provider lifecycle.

`dagmldata_coordinator_feature_collation_tensor_f64_json` exports the same
result as an ABI-owned `DagMlDataTensorF64` instead of JSON. The tensor carries
`abi_version`, block/representation/container strings, observation ids, sample
ids, `shape`, contiguous row-major `values`, optional `presence_mask`, optional
`validity_mask` and optional `feature_names`. Masks are byte arrays with values
0 or 1. The caller must release the tensor with `dagmldata_tensor_f64_free`.

`dagmldata_inmemory_provider_feature_collation_json` and
`dagmldata_inmemory_provider_feature_collation_tensor_f64_json` exercise the
same late-collation kernel against provider-owned typed numeric buffers. They
accept `{ feature_set_id, policy? }` for a single provider feature table or
`{ fusion, policy? }` where `fusion` is the provider feature-fusion selector.
The JSON export is a conformance/debug path; the `DagMlDataTensorF64` export is
the binding-oriented path. These helpers are specific to vtables created by
`dagmldata_inmemory_provider_new_with_features_json`; they do not change the
stable `DagMlDataVTable` layout.

`dagmldata_inmemory_provider_feature_buffer_manifest_json` returns an array of
`NumericFeatureBufferManifest` values for the provider-owned typed buffers. Each
manifest includes the feature-set id, representation id, feature and observation
ids, row/feature/value counts, estimated f64 storage bytes and a deterministic
buffer fingerprint. Bindings can use this before creating feature views or
tensors to verify that the provider loaded the expected data buffers.

## In-Memory Provider VTable

The in-memory provider is the current ABI conformance target. It accepts one
validated coordinator envelope plus optional sample-level target tables and
observation-level feature tables, then implements:

- `materialize`: validates a coordinator materialization request and returns an
  opaque data handle whose coordinator relations are scoped to the requested
  `source_ids`;
- `make_view`: applies a `DataView` to a data handle and returns an opaque view
  handle;
- `view_identity`: returns the filtered relation table as Arrow C Data;
- `target_arrow`: returns sample-level numeric targets aligned to the view;
- `feature_arrow`: returns observation-level numeric features aligned to the
  view and filtered by `DataView.columns` when passed a plain feature-set id;
  when passed `{ feature_set_id, sources, alignment, policy? }` JSON, where
  each source is `{ source_id, feature_set_id, columns? }`, it fuses
  provider-owned source feature buffers through the core feature-fusion kernel;
- `release` and `destroy`: release handles and provider state.

The conformance provider still receives small JSON fixture feature tables at
construction time, but it converts them once into column-major
`NumericFeatureBuffer` values grouped by `NumericFeatureBufferStore` in the
provider state. `feature_arrow` exports are then view projections over those
owned buffers, not per-call JSON numeric parsing. Fusion selectors reuse those
typed buffers, filter each source by source identity in the view, and then call
the same pure Rust fusion kernel used by the standalone ABI helper. Provider
feature-collation selectors then collate either a single feature table or the
fused block into deterministic row-major JSON or `DagMlDataTensorF64` tensors
without reparsing feature values. Full provider implementations will use the
same vtable shape while keeping production data buffers host-owned.

`tests/c_header_smoke.rs` has two C checks: a header syntax smoke with
`cc -fsyntax-only`, and a linked C program that loads the Rust `cdylib`, creates
the provider vtable, materializes a view, exports identity, target and feature
Arrow arrays, then releases all handles.
The syntax smoke also includes the sibling `dag_ml.h` in both include orders
when a `dag-ml` checkout is available, so shared data-provider vtable guards
cannot drift silently.

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
5. Add ABI conformance for the core multi-source feature-fusion and numeric
   collation kernels.
6. Route in-memory provider `feature_arrow` fusion selectors through the same
   kernel.
7. Route in-memory provider feature-collation selectors through provider-owned
   typed buffers.
8. Expose provider-backed collation as `DagMlDataTensorF64` for bindings.
9. Expose provider-owned feature-buffer manifests for binding conformance.
10. Replace in-memory typed fixture buffers with production feature-buffer
   lifecycles.
