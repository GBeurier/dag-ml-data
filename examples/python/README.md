# Python ABI Example

This directory contains a standard-library-only `ctypes` smoke wrapper for the
current `dag-ml-data` C ABI.

Files:

- `dag_ml_data_provider.py`: minimal wrapper around `DagMlDataVTable`,
  `ArrowArray` and `ArrowSchema`;
- `provider_smoke.py`: executable conformance smoke over the in-memory provider,
  including provider-wide feature-buffer manifests, data-handle-scoped
  feature-buffer bindings, observation-level features and sample-level targets.

Run from the repository root:

```bash
cargo build -p dag-ml-data-capi --lib
python3 examples/python/provider_smoke.py --lib target/debug/libdag_ml_data_capi.so --envelope examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json --request examples/fixtures/oof_campaign/materialization_request_model_base_x.json
```

The wrapper is intentionally not the final Python package API. It is a small
binding reference for materialization, view creation, identity export,
feature-buffer manifest export, data-handle binding export, observation-level
feature export, sample-level target export, handle release and provider
destruction.
