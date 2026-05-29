# dag-ml-data Python bindings

Thin PyO3/maturin bindings for `dag-ml-data` JSON contracts.

This package validates schemas, plans, relations and coordinator envelopes. It
also exposes the deterministic planner and fingerprint functions. It does not
own host data buffers, execute adapters or use the C ABI provider vtable.

## Build

```bash
cargo test -p dag-ml-data-py
maturin build --release --features extension-module
python3 ../../scripts/smoke_python_bindings.py  # after installing the wheel
```

## Python Surface

```python
import dag_ml_data

schema = dag_ml_data.DatasetSchema(schema_json)
model_input = dag_ml_data.ModelInputSpec(model_input_json)
registry = dag_ml_data.AdapterRegistry(adapter_registry_json)
relations = dag_ml_data.SampleRelationTable(sample_relations_json)

fingerprint = schema.fingerprint()
plan = dag_ml_data.plan_model_input(
    schema,
    model_input,
    registry,
    {"id": "nir-to-tabular", "source_ids": ["nir"]},
)
envelope = dag_ml_data.build_coordinator_data_plan_envelope(
    schema,
    plan,
    relations,
)
envelope_json = envelope.json()
```

All Rust-side validation failures are raised as `dag_ml_data.DagMlDataError`.
Native errors expose `category`, `code`, `severity`, `remediation_hint`,
`context`, `context_json` and `descriptor_json` attributes for
ADR-11-compatible handling.

`validate_fold_set_json` checks only the exhaustive partition shape of a
`FoldSet`: every sample must appear in validation exactly once. Use
`validate_fold_set_against_sample_relations_json` when the caller also needs
group and augmentation-origin leakage checks.
