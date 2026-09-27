# dag-ml-data-provider (Python)

Stdlib-only `ctypes` shim for the `dag-ml-data` C ABI provider vtable. It is a
thin, modality-neutral wrapper over the Rust `dag-ml-data-capi` cdylib: it hands
JSON payloads and host buffers to Rust and decodes the Arrow / owned-struct
results. It contains no NIRS, ML, or scheduling logic — those live in
`dag-ml-data` / `dag-ml`.

This is the **provider** binding. The sibling `dag-ml-data` PyO3 package target
(`crates/dag-ml-data-py`) binds the JSON *contracts* and is unrelated.

## Locating the cdylib

The platform wheel bundles `dag_ml_data_capi` under `.libs/` and records its
size and SHA-256 in a package-local provenance manifest. Automatic discovery
verifies that manifest before loading the bundled library. Existing source
checkout and explicit-library workflows remain available. Discovery order:

1. an explicit `library_path=...` argument,
2. the `DAG_ML_DATA_CAPI_LIB` environment variable,
3. the verified package-local `dag_ml_data_provider/.libs/` library,
4. the Cargo target directory of a source checkout (honoring `CARGO_TARGET_DIR`,
   else `<workspace>/target/{debug,release}`).

Build the cdylib from a checkout with:

```bash
cargo build -p dag-ml-data-capi --lib
```

Build an autonomous wheel from a clean checkout:

```bash
python -m build --wheel crates/dag-ml-data-capi/bindings/python
```

The build hook refuses dirty sources
unless `DAG_ML_DATA_PROVIDER_ALLOW_DIRTY_BUILD=1` is explicitly set for a local
development artifact. `scripts/smoke_python_provider_wheel.py` verifies an
installed wheel in an isolated environment.

## Usage

```python
from dag_ml_data_provider import InMemoryProvider

with InMemoryProvider.from_files(
    "coordinator_data_plan_envelope.json",
    f64_feature_matrices=[...],
    # library_path=... optional; discovered automatically otherwise
) as provider:
    data_handle = provider.materialize_file("materialization_request.json")
    view_handle = provider.make_view(data_handle, {"sample_ids": [...]})
    features = provider.feature_values(view_handle, "x")
```

`PackageProvider.from_package_resources(...)` takes explicit relative resource
names for a coordinator envelope, a deterministic `.n4d` feature store and
optional target tables in an installed package. It refuses traversal, escaping
symlinks and namespace packages; Rust validates the `.n4d` integrity trailer.
Direct `PackageProvider(...)` and `from_files(...)` accept host filesystem paths.
All three constructors retain the optional `library_path=` override. This is a
numeric package-backed provider slice; N-D package payloads and direct
`nirs4all-io DatasetPackage` conversion remain future work.

## Tests

```bash
pip install -e ".[test]"
cargo build -p dag-ml-data-capi --lib   # so the cdylib can be discovered
pytest
```

The package tests skip automatically if the cdylib cannot be located.
