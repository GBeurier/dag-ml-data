# C ABI

The data ABI lets a host runtime keep buffers and fitted data adapters in its own
memory while exposing deterministic descriptors and identity tables to the core.

## Current Scaffold

`crates/dag-ml-data-capi/include/dag_ml_data.h` exposes:

- version and string-free helpers;
- `dagmldata_schema_fingerprint_json`;
- `DagMlDataVTable` with materialize/view/identity/target/release hooks.

## Ownership Rules

| Object | Owner | Release path |
|---|---|---|
| Materialized data handle | Host | `DagMlDataVTable.release` |
| View handle | Host | `DagMlDataVTable.release` |
| Fitted adapter handle | Host | future fitted-adapter release hook |
| Rust error/fingerprint string | Rust allocation returned through ABI | `dagmldata_string_free` |
| Arrow arrays | Producer of the Arrow array | Arrow C Data Interface release callback |

## ABI Roadmap

1. Freeze byte/string/status conventions.
2. Add C smoke test for schema fingerprinting.
3. Add explicit Arrow C Data forward declarations in the header.
4. Add path-solving and data-plan validation over canonical JSON.
5. Add conformance tests for Python and native host data providers.
