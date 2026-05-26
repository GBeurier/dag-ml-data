# Architecture

`dag-ml-data` is the data contract and planning layer. It describes what data is
available and how it can be converted, but it does not decide ML phases or
enforce OOF invariants.

## Layers

| Layer | Owned here | Not owned here |
|---|---|---|
| Schema | dataset id, sample ids, sources, targets, metadata | graph nodes and scheduler |
| Representation | axes, units, containers, dtype, ragged/sparse flags | model execution |
| Views | immutable selectors over samples/sources/columns | fold construction |
| Planning | materialize/adapt/align/join/collate data plans | selecting variants or models |
| Fingerprints | deterministic schema hashes for replay | artifact lineage graph |
| ABI | host data-provider vtable and schema helpers | controller/model ABI |

## Crates

| Crate | Responsibility |
|---|---|
| `dag-ml-data-core` | Pure Rust serializable contracts and validation. |
| `dag-ml-data` | Stable Rust facade for downstream crates and bindings. |
| `dag-ml-data-capi` | C ABI helper functions and data provider vtable contracts. |
| `dag-ml-data-cli` | Local schema validation and fingerprint utility. |

## Data Planning Flow

```text
DatasetSchema + ModelInputSpec + policy
  -> resolve representation path
  -> DataPlan(materialize -> adapt* -> align -> join -> collate)
  -> execute_fit / execute_transform in a host data provider
  -> host-owned data handle returned to dag-ml
```

The current scaffold reaches validated handles/views, source-scoped
materialization relations, typed numeric feature-buffer stores with manifests,
data-handle-scoped feature-buffer bindings, and Arrow C Data smokes for
identity, sample-level targets and observation-level features. Runtime adapters
and production host buffer arenas are still outside the scaffold.

## Boundary With `dag-ml`

`dag-ml-data` may expose:

- source descriptors and semantic axes;
- identity, group and origin relations;
- data plans and fitted adapter references;
- schema fingerprints;
- host handles for materialized views.

It must not expose:

- fold sets as a planning primitive;
- OOF prediction blocks;
- graph lineage records;
- model selection decisions.
