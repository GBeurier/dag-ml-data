# dag-ml-data

`dag-ml-data` is the Rust-first data-contract and planning layer for typed,
sample-aligned, multi-source ML data. It owns schemas, semantic axes,
representations, immutable data views, sample relations, representation
adapters, data plans, alignment/collation contracts, schema fingerprints and
the host-provider C ABI. ML phases, OOF joins and model execution belong to the
companion `dag-ml` repo.

This site is the contributor and integration entry point for data contracts used
by future nirs4all and nirs4all-lite integrations. The nirs4all connector is
owned by `nirs4all-io`; this repo stays NIRS-agnostic.

## Start Here

| Need | Page |
|---|---|
| Build and validate locally | [Installation](installation.md) |
| Understand data/runtime boundaries | [Architecture](ARCHITECTURE.md) |
| Integrate a data provider over C ABI | [C ABI](ABI.md) |
| Check shipped vs pending scope | [Status](STATUS.md) |
| Run the documented gates | [Test plan](TEST_PLAN.md) |
| Review roadmap and release gates | [Roadmap](ROADMAP.md) |
| Map nirs4all replacement data capabilities | [Capability matrix](CAPABILITY_MATRIX.md) |
| Inspect shared contracts | [Contract manifests](contracts/README.md) |
| Review shared decisions | [Architecture decisions](adr/README.md) |
| Pick an example by audience | `examples/README.md` |

## API References

- Rust core API: <https://docs.rs/dag-ml-data-core/latest/>
- Rust facade API: <https://docs.rs/dag-ml-data/latest/>
- C ABI source: `crates/dag-ml-data-capi/include/dag_ml_data.h`
- Python binding source: `crates/dag-ml-data-py`
- WASM binding source: `crates/dag-ml-data-wasm`

```{toctree}
:maxdepth: 2
:hidden:

installation
ARCHITECTURE
ABI
STATUS
TEST_PLAN
ROADMAP
CAPABILITY_MATRIX
contracts/README
adr/README
```
