# dag-ml-data

Rust-first data contract and planning layer for typed, sample-aligned,
multi-source ML data.

`dag-ml-data` owns schemas, axes, representations, immutable data views,
sample relations, representation adapters, data plans, alignment/collation
contracts and schema fingerprints. It does not own ML phases, CV orchestration,
OOF joins or model execution; those belong to `dag-ml`.

> Status: foundation scaffold plus coordinator envelope, handle/materialized-view
> smoke, target alignment smoke, an in-memory C ABI provider vtable and a
> reusable Python `ctypes` smoke wrapper for identity/target/feature Arrow
> exports. The project has executable Rust crates, C ABI header, CLI
> fingerprinting/planning/materialization commands, design documents,
> rationale, roadmap, CI and contract tests.

## Repository Layout

```text
crates/
  dag-ml-data-core/   # schema, representation, view, relation and plan types
  dag-ml-data/        # Rust facade re-exporting stable core APIs
  dag-ml-data-capi/   # C ABI helpers and DataVTable contracts
  dag-ml-data-cli/    # local schema validation/fingerprint utility
docs/
  TOC.md              # validation-oriented table of contents
  ARCHITECTURE.md     # data-layer boundaries and flow
  ABI.md              # C ABI ownership model for host data providers
  RATIONALE.md        # why this is separate from dag-ml
  ROADMAP.md          # delivery phases and gates
  STATUS.md           # current state and next tasks
  TEST_PLAN.md        # contract/conformance test strategy
  design/source/      # moved ML_DATA source specification
examples/
  minimal_schema.json
  python/             # stdlib-only ctypes wrapper and provider smoke
```

## Quick Start

```bash
cargo fmt --all --check
cargo test --workspace
cargo run -p dag-ml-data-cli -- fingerprint-schema examples/minimal_schema.json
cargo run -p dag-ml-data-cli -- materialize-envelope --envelope examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json --request examples/fixtures/oof_campaign/materialization_request_model_base_x.json
cargo build -p dag-ml-data-capi --lib
python3 examples/python/provider_smoke.py --lib target/debug/libdag_ml_data_capi.so --envelope examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json --request examples/fixtures/oof_campaign/materialization_request_model_base_x.json
```

## First Implementation Target

The first useful milestone is a schema and planning core that can:

1. parse and validate canonical dataset schemas;
2. describe semantic axes and representations;
3. produce deterministic schema fingerprints;
4. represent unresolved data plans without executing ML;
5. expose fingerprinting and basic validation through the C ABI;
6. materialize validated coordinator envelopes into opaque handle records for
   DAG-ML controller tasks;
7. expose coordinator identity relations through a minimal Arrow C Data ABI
   smoke path;
8. create identity-filtered view handles and align sample-level target values
   across repeated observations;
9. exercise the provider vtable lifecycle for
   materialize/view/identity/target/feature operations;
10. provide a stdlib-only Python ABI smoke wrapper that external bindings can
    use as a starting conformance target.
