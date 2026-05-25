# dag-ml-data

Rust-first data contract and planning layer for typed, sample-aligned,
multi-source ML data.

`dag-ml-data` owns schemas, axes, representations, immutable data views,
sample relations, representation adapters, data plans, alignment/collation
contracts and schema fingerprints. It does not own ML phases, CV orchestration,
OOF joins or model execution; those belong to `dag-ml`.

> Status: foundation scaffold. The project is ready for implementation work:
> executable Rust crates, C ABI header, CLI fingerprinting, design documents,
> rationale, roadmap, CI and first contract tests are present.

## Repository Layout

```text
crates/
  dag-ml-data-core/   # schema, representation, view, relation and plan types
  dag-ml-data/        # Rust facade re-exporting stable core APIs
  dag-ml-data-capi/   # C ABI helpers and DataVTable skeleton
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
```

## Quick Start

```bash
cargo fmt --all --check
cargo test --workspace
cargo run -p dag-ml-data-cli -- fingerprint-schema examples/minimal_schema.json
```

## First Implementation Target

The first useful milestone is a schema and planning core that can:

1. parse and validate canonical dataset schemas;
2. describe semantic axes and representations;
3. produce deterministic schema fingerprints;
4. represent unresolved data plans without executing ML;
5. expose fingerprinting and basic validation through the C ABI.
