# dag-ml-data Table Of Contents

Use this as a validation map before development starts.

| Area | File | Purpose | Validate |
|---|---|---|---|
| Entry point | `README.md` | Project scope, layout and quick start | The repo can be understood in under five minutes |
| Agent handoff | `AGENTS.md` | Rules for autonomous implementation work | A new agent knows boundaries and green gate |
| Architecture | `docs/ARCHITECTURE.md` | Data-layer modules and DAG-ML frontier | No execution graph responsibility leaks in |
| ABI | `docs/ABI.md` | Data provider ABI and ownership | Host buffers stay host-owned |
| Rationale | `docs/RATIONALE.md` | Split from DAG-ML and design tradeoffs | Data scope is independently defensible |
| Roadmap | `docs/ROADMAP.md` | Sequenced delivery phases | Every phase has an observable definition of done |
| Status | `docs/STATUS.md` | Current scaffold state and next actions | No hidden implementation claims |
| Test plan | `docs/TEST_PLAN.md` | Schema, planner and ABI tests | Fingerprints and path solving are covered |
| Source design | `docs/design/source/ml_data_specification_v1.md` | Full ML_DATA contract | Used as implementation source of truth |
| Core crate | `crates/dag-ml-data-core` | Schema, representation and data-plan primitives | `cargo test -p dag-ml-data-core` passes |
| C ABI crate | `crates/dag-ml-data-capi` | FFI-safe helpers and `DataVTable` shape | Header mirrors Rust ABI structs |
| CLI crate | `crates/dag-ml-data-cli` | Local validation and fingerprinting | Example schema fingerprints |

## Validation Checklist

| Check | Command |
|---|---|
| Rust formatting | `cargo fmt --all --check` |
| Rust tests | `cargo test --workspace` |
| Lints | `cargo clippy --workspace --all-targets -- -D warnings` |
| Example schema | `cargo run -p dag-ml-data-cli -- fingerprint-schema examples/minimal_schema.json` |
