# DMD contributor instructions

## Ownership and navigation

- DMD owns generic schemas/descriptors, axes/representations/views, sample relations/identity, adapters/path solving, alignment/collation, deterministic fingerprints and host-provider lifecycle/data plans.
- [DAG](../dag-ml/AGENTS.md) owns ML phases/fold scheduling/OOF/selection/execution/leakage policy; [Methods](../nirs4all-methods/AGENTS.md) owns splitter algorithms/numerical methods; [IO](../nirs4all-io/AGENTS.md) materializes datasets.
- Sibling links are optional navigation.
- Do not choose/generate/schedule folds or add NIRS-specific assumptions. Validate supplied fold contracts only against data-owned sample/group/origin identities. Declare fit scope here; runtime enforcement remains DAG-owned.
- Host data may remain behind opaque handles.
- Schemas/descriptors stay serializable, deterministic and source-order-independent; stable schema/plan/relation fingerprints are validated with identity and request compatibility at materialization.
- Start with [DEVELOPMENT](docs/DEVELOPMENT.md), [RATIONALE](docs/RATIONALE.md), [ARCHITECTURE](docs/ARCHITECTURE.md), [ABI](docs/ABI.md), [contracts](docs/contracts/README.md), [SUPPORTED](docs/SUPPORTED.md) and [operations](docs/AGENT_OPERATIONS.md).

## ABI, data and performance invariants

- Use typed ID wrappers and Result/DataError; never accept raw identifier strings on public APIs. Keep facade/core re-exports one surface and workspace dependency declarations centralized.
- Host owns materialized/view/fitted-adapter handles released by vtable.release; Rust-owned strings/tensors/Arrow arrays/schemas use owner free helpers.
- Borrowed numeric inputs are copied during constructors, so callers may release them afterwards.
- Prefer typed row-major numeric matrices or borrowed feature views over JSON values on hot numeric input. Reuse NumericFeatureBufferArena manifest/bind/project/release lifecycle rather than another C-ABI arena implementation.
- Extend the existing provider vtable for production; name conformance-only helpers `dagmldata_inmemory_provider_*`. Shared provider/tensor version macros and headers must match DAG and be includable in either order.
- Keep shared JSON schemas/fixtures/conformance pack JSON-identical.
- Wire changes need version updates and explicit unsupported-version refusal, both-owner drift checks, public SUPPORTED/CHANGELOG updates.
- Private historical notes are optional context, never required build inputs or force-added files.

## Gate and provider checks

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p dag-ml-data-cli -- fingerprint-schema examples/minimal_schema.json
python3 scripts/validate_contracts.py
```

- Retain targeted core, C-header and embedded-Python ctypes/provider smokes, envelope/materialization CLI commands and their C/Python prerequisites in operations. Cross-header checks include both orders with the sibling present.
- Rust 2021/owner toolchain constraints, Result/thiserror conventions and existing crate patterns apply. Discover supported current capabilities from contracts/SUPPORTED and executed evidence, not old phase/version/status assertions.

- When a sibling DAG checkout exists, also run `DAG_ML_REPO=../dag-ml python3 scripts/validate_contracts.py` for cross-repository contract drift. Do not point that variable at a missing checkout; the plain command is the standalone local check.

## Working rules

- Read applicable local and deeper instructions before editing. Preserve dirty/concurrent work; make authorized owner fixes and propagate affected contracts/bindings/consumers. Do not reset another author's work.
- Use CodeGraph FIRST for implementation navigation only if this checkout's root already has `.codegraph/`; use its exact projectPath or run the CLI here. Never auto-index or silently substitute an ancestor index.
- Unindexed code and documentation use ordinary source navigation. Explicit source, tests and build tools are final authority. Owner links are navigation; this policy works without a parent workspace.
- All application/project content is English: UI text/subtitles/tooltips/errors, docs/comments/docstrings, test/fixture prose, reports/issues/PRs, commits and generated material. Preserve identifiers, quotations and historical/provenance meaning; converse with the maintainer in French.
- Do not commit credentials, private data, generated workspaces, caches or build outputs. Follow local licensing/fixture restrictions and report actual commands/exits/skips; a command reference grants no execution/publication permission.
