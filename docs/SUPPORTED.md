# Supported Surface

This page is the 0.2.0 support contract for `dag-ml-data`. It separates
production-facing data contracts from conformance providers and backlog work.
It does not change any public ABI, JSON schema, Rust, Python or WASM signature.

## Support Levels

| Level | Meaning |
|---|---|
| Supported | Included in the release scope and covered by CI gates. |
| Conformance | Stable enough for binding authors and integration tests, but not a complete production backend. |
| Experimental | Public shape may exist, but release notes must call out limitations. |
| Backlog | Not part of the release promise. |

## dag-ml-data Surface

| Area | Level | Notes |
|---|---|---|
| Dataset schema, ids, axes and representation contracts | Supported | Rust validation, JSON-compatible structs and deterministic fingerprints are gated. |
| Model-input planning and adapter path solving | Supported | Fixture planning and deterministic path selection are tested. |
| Coordinator data-plan envelope v1 | Supported | Shared schema/fixture drift is checked against `dag-ml`. |
| Sample relations and FoldSet boundary validation | Supported | Group/origin/repetition leakage checks are exposed through Rust, C ABI, Python and WASM helpers. |
| Numeric feature buffers and `.n4d` persistence | Supported | Manifest, fingerprint, bind/project/release and file tamper checks are tested. |
| N-D tensor transport | Supported | Borrowed tensor views are copied to provider-owned buffers and exported through view-filtered handles. |
| Fitted adapter refs/manifests/store | Supported | Portable URI/fingerprint validation and in-memory store ABI are covered. |
| In-memory provider vtable | Conformance | Full materialize/view/identity/target/feature lifecycle is tested; it is not a production storage backend. |
| Python `ctypes` provider package | Conformance | Useful binding template and smoke target; not a domain-specific provider backend. |
| Arrow IPC feature-buffer reader | Conformance | Optional crate/feature path for IPC-backed numeric buffers. |
| `branch_view` mode `by_source` | Supported | Executed natively by the in-memory provider. |
| `branch_view` modes `by_metadata`, `by_tag`, `by_filter` | Backlog | Validated and explicitly refused by the in-memory provider with `requires host-side filtering`. |
| Production provider arenas beyond in-memory | Backlog | Host providers must implement native storage/filtering semantics. |
| nirs4all `SpectroDataset` connector | Backlog | Descoped to `nirs4all-io`; `dag-ml-data` remains NIRS-agnostic. |

## Cross-Repo Contract With dag-ml

The 0.2.0 release must keep these artifacts JSON-identical with `dag-ml`:

- `coordinator_data_plan_envelope.schema.json`;
- `feature_fusion_selector.schema.json`;
- `coordinator_branch_view.schema.json`;
- `fitted_adapter_ref.schema.json`;
- `conformance_pack.v1.json`;
- `parity_oracle.v1.json`;
- shared `FoldSet` fixture.

The public vtable ABI version shared with `dag-ml` must remain guarded so
`dag_ml_data.h` and `dag_ml.h` compile in both include orders.

## Public-Signature Policy

For the 0.2.0 release window:

- no C ABI symbol, struct layout, JSON schema id/version, Rust public function,
  Python facade function, R package native surface or WASM export changes
  without an explicit contract entry and ABI/schema snapshot update;
- if such a change is accepted, downstream chains such as `nirs4all-lite`,
  `nirs4all-web` and browser/Python smoke packages must be rebuilt before tag;
- documentation, CI jobs, tests and private benchmark helpers are allowed when
  they do not alter exported signatures.

## Post-0.2.0 Backlog

1. Keep the documented mapping between `dag-ml` coordinator aggregation
   policies and data-side reducer vocabulary current (`robust_mean` /
   `exclude_outliers` remain data-side only unless a shared schema is added).
2. Wire signal-type expectations through paired `dag-ml` replay contracts before
   claiming coordinator-enforced signal-type predict checks.
3. Add production host provider backends for host-filtered branch views.
4. Keep the existing AddressSanitizer C ABI lane green through release.
5. Extend the current fingerprint performance probes to provider export paths.
