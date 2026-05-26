# Roadmap

## Phase 0: Contracts Frozen

Definition of done:

- Rust types for ids, axes, representations, sources, schemas, views and plans;
- deterministic schema fingerprint;
- C ABI helper for fingerprinting;
- source design docs moved into `docs/design/source`;
- first CLI and tests pass.

## Phase 1: Path Solver

Definition of done:

- adapter registry;
- BFS/Dijkstra representation path search;
- refusal and `requires_user_choice` reporting;
- fixture tests for tabular, dense signal, image and time-series paths.

Status: implemented for the current fixtures, including deterministic
model-input planning and coordinator envelope export.

## Phase 2: Host Providers

Definition of done:

- in-memory Rust provider for tests;
- Python provider skeleton;
- C ABI conformance around view creation and identity export;
- Arrow identity table smoke tests.

Status: first in-memory coordinator handle arena implemented. It validates a
coordinator envelope plus materialization request, returns an opaque handle
record, and records run/node/phase/variant/fold/fingerprint traceability for
`dag-ml` controller tasks. It also creates identity-filtered view handles and
aligns sample-level target values while de-duplicating repeated observations. A
minimal Arrow C Data identity-table export exists for coordinator envelopes.
Next: attach real buffer lifecycles and export view identity/target arrays
through the provider vtable.

## Phase 3: Alignment And Fusion

Definition of done:

- `inner`, `left`, `outer` alignment plans;
- presence-mask propagation;
- feature joiner contracts;
- late collation contracts for dense tensor models.

## Phase 4: nirs4all Connector

Definition of done:

- `SpectroDataset` connector exposing `SourceDescriptor`;
- dense signal representations for current NIRS layouts;
- compatibility tests against current `nirs4all` pipeline behavior.

## Phase 5: Bundle Replay

Definition of done:

- serialized `DataPlan` and fitted adapter references;
- schema fingerprint compatibility checks;
- replay fixtures shared with `dag-ml`.
