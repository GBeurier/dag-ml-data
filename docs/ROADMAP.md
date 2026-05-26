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
model-input planning, coordinator envelope export, explicit envelope schema
versioning and a published v1 JSON Schema artifact shared and compared with
`dag-ml` in CI. A shared conformance-pack manifest now pins the normalized
schema digests, canonical fixture digests, C ABI requirements and required
cross-repo checks.

## Phase 2: Host Providers

Definition of done:

- in-memory Rust provider for tests;
- Python provider skeleton and reusable `ctypes` smoke;
- C ABI conformance around view creation and identity/target/feature export;
- Arrow identity, target and feature table smoke tests.

Status: first in-memory coordinator handle arena implemented. It validates a
coordinator envelope plus materialization request, returns an opaque handle
record, and records run/node/phase/variant/fold/fingerprint traceability for
`dag-ml` controller tasks. It also creates identity-filtered view handles and
aligns sample-level target values while de-duplicating repeated observations.
The C ABI now exposes a Rust-owned in-memory provider vtable that materializes
data handles, creates view handles, exports view identity, exports numeric
targets, exports numeric observation-level features, and supports
release/destroy callbacks. A stdlib-only Python example now exercises the same
lifecycle through the public ABI. Native vtable conformance now checks
materialize/view/export paths plus release behavior, including parent-handle
release invalidating child views. Provider feature inputs are converted once at
provider construction into reusable column-major `NumericFeatureBuffer` values
from `dag-ml-data-core`, grouped in a `NumericFeatureBufferArena`, and exposed
through deterministic feature-buffer manifests before Arrow export. The
preferred numeric conformance inputs are typed row-major
`NumericFeatureMatrixF64` values and borrowed C `DagMlDataFeatureMatrixF64View`
descriptors with optional validity masks, avoiding per-cell JSON numeric value
transport. The arena binds compatible buffers to each materialized data handle
after source-scoped relations are known, and provider feature/fusion/collation
exports validate those bindings before exporting Arrow or tensors. Next:
replace the in-memory typed input path with production buffer sources using the
same manifest/binding/projection contract.

## Phase 3: Alignment, Fusion And Collation

Definition of done:

- `inner`, `left`, `outer` alignment plans;
- presence-mask propagation;
- feature joiner contracts;
- late collation contracts for dense tensor models.

Status: started. The core now has a deterministic sample-alignment planner for
`inner`, `left` and `outer` policies, and model-input planning emits an explicit
`Align` step before multi-source joins. The core also has an executable
observation-level feature-fusion kernel for aligned feature blocks. It preserves
reference-source repetitions, broadcasts singleton non-reference rows, fills
missing outer/left values with nulls, namespaces columns by default and refuses
ambiguous repeated non-reference joins. A C ABI conformance helper exports that
kernel as Arrow over already materialized feature blocks, and the in-memory
provider vtable can route `feature_arrow` JSON fusion selectors through the same
kernel using provider-owned typed feature buffers. A core numeric collation
kernel now produces row-major tensor blocks with deterministic padding,
truncation, presence masks and value-validity masks. The in-memory provider can
now route feature-collation JSON selectors, including fused selectors, through
provider-owned typed buffers and return deterministic `NumericTensorBlock` JSON
or ABI-owned `DagMlDataTensorF64` buffers. Production provider arenas and
non-f64 tensor-buffer variants are still pending.

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
