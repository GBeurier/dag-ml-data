# Test Plan

## Unit Tests

| Area | First tests |
|---|---|
| Identifiers | invalid chars, empty ids, max length |
| Representations | missing sample axis, rank/axis mismatch, ragged rules |
| Schema | duplicate sample ids, duplicate sources, source validation |
| Fingerprint | source-order independence, deterministic JSON hash |
| Plans | unresolved choices, empty plans, declared output representation |
| ABI | null pointer handling, invalid JSON, valid fingerprint |

## Conformance Tests

Add after providers exist:

- Python and Rust providers return identical identity Arrow tables;
- path solver returns same plan independent of adapter registration order;
- source alignment is stable for `inner`, `left` and `outer`;
- schema fingerprint rejects incompatible predict-time schemas.

## Shared Fixtures With `dag-ml`

The first shared fixture should be a minimal UC6 stacking dataset: two base
prediction sources, a meta-model input plan and a shuffled sample order that
forces identity-based alignment.
