# Test Plan

## Unit Tests

| Area | First tests |
|---|---|
| Identifiers | invalid chars, empty ids, max length |
| Representations | missing sample axis, rank/axis mismatch, ragged rules |
| Schema | duplicate sample ids, duplicate sources, source validation |
| Fingerprint | source-order independence, schema hash, data-plan hash |
| Plans | unresolved choices, empty plans, declared output representation |
| Planner | fixture schema/model-input/adapters produce expected data plan |
| Relations | duplicate observations, group consistency, augmentation origin validity |
| Coordinator envelope | explicit schema version, published envelope JSON Schema version, unsupported schema version refusal, schema/plan/relation fingerprint validation |
| Handles | materialization request/envelope fingerprint match, opaque data/view handle traceability |
| Views/features/targets | sample/source/augmentation filtering, requested sample-order preservation, repetition-preserving identity, observation-level feature alignment, feature-column filtering, feature representation mismatch refusal, sample-level target de-duplication |
| ABI | null pointer handling, invalid JSON, valid fingerprint, coordinator identity plus numeric target/feature Arrow exports, in-memory provider vtable lifecycle, parent/child handle release, C header syntax, linked C runtime, embedded Python ctypes smoke and reusable Python example smoke |

## Conformance Tests

Add after providers exist:

- handle arena refuses schema/plan/relation mismatch and missing required relations;
- provider views return identical identity, feature and target rows independent
  of handle order;
- Python and Rust providers return identical provider-vtable identity, feature
  and target Arrow tables;
- path solver returns same plan independent of adapter registration order;
- source alignment is stable for `inner`, `left` and `outer`;
- schema fingerprint rejects incompatible predict-time schemas.

## Shared Fixtures With `dag-ml`

The first shared fixture should be a minimal UC6 stacking dataset: two base
prediction sources, a meta-model input plan and a shuffled sample order that
forces identity-based alignment.

Current CLI smoke commands:

```bash
cargo run -p dag-ml-data-cli -- validate-envelope examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json
cargo run -p dag-ml-data-cli -- materialize-envelope --envelope examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json --request examples/fixtures/oof_campaign/materialization_request_model_base_x.json
python3 -m json.tool docs/contracts/coordinator_data_plan_envelope.schema.json >/dev/null
DAG_ML_REPO=../dag-ml python3 scripts/validate_contracts.py
```
