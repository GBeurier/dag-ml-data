# Capability Matrix

`dag-ml-data` supports the replacement of the current `nirs4all` data pipeline
surface by making data shape, identity and conversion explicit. It does not own
OOF or leakage decisions, but it must expose enough information for `dag-ml` to
enforce them.

## Data Surface

| Capability | Data contract support | Enforcement owner |
|---|---|---|
| Multisource | `DatasetSchema`, `SourceDescriptor`, alignment policy, presence masks, planner-visible `Align` steps | `dag-ml` decides phase and accepted fusion policy |
| Repetitions | `SampleRelation` with observation/sample/target/group/origin ids | `dag-ml` validates split unit and aggregation |
| Grouped samples | group ids exposed through sample relations | `dag-ml` validates group-aware folds |
| Augmentation | augmentation adapters declare output origin ids | `dag-ml` validates train-only use |
| Processings | representation adapters, fit scope, fitted adapter refs | `dag-ml` chooses fold/full-train scope |
| Splits | identity/group/origin inputs only | `dag-ml` builds folds |
| Models | `ModelInputSpec`, accepted representations/types, aux inputs | controller and `dag-ml` execute model phases |
| Refit | serialized `DataPlan`, schema fingerprints, fitted refs | `dag-ml` controls replay/refit phase |
| Branching | immutable views and source filters | `dag-ml` owns branch graph semantics |
| Merging | alignment, feature join, source join, collation contracts | `dag-ml` validates prediction joins and downstream use |
| Concatenation | namespace columns, presence indicators, output representation | `dag-ml` decides whether the merge is legal in phase |
| Finetuning | stateful/supervised adapter declarations | `dag-ml` enforces fold-train fit boundaries |
| Generation | serializable adapter params and plugin versions | `dag-ml` owns variant enumeration |
| Tuning | dry-run shapes and deterministic data plans | `dag-ml` owns tuner phase and nested CV |

## Contract Requirements

1. Every source has stable sample identity.
2. Every representation carries semantic axes.
3. Every conversion path is explicit, costed, versioned and deterministic.
4. Lossy/stateful/supervised adapters are opt-in at planning time.
5. Presence masks and alignment choices are serializable and planned before
   multi-source joins.
6. Schema fingerprints are stable under irrelevant ordering changes.
7. No fold, OOF, prediction partition or leakage decision is made in this repo.
