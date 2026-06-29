# ML_DATA Specification v1

Status: design v1, ready for implementation.
Companion library: DAG-ML (in-process ML engine). ML_DATA describes possibilities;
DAG-ML decides plans, phases and invariants.

The document is auto-sufficient: every type, method, policy and algorithm referenced
in one section is defined in the document (typically in section 2, 4, 5, 6, 7 or 12).
Pseudocode appears whenever an algorithm has multiple branches.

---

## 1. Mission, perimeter, non-perimeter

### 1.1 Mission

ML_DATA is the generic, extensible data layer that feeds DAG-ML and any other ML
engine that consumes typed, sample-aligned, multi-source data. It generalises the
multi-source / multi-processing logic that `nirs4all` currently bakes into
`SpectroDataset` and exposes it as a small set of typed protocols, registries,
adapters and serialisable specs.

ML_DATA owns:

- the **storage** and the **logical model** of sources (descriptors, axes, units, granularity).
- the **schema** of a dataset (sample ids, sources, targets, metadata).
- **immutable views** of the data (`DataView`).
- **typed payloads** (`DataBlock`, `FeatureTable`, `TargetBlock`).
- **sample alignment** across sources (`inner` / `left` / `outer` / explicit masks).
- the **type registry** (`DataTypePlugin` + `DataTypeRegistry`) so domains can declare
  custom types: dense signals, images, genotype matrices, time series, graphs, tables,
  text, hyperspectral cubes, mass-spec spectra, etc.
- the **adapter registry** (`RepresentationAdapter` + `AdapterRegistry`) and a
  **path solver** (BFS) so a source representation can be converted into a target
  representation in a deterministic, declarative way.
- the **fusion** and **collation** primitives (`FeatureJoiner`, `BatchCollator`).
- a **`DataPlanner`** that, given a `ModelInputSpec`, produces a `DataPlan` (a series
  of `materialize` / `adapt` / `align` / `join` / `collate` steps).
- **serialisation**: every spec is JSON-serialisable; fitted artifacts reference an
  external store via `SerializableRef`; a stable `schema_fingerprint` is produced
  for replay.
- **`SampleRelation`**: the central pivot to model repetitions, augmentations,
  group constraints and aggregation, *without* implementing a splitter.

### 1.2 Non-perimeter

ML_DATA strictly **does not** own:

- the execution graph, the topological scheduler, the node registry.
- ML phases (`FIT_CV`, `SELECT`, `REFIT`, `PREDICT`, `EXPLAIN`). ML_DATA accepts a
  `phase` field on `AdapterContext` but never produces one and never gates behaviour
  on it.
- the OOF / no-leakage invariant. ML_DATA exposes the building blocks
  (`SampleRelation.origin_id`, `AdapterSpec.fit_scope`, `AlignmentPolicy`) so DAG-ML
  can enforce the invariant.
- fold construction, cross-validation, stacking orchestration.
- the cache of execution results, the lineage graph, the artifact store.
- variant-level parallelism, generator enumeration, hyperparameter search.
- refusing a plan. `DataPlanner.resolve` can flag `requires_user_choice` or raise
  on unsolvable cases; it never enforces an ML invariant.

### 1.3 Frontier diagram

```text
+---------------------------- DAG-ML --------------------------------+
| graph, nodes, ports, edges                                         |
| phases (FIT_CV / SELECT / REFIT / PREDICT / EXPLAIN)               |
| fold orchestration, splitters, OOF invariant                       |
| selection / refit                                                  |
| artifact store, prediction store, lineage, cache                   |
| operator/model adapters, schedulers, parallelism                   |
|                                                                    |
|         calls into ML_DATA only through the shared contract:       |
|         SourceDescriptor, RepresentationSpec, AxisSpec,            |
|         DataView, DataBlock, FeatureTable, PresenceMask,           |
|         ModelInputSpec, FusionPolicy, AlignmentPolicy,             |
|         AuxInputSpec, DataPlan, FittedAdapter                      |
+--------------------------------------------------------------------+
                              ^   |
   shared contract types ---->|   | calls   <----- shared contract
                              |   v
+---------------------------- ML_DATA -------------------------------+
| stores sources (file-backed / in-memory)                           |
| DatasetSchema, SourceDescriptor, AxisSpec, RepresentationSpec      |
| DataView, DataBlock, FeatureTable, PresenceMask, SampleRelation    |
| DataTypeRegistry, AdapterRegistry, FeatureJoiner, BatchCollator    |
| DataPlanner.resolve / execute_fit / execute_transform              |
| schema_fingerprint, JSON canonical serialisation                   |
+--------------------------------------------------------------------+
```

What crosses the frontier:

- **inputs to ML_DATA**: a `DataView` selector, an optional `RepresentationSpec`
  target, an `AdapterContext` carrying phase / fold_id / random_state.
- **outputs from ML_DATA**: `DataBlock`s, `FeatureTable`s, `DataPlan`s,
  `AlignmentPlan`s, `FittedAdapter` handles.

What does **not** cross the frontier:

- DAG-ML never passes `FoldSet`, `PredictionBlock`, `LineageRecord` to ML_DATA.
- ML_DATA never sees the graph, the nodes, the scheduler.
- The execution-side cache lives in DAG-ML. ML_DATA may have its own *in-process*
  block cache (LRU on materialized DataBlock) but this is an implementation
  detail, not a contract.

---

## 2. Data model

This is the canonical type system. Every other section consumes it.

### 2.1 Identifier aliases

```python
from typing import NewType

SampleId       = NewType("SampleId", str)        # canonical: str. int is coerced via str(int).
SourceId       = NewType("SourceId", str)        # lowercase, ascii, [-a-z0-9_.]+
RepresentationId = NewType("RepresentationId", str)  # e.g. "tabular_numeric", "rgb_image"
TypeId         = NewType("TypeId", str)          # plugin id, e.g. "dense_signal"
ObservationId  = NewType("ObservationId", str)   # row id in a source
TargetId       = NewType("TargetId", str)        # id of the y unit
GroupId        = NewType("GroupId", str)         # leakage-prevention unit
```

Rules:

- **Canonical form**: all ids are strings. Integers are accepted at the boundary
  (`from_int=lambda i: str(i)`) but ML_DATA always stores them as strings to keep
  JSON serialisation deterministic.
- **Uniqueness**: `SampleId` is unique within a `DatasetSchema`. `SourceId` is unique
  within a `DatasetSchema`. `RepresentationId` is unique within
  `DataTypeRegistry.list_representations()`. `TypeId` is unique within
  `DataTypeRegistry`.
- **Charset**: `[A-Za-z0-9_.-]+`, length <= 128. Validated on registration.
- **Serialisation**: JSON strings. Ordering is lexicographic.

### 2.2 Axes

```python
from dataclasses import dataclass
from typing import Literal, Any

AxisKind = Literal[
    "sample", "feature", "processing",
    "time", "height", "width", "channel",
    "node", "edge", "variant", "token", "target",
    "wavelength", "wavenumber", "frequency", "depth",
]

@dataclass(frozen=True)
class AxisSpec:
    name: str                      # local name, e.g. "wavelength", "time", "h"
    kind: AxisKind                 # semantic role
    unit: str | None = None        # "nm", "cm-1", "s", "px", None
    size: int | None = None        # None when variable=True
    variable: bool = False         # True for ragged / dynamic axes
    coordinate: CoordinateSpec | None = None  # typed axis coordinates

CoordinateDType = Literal["numeric", "categorical", "datetime"]

@dataclass(frozen=True)
class CoordinateSpec:
    dtype: CoordinateDType
    ordered: bool = False
    # values is a tagged union ("kind"):
    #   {"kind": "explicit", "values": [...]}  one JSON value per axis index
    #   {"kind": "regular_grid", "start": float, "step": float}  value(i)=start+i*step
    values: CoordinateValues
```

Rules:

- `kind="sample"` axis is mandatory in every `RepresentationSpec.axes`.
- `size` and `variable` are mutually exclusive: if `variable=True`, `size` is `None`.
- `unit` must be non-empty when present.
- A `coordinate` is rejected on a `variable` axis. `AxisSpec` denies unknown fields,
  so the removed untyped `coordinates` field is a hard error, not silently dropped.
- `explicit` coordinate length must match `size` when both are provided; an explicit
  list must be non-empty. `numeric` values are finite numbers; `categorical` values are
  unique non-empty strings; `datetime` values are canonical RFC 3339 UTC second-precision
  strings (`YYYY-MM-DDThh:mm:ssZ`, calendar-validated).
- `ordered` numeric / datetime coordinates must be strictly monotonic (ascending or
  descending). For `categorical`, `ordered` is the declared label order (labels are not
  compared).
- `regular_grid` requires `dtype="numeric"`, a known `size`, finite `start`/`step`,
  `step != 0`, and `ordered=true` (the grid sign sets the direction).
- Two `AxisSpec`s are equal iff all fields are equal.

### 2.3 Representation

```python
@dataclass(frozen=True)
class RepresentationSpec:
    id: RepresentationId
    type_id: TypeId
    rank: int | None               # None for ragged / object containers
    axes: tuple[AxisSpec, ...]     # length == rank when rank is set
    container: str                 # see allowed values below
    dtype: str | None = None       # numpy dtype string, e.g. "float32"
    sparse: bool = False
    ragged: bool = False
```

Allowed `container` values (extensible by plugins, but core supports):
`ndarray`, `array`, `dataframe`, `feature_block_set`, `ragged_array`, `list`,
`sparse_csr`, `sparse_csc`, `graph_batch`, `list_of_arrays`, `dict_of_arrays`,
`pil_image_batch`, `torch_tensor`, `object_array`.

Rules:

- `rank is None` implies `ragged=True`.
- If `rank` is set, `len(axes) == rank`.
- The first axis must be `kind="sample"` for sample-major containers, except when
  `container == "graph_batch"` where the sample dimension is implicit in the batch
  pointer (the plugin documents this).
- `RepresentationSpec`s are content-hashed: two specs are equal iff every field is
  equal (axes are compared element-wise).

### 2.4 Sources

```python
SourceGranularity = Literal[
    "per_sample",          # one record per sample
    "per_sample_repeated", # multiple records per sample (NIRS reps, plates)
    "per_sample_sequence", # one variable-length sequence per sample (weather)
    "per_sample_set",      # one variable-size set per sample (multiple images)
    "per_group",           # one record per group of samples
    "per_target",          # one record per target id
]

@dataclass(frozen=True)
class SourceDescriptor:
    id: SourceId
    name: str
    type_id: TypeId
    modality: str                            # free string: "spectroscopy", "image",
                                             # "genotype", "weather", "metadata", ...
    native_representation: RepresentationSpec
    sample_key: str                          # name of the column / field that holds the
                                             # SampleId (e.g. "sample_id")
    granularity: SourceGranularity
    schema: dict[str, Any] = field(default_factory=dict)
                                             # JSON-able structural schema:
                                             # tabular: {"columns": [{"name", "dtype"}]}
                                             # image:   {"size": [H, W, C], "dtype": "uint8"}
                                             # graph:   {"node_attrs": [...], "edge_attrs": [...]}
    tags: dict[str, Any] = field(default_factory=dict)
```

Rules:

- `type_id` must be registered in the `DataTypeRegistry` used to validate the
  dataset.
- `native_representation.type_id == type_id`.
- `sample_key` is the column / attribute that ML_DATA uses to align this source on
  the canonical sample axis. For `per_sample`, it is unique within the source. For
  `per_sample_repeated`, it is *not* unique and ML_DATA exposes a `SampleRelation`
  to expand sample <-> observation mapping (see 2.10).

### 2.5 DatasetSchema

```python
@dataclass(frozen=True)
class DatasetSchema:
    dataset_id: str
    sample_ids: tuple[SampleId, ...]
    sources: tuple[SourceDescriptor, ...]
    targets: dict[str, RepresentationSpec]    # by name, e.g. {"y": tabular_numeric}
    metadata: dict[str, RepresentationSpec]   # ditto, but for non-target metadata
```

Rules:

- `sample_ids` is the canonical sample axis of the dataset. Every source aligns to
  this axis via `sample_key`.
- All `SourceId`s in `sources` are unique. Same for `targets` keys and `metadata`
  keys.
- A target name and a metadata name may not collide.
- A `DatasetSchema` is JSON-serialisable (see section 12).

### 2.6 DataView

```python
@dataclass(frozen=True)
class DataView:
    sample_ids: tuple[SampleId, ...] | None = None
    partition: str | None = None            # "train" / "val" / "test" / custom tag
    fold_id: str | int | None = None        # opaque label, set by DAG-ML
    source_ids: tuple[SourceId, ...] | None = None
    columns: tuple[str, ...] | None = None  # tabular / metadata column filter
    include_augmented: bool = True
    include_excluded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
```

Rules:

- A `DataView` is **purely declarative**: it describes a selection, not a result.
- `MLDataset.view(view)` is **idempotent and deterministic**: calling it twice
  with identical input produces an equal `DataView`. Implementations may canonicalise
  fields (e.g. sort `source_ids`, dedupe `sample_ids`) but must do so deterministically.
- `extra` carries opaque key/value selectors that domain plugins can interpret
  (e.g. a NIRS plugin may use `extra={"signal_type": "absorbance"}`). Keys must be
  strings; values must be JSON-serialisable.

### 2.7 PresenceMask

```python
@dataclass(frozen=True)
class PresenceMask:
    sample_ids: tuple[SampleId, ...]
    source_id: SourceId
    present: tuple[bool, ...]                # len == len(sample_ids)
```

Rules:

- `len(present) == len(sample_ids)`.
- `present[i]` is `True` iff this source has data for `sample_ids[i]`.
- The mask is sample-major and aligned with `sample_ids` order, not the source's
  internal ordering.

### 2.8 DataBlock

```python
@dataclass(frozen=True)
class DataBlock:
    source_id: SourceId
    representation: RepresentationSpec
    sample_ids: tuple[SampleId, ...]
    data: Any                                 # actual payload, type follows
                                              # representation.container
    axes: tuple[AxisSpec, ...]                # mirror of representation.axes,
                                              # but may carry instance-level
                                              # coordinates / sizes
    presence: PresenceMask | None = None
    feature_names: tuple[str, ...] | None = None
    lineage: tuple[str, ...] | None = None    # opaque chain of adapter ids, append-only
```

Rules:

- `len(sample_ids)` equals the size of the sample axis of `data`. For `graph_batch`
  containers, `len(sample_ids)` equals the number of graphs in the batch.
- `representation` must validate (`DataTypePlugin.validate(self)`) under the type
  plugin identified by `representation.type_id`.
- A `DataBlock` is **immutable**: once returned by `MLDataset.materialize`, its
  fields must not be mutated. Implementations may freeze numpy arrays
  (`arr.flags.writeable = False`).
- `lineage` is an append-only list of adapter ids. Adapters append their `spec.id`
  before returning a new block; ML_DATA never mutates an existing block.
- `feature_names`, when non-None, has length equal to the size of the feature axis
  (when there is exactly one feature axis). Otherwise it is None.

### 2.9 FeatureTable

```python
@dataclass(frozen=True)
class FeatureTable:
    sample_ids: tuple[SampleId, ...]
    X: Any                                    # ndarray, scipy sparse, or pandas-like
    columns: tuple[str, ...]                  # len == X.shape[1]
    source_ids: tuple[SourceId, ...]          # provenance per column, len == X.shape[1]
    presence: dict[SourceId, PresenceMask] = field(default_factory=dict)
    lineage: tuple[str, ...] | None = None
```

Rules:

- `len(columns) == X.shape[1]`.
- `len(source_ids) == X.shape[1]`. Each entry tells which `SourceId` produced the
  column.
- `FeatureTable` is the canonical intermediate representation for early-fusion
  (concatenation of sources into a single 2D matrix). A `FeatureTable` is always
  convertible to a `DataBlock` with `representation.id == "tabular_numeric"`.

### 2.10 TargetBlock

```python
@dataclass(frozen=True)
class TargetBlock:
    name: str
    sample_ids: tuple[SampleId, ...]
    representation: RepresentationSpec
    y: Any                                    # ndarray (regression), int array (classif)
                                              # or dataframe column
    y_transform_lineage: tuple[str, ...] = ()  # adapter ids that transformed y
    classes: tuple[Any, ...] | None = None    # for classification
    presence: PresenceMask | None = None
```

Rules:

- `TargetBlock.name` matches a key of `DatasetSchema.targets`.
- A `TargetBlock` carries its **transform lineage** so DAG-ML can invert the
  transform at predict time without re-fitting (the inverse adapter is registered
  in the `AdapterRegistry`).
- `classes` is only set for classification targets and is preserved across folds.

### 2.11 SampleRelation

`SampleRelation` is the pivot that lets ML_DATA model repetitions, augmentation,
group constraints and aggregation without itself touching the splitter.

```python
@dataclass(frozen=True)
class SampleRelation:
    source_id: SourceId
    observation_ids: tuple[ObservationId, ...]
    sample_ids: tuple[SampleId, ...]                  # parallel to observation_ids
    target_ids: tuple[TargetId, ...] | None = None    # parallel; one Y per row
    group_ids: tuple[GroupId, ...] | None = None      # parallel; leakage unit
    origin_ids: tuple[SampleId | None, ...] | None = None
                                                       # parallel; non-None iff the row
                                                       # is augmented from another sample
```

Semantics:

- **`observation_id`**: physical row in a source. Always unique per source.
- **`sample_id`**: the logical sample. One sample can map to N observations in a
  single source (NIRS repetitions: 3 scans per leaf) or across sources.
- **`target_id`**: identifier of the y unit. A `target_id` may carry multiple
  `sample_id`s (one leaf -> multiple samples -> one chemistry measurement).
- **`group_id`**: the unit that DAG-ML must keep within the same fold to avoid
  leakage (plant, plot, patient, batch).
- **`origin_id`**: when an observation is augmented from another sample,
  `origin_id` is the source `sample_id`. Otherwise it is `None`.

Rules:

- Tuples are parallel. `len(observation_ids) == len(sample_ids)`. The other tuples
  are either `None` or have the same length.
- `origin_id is None` for original rows. `origin_id == sample_id` is forbidden
  (would denote self-augmentation; use `None` instead).
- ML_DATA exposes `MLDataset.sample_relation(source_id, view) -> SampleRelation`
  and DAG-ML uses it to:
  - choose `split_unit` (sample / target / group).
  - prevent augmented copies of a validation sample from leaking into training.
  - aggregate predictions back to the requested level.

ML_DATA does **not** do any split. It only describes the relation.

---

## 3. Dataset interface

```python
from typing import Protocol, Sequence

class MLDataset(Protocol):
    def schema(self) -> DatasetSchema: ...

    def view(self, selector: DataView) -> DataView: ...

    def materialize(
        self,
        source_id: SourceId,
        view: DataView,
        representation: RepresentationId | None = None,
    ) -> DataBlock: ...

    def target(
        self,
        name: str,
        view: DataView,
        representation: RepresentationId | None = None,
    ) -> TargetBlock: ...

    def metadata(
        self,
        view: DataView,
        columns: Sequence[str] | None = None,
    ) -> DataBlock: ...

    def presence(
        self,
        source_id: SourceId,
        view: DataView,
    ) -> PresenceMask: ...

    def sample_relation(
        self,
        source_id: SourceId,
        view: DataView,
    ) -> SampleRelation: ...
```

Invariants:

1. **Determinism of the selector**: `dataset.view(v1) == dataset.view(v1)`
   for any `v1`. The canonicalisation rules (sort, dedupe) are deterministic.
2. **Immutability of blocks**: every returned `DataBlock`, `TargetBlock`,
   `FeatureTable`, `PresenceMask`, `SampleRelation` is immutable in the Python
   sense (frozen dataclass, frozen numpy arrays, immutable tuples).
3. **Sample id identity across sources**: if two sources are queried with the
   same view and both align (`per_sample` or sample_id-deduplicated view),
   `block_a.sample_ids == block_b.sample_ids`. The ordering is the canonical
   ordering imposed by the dataset (typically lexicographic over `SampleId`).
4. **Materialisation cost**: `materialize` may be expensive (file read,
   decompression). `view()`, `schema()`, `presence()` and `sample_relation()`
   are O(n_samples) at most.
5. **Representation negotiation**: if `representation` is `None`, the dataset
   returns the source's `native_representation`. If `representation` is set and
   ML_DATA knows a 0-hop conversion, it applies it; otherwise it raises
   `RepresentationError`. Multi-hop conversion goes through `DataPlanner`.

---

## 4. Custom type plugins

### 4.1 TypeCapability

```python
@dataclass(frozen=True)
class TypeCapability:
    type_id: TypeId
    native_representations: tuple[RepresentationId, ...]
    default_batching: Literal["dense", "ragged", "graph"] = "dense"
    supports_missing: bool = True
    supports_sample_alignment: bool = True
```

### 4.2 DataTypePlugin

```python
class DataTypePlugin(Protocol):
    @property
    def type_id(self) -> TypeId: ...

    @property
    def version(self) -> str: ...                 # semver, e.g. "1.0.0"

    def infer_source(
        self,
        obj: Any,
        *,
        source_id: SourceId,
        sample_key: str,
    ) -> SourceDescriptor: ...

    def validate(self, block: DataBlock) -> None:
        """Raise if the block payload does not conform to its representation."""
        ...

    def capability(self) -> TypeCapability: ...

    def default_collator(self) -> "BatchCollator | None": ...

    def known_representations(self) -> tuple[RepresentationSpec, ...]: ...
```

### 4.3 DataTypeRegistry

```python
class DataTypeRegistry(Protocol):
    def register_type(self, plugin: DataTypePlugin) -> None: ...
    def get_type(self, type_id: TypeId) -> DataTypePlugin: ...
    def list_types(self) -> tuple[TypeId, ...]: ...
    def list_representations(
        self, type_id: TypeId | None = None
    ) -> tuple[RepresentationId, ...]: ...
```

### 4.4 Core plugins shipped with ML_DATA

| `type_id`            | Native representations                              | Typical axes                                      | supports_missing | Example container       |
|----------------------|-----------------------------------------------------|---------------------------------------------------|------------------|-------------------------|
| `dense_signal`       | `signal_1d`, `signal_with_processings`              | `(sample, processing, wavelength)`                | yes (mask)       | `ndarray` float32       |
| `image_rgb`          | `rgb_image`                                         | `(sample, h, w, channel=3)`                       | yes              | `ndarray` uint8         |
| `gray_image`         | `gray_image`                                        | `(sample, h, w)`                                  | yes              | `ndarray` uint8         |
| `multichannel_image` | `mc_image`                                          | `(sample, h, w, channel)`                         | yes              | `ndarray` float32       |
| `genotype_matrix`    | `variant_matrix`, `dosage_matrix`                   | `(sample, variant)`                               | yes              | `ndarray` int8/float32  |
| `time_series`        | `series_uv`, `series_mv`                            | `(sample, time, variable)`                        | yes              | `ndarray` float32 / ragged list |
| `graph`              | `graph_batch`                                       | `(node*, edge*)` plus implicit sample pointer     | yes              | torch_geometric / pyg-like dict |
| `table`              | `tabular_numeric`, `tabular_mixed`                  | `(sample, column)`                                | yes (NaN)        | `dataframe` / `ndarray` |
| `text`               | `text_raw`, `text_token_ids`                        | `(sample, token)` for tokens; ragged              | yes              | list[str] or ragged int |

Plugin contracts:

- `dense_signal.validate(block)` requires `data.shape` matches the axes,
  `dtype` matches `representation.dtype`, the first axis equals
  `len(block.sample_ids)`, and the wavelength axis size matches
  `axes[-1].size`.
- `image_rgb` enforces `axes[-1].size == 3` and `axes[-1].kind == "channel"`.
- `genotype_matrix` allows `int8` (allele counts 0/1/2 plus -1 for missing) or
  `float32` (dosages in [0, 2]).
- `time_series` may be ragged (`ragged=True`); `BatchCollator` decides padding /
  truncation policy.
- `graph` must expose a `graph_batch` container: a dict containing
  `node_features`, `edge_index`, `edge_features`, `batch_ptr`, `num_nodes`,
  `num_edges`. Sample id mapping is via the `batch_ptr` segments.
- `table` accepts `dataframe` containers (pandas-like) or `ndarray` for the
  numeric subset.
- `text` is ragged. `text_token_ids` requires a vocab id reference in
  `representation.dtype = "int32"`; the vocabulary is exposed as side data
  (`AuxInputSpec(kind="side_data")`), not an axis coordinate, since the token axis
  is variable and variable axes cannot carry a `coordinate`.

---

## 5. Representation adapters

Adapters are the composable bridges between representations. Their **only** job
is to transform a `DataBlock` from one representation to another. They are not
ML steps; they are data shape / encoding conversions.

### 5.1 AdaptationPolicy

```python
@dataclass(frozen=True)
class AdaptationPolicy:
    allow_lossy: bool = False
    allow_stateful: bool = True
    require_fit_on_train_only: bool = True
    max_output_features: int | None = None
    preferred_adapters: tuple[str, ...] = ()
    forbidden_adapters: tuple[str, ...] = ()
```

### 5.2 AdapterSpec

```python
@dataclass(frozen=True)
class AdapterSpec:
    id: str                                       # globally unique adapter id
    version: str                                  # semver
    input_type: TypeId
    input_representation: RepresentationId | None # None means "any rep of input_type"
    output_representation: RepresentationId
    output_type: TypeId
    supervised: bool = False                      # consumes y during fit
    stateful: bool = False                        # produces a FittedAdapter
    lossy: bool = False                           # cannot be inverted losslessly
    fit_scope: Literal["none", "train_only", "fold_train"] = "none"
    cost_hint: dict[str, Any] = field(default_factory=dict)
                                                  # e.g. {"output_features": 1024,
                                                  # "flops_per_sample": 1.2e6,
                                                  # "wall_seconds_per_1k": 4.0}
```

### 5.3 AdapterContext

```python
@dataclass(frozen=True)
class AdapterContext:
    phase: str                                    # opaque label set by DAG-ML
    view: DataView
    fold_id: str | int | None
    random_state: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
```

Note: `phase` and `fold_id` come from DAG-ML. ML_DATA never invents them. The
seed propagation contract is described in section 10.4.

### 5.4 FittedAdapter

```python
@dataclass(frozen=True)
class FittedAdapter:
    spec: AdapterSpec
    artifact: Any | None                          # the fitted state. Opaque.
    output_schema: RepresentationSpec
    feature_names: tuple[str, ...] | None = None
```

### 5.5 RepresentationAdapter

```python
class RepresentationAdapter(Protocol):
    @property
    def spec(self) -> AdapterSpec: ...

    def can_adapt(
        self,
        source: SourceDescriptor,
        target: RepresentationSpec,
        policy: AdaptationPolicy,
    ) -> bool: ...

    def fit(
        self,
        block: DataBlock,
        y: TargetBlock | None,
        context: AdapterContext,
    ) -> FittedAdapter: ...

    def transform(
        self,
        block: DataBlock,
        fitted: FittedAdapter | None,
        context: AdapterContext,
    ) -> DataBlock: ...

    def fit_transform(
        self,
        block: DataBlock,
        y: TargetBlock | None,
        context: AdapterContext,
    ) -> tuple[DataBlock, FittedAdapter | None]: ...
```

Stateful-fit rule (enforced by adapter implementations and asserted by
`AdapterRegistry` on registration):

```text
if spec.stateful is True:
    transform(block, fitted=None, context) MUST raise StatefulAdapterMisuse
```

Stateless adapters (`stateful=False`) ignore the `fitted` argument and may pass
`fitted=None` through `transform()`. They also have `fit_scope == "none"`.

### 5.6 AdapterRegistry

```python
class AdapterRegistry(Protocol):
    def register_adapter(self, adapter: RepresentationAdapter) -> None: ...
    def adapters_from(
        self,
        source: SourceDescriptor,
        target: RepresentationSpec,
        policy: AdaptationPolicy,
    ) -> tuple[RepresentationAdapter, ...]: ...
    def find_path(
        self,
        source: SourceDescriptor,
        target: RepresentationSpec,
        policy: AdaptationPolicy,
    ) -> tuple[RepresentationAdapter, ...] | None: ...
```

### 5.7 `find_path` algorithm (BFS, weighted by cost / lossiness)

The path search is a typed BFS on a directed multi-graph where:

- nodes are `(TypeId, RepresentationId)` pairs.
- edges are adapters whose `(input_type, input_representation) -> (output_type,
  output_representation)` connects two nodes.
- edge weight is `cost_hint.get("wall_seconds_per_1k", 1.0) +
  (1000 if spec.lossy and not policy.allow_lossy else 0) +
  (100 if spec.stateful and not policy.allow_stateful else 0)`.

```text
def find_path(source, target, policy):
    start  = (source.type_id, source.native_representation.id)
    goal   = (target.type_id, target.id)
    if start == goal:
        return ()                                # 0-hop
    # forbidden adapters dropped upfront
    edges = [
        e for e in all_adapters
        if e.spec.id not in policy.forbidden_adapters
        and (policy.allow_lossy or not e.spec.lossy)
        and (policy.allow_stateful or not e.spec.stateful)
    ]
    # Dijkstra with the weight described above
    dist[start] = 0
    prev = {}
    heap = [(0, start)]
    while heap:
        d, u = heappop(heap)
        if u == goal:
            return reconstruct(prev, start, goal)
        for e in edges_from[u]:
            v = (e.spec.output_type, e.spec.output_representation)
            # honour preferred_adapters: subtract a discount
            bonus = -5 if e.spec.id in policy.preferred_adapters else 0
            nd = d + weight(e) + bonus
            if nd < dist.get(v, inf):
                dist[v] = nd
                prev[v] = (u, e)
                heappush(heap, (nd, v))
    return None
```

Notes:

- The discount on `preferred_adapters` is bounded so it cannot make the cost
  negative: in practice the implementation clamps `weight + bonus >= 0`.
- `policy.max_output_features` is enforced at plan time
  (`DataPlanner.resolve`) by inspecting `cost_hint["output_features"]` along
  the chosen path. If exceeded, the planner emits a warning or falls back to
  a different path.
- The output is a *tuple of adapters in order*, length 0 means already
  compatible.

### 5.8 Core adapters shipped with ML_DATA

| Adapter id                        | input_type        | input_representation       | output_type   | output_representation | supervised | stateful | lossy | fit_scope    | cost_hint                                                  |
|-----------------------------------|-------------------|----------------------------|---------------|-----------------------|------------|----------|-------|--------------|------------------------------------------------------------|
| `spectra.flatten`                 | `dense_signal`    | `signal_with_processings`  | `table`       | `tabular_numeric`     | no         | no       | no    | `none`       | `{"output_features": "<n_proc>*<n_wl>"}`                   |
| `spectra.resample`                | `dense_signal`    | `signal_1d`                | `dense_signal`| `signal_1d`           | no         | no       | yes   | `none`       | resamples to a target axis                                 |
| `image.embedding`                 | `image_rgb`       | `rgb_image`                | `table`       | `tabular_numeric`     | no         | yes      | yes   | `train_only` | `{"output_features": 512, "model": "resnet18"}`            |
| `image.raw_tensor`                | `image_rgb`       | `rgb_image`                | `image_rgb`   | `tensor_image_chw`    | no         | no       | no    | `none`       | `{"reshape_only": true}`                                   |
| `genotype.dosage`                 | `genotype_matrix` | `variant_matrix`           | `table`       | `tabular_numeric`     | no         | no       | no    | `none`       | `{"output_features": "<n_variant>"}`                       |
| `genotype.pca`                    | `genotype_matrix` | `dosage_matrix`            | `table`       | `tabular_numeric`     | no         | yes      | yes   | `train_only` | `{"output_features": 16}`                                  |
| `weather.aggregate`               | `time_series`     | `series_mv`                | `table`       | `tabular_numeric`     | no         | no       | yes   | `none`       | mean/std/min/max/quantiles per variable                    |
| `weather.sequence`                | `time_series`     | `series_mv`                | `time_series` | `sequence_tensor`     | no         | no       | no    | `none`       | pads / truncates                                           |
| `tabular.encoder`                 | `table`           | `tabular_mixed`            | `table`       | `tabular_numeric`     | no         | yes      | no    | `train_only` | one-hot / ordinal / target-encoding (separate spec ids)    |
| `text.embedding`                  | `text`            | `text_raw`                 | `table`       | `tabular_numeric`     | no         | yes      | yes   | `train_only` | `{"output_features": 384, "model": "miniLM"}`              |

Implementation note: each adapter must implement `can_adapt()` to refuse a
target representation that is incompatible (e.g. `spectra.flatten` refuses if
`policy.max_output_features < n_proc * n_wl`).

---

## 6. Alignment and fusion

### 6.1 AlignmentPolicy

```python
@dataclass(frozen=True)
class AlignmentPolicy:
    join: Literal["inner", "left", "outer", "exact"] = "inner"
    reference_source: SourceId | None = None
    on_missing_sample: Literal["error", "drop", "impute", "mask"] = "error"
```

Semantics:

- `inner`: keep only samples present in every source.
- `left`: keep samples from `reference_source` (if `None`, the first source in the
  request). Samples missing in other sources fall back to `on_missing_sample`.
- `outer`: keep the union of sample ids across sources.
- `exact`: every source must contain exactly the same set of sample ids; raise on
  any mismatch.
- `on_missing_sample`:
  - `error`: raise.
  - `drop`: remove that sample id from the alignment.
  - `impute`: ML_DATA *requests* an imputer adapter from the registry; if none is
    available, fall through to `error`.
  - `mask`: keep the sample id but build a `PresenceMask` (the model is expected
    to consume the mask).

### 6.2 FusionPolicy

```python
@dataclass(frozen=True)
class FusionPolicy:
    mode: Literal["concat_features", "stack_channels", "dict_input", "list_input"]
    target_representation: RepresentationId
    alignment: AlignmentPolicy
    missing_source: Literal["error", "drop", "impute", "indicator", "mask"] = "error"
    namespace_columns: bool = True
    allow_lossy_adapters: bool = False
    max_output_features: int | None = None
```

`missing_source` is independent of `alignment.on_missing_sample`: it controls what
happens when an *entire* source is absent or empty in the requested view.
`indicator` means "add a binary column that flags presence", which is the
recommended default for tree-based models that cannot consume masks natively.

### 6.3 AlignmentPlan

```python
@dataclass(frozen=True)
class AlignmentPlan:
    sample_ids: tuple[SampleId, ...]
    per_source_positions: dict[SourceId, tuple[int | None, ...]]
                              # for each source, gives, for each canonical sample,
                              # the row position in that source (None if missing).
    presence: dict[SourceId, PresenceMask]
```

### 6.4 Alignment algorithm

Given a `DataView`, the sources `S = (s_1, ..., s_k)` and an `AlignmentPolicy p`:

```text
def align(view, sources, policy):
    ids_per_src = {s.id: dataset.materialize(s.id, view).sample_ids for s in sources}
    if policy.join == "exact":
        ref_set = set(ids_per_src[sources[0].id])
        for s in sources[1:]:
            if set(ids_per_src[s.id]) != ref_set:
                raise AlignmentError(...)
        canonical = sorted(ref_set)
    elif policy.join == "inner":
        canonical = sorted(set.intersection(*(set(v) for v in ids_per_src.values())))
    elif policy.join == "left":
        ref_id = policy.reference_source or sources[0].id
        canonical = list(ids_per_src[ref_id])  # preserve ref order
    elif policy.join == "outer":
        canonical = sorted(set.union(*(set(v) for v in ids_per_src.values())))
    # build per_source_positions and presence
    positions = {}
    presence  = {}
    for s in sources:
        idx_map = {sid: i for i, sid in enumerate(ids_per_src[s.id])}
        positions[s.id] = tuple(idx_map.get(c) for c in canonical)
        presence[s.id]  = PresenceMask(
            sample_ids=tuple(canonical),
            source_id=s.id,
            present=tuple(c in idx_map for c in canonical),
        )
    # apply on_missing_sample
    missing = [c for c, ok in zip(canonical, ...) if not all(presence[s.id].present)]
    if missing:
        if policy.on_missing_sample == "error":
            raise AlignmentError(...)
        elif policy.on_missing_sample == "drop":
            keep = [c for c in canonical if all(presence[s.id].present[i]
                                                for s in sources
                                                for i, x in enumerate(canonical) if x == c)]
            canonical = keep
            recompute positions / presence
        elif policy.on_missing_sample == "impute":
            adapter = registry.imputer_for(missing_sample_kind)
            # adapter is invoked downstream by the DataPlan, not here
        elif policy.on_missing_sample == "mask":
            pass  # leave missing rows, expose presence
    return AlignmentPlan(canonical, positions, presence)
```

Notes:

- The canonical sample order is deterministic: sorted ascending for `inner`,
  `outer`, `exact`; preserved from `reference_source` for `left`.
- The plan does not materialise data; it only computes the alignment.

### 6.5 FeatureJoiner

```python
class FeatureJoiner(Protocol):
    @property
    def spec(self) -> AdapterSpec: ...

    def fit(
        self,
        tables: Sequence[FeatureTable],
        policy: FusionPolicy,
        context: AdapterContext,
    ) -> FittedAdapter: ...

    def transform(
        self,
        tables: Sequence[FeatureTable],
        fitted: FittedAdapter,
        context: AdapterContext,
    ) -> FeatureTable: ...
```

The `FeatureJoiner` is a standard adapter that produces a stable schema at fit
time:

```text
def fit(tables, policy, context):
    plan = align(context.view, tables_as_sources, policy.alignment)
    columns = []
    src_ids = []
    for t in tables:
        for col in t.columns:
            namespaced = f"{t.source_ids[0]}.{col}" if policy.namespace_columns else col
            columns.append(namespaced)
            src_ids.append(t.source_ids[0])
    if policy.max_output_features and len(columns) > policy.max_output_features:
        raise FusionError(...)
    output_schema = RepresentationSpec(
        id=policy.target_representation,
        type_id="table",
        rank=2,
        axes=(AxisSpec("sample", "sample"), AxisSpec("feature", "feature",
                                                     size=len(columns))),
        container="ndarray",
        dtype="float32",
    )
    return FittedAdapter(
        spec=self.spec,
        artifact={"plan": plan, "columns": tuple(columns),
                  "source_ids": tuple(src_ids)},
        output_schema=output_schema,
        feature_names=tuple(columns),
    )

def transform(tables, fitted, context):
    plan = fitted.artifact["plan"]
    columns = fitted.artifact["columns"]
    src_ids = fitted.artifact["source_ids"]
    X_parts = []
    for t in tables:
        pos = plan.per_source_positions[t.source_ids[0]]
        if any(p is None for p in pos):
            X_part = reindex_with_missing(t.X, pos, policy.missing_source)
        else:
            X_part = t.X[pos, :]
        X_parts.append(X_part)
    X = horizontal_concat(X_parts)
    return FeatureTable(
        sample_ids=plan.sample_ids,
        X=X,
        columns=columns,
        source_ids=src_ids,
        presence=plan.presence,
    )
```

Key property: **schema is fixed at fit time**. At predict time the joiner
re-applies the exact same column order, same namespacing and same missing-value
strategy. This is what makes the train/predict round-trip safe.

---

## 7. Data resolution for a model (`DataPlanner`)

### 7.1 InputPortSpec / ModelInputSpec

```python
@dataclass(frozen=True)
class InputPortSpec:
    name: str
    accepted_representations: tuple[RepresentationId, ...]
    accepted_types: tuple[TypeId, ...]
    rank: int | None = None
    multi_source: bool = False
    optional: bool = False

@dataclass(frozen=True)
class ModelInputSpec:
    ports: tuple[InputPortSpec, ...]
    default_fusion: FusionPolicy | None = None
```

Note: these two dataclasses live in the shared contract (section 13). ML_DATA
does not invent the model side; it only consumes the spec.

### 7.2 DataPlanStep / DataPlan

```python
@dataclass(frozen=True)
class DataPlanStep:
    kind: Literal["materialize", "adapt", "align", "join", "collate"]
    inputs: tuple[str, ...]          # symbolic ids of upstream step outputs
    output: str                      # symbolic id produced by this step
    adapter_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DataPlan:
    steps: tuple[DataPlanStep, ...]
    output_ports: dict[str, str]                # port name -> step output id
    warnings: tuple[str, ...] = ()
    requires_user_choice: tuple[str, ...] = ()
```

A `DataPlan` is a small, linear (or near-linear) DAG. `inputs` references either
`SourceId`-prefixed step output ids (`"src:<source_id>"` for raw materialisation)
or previously declared step outputs (`"step:<n>"`).

### 7.3 DataPlanner

```python
class DataPlanner(Protocol):
    def resolve_from_schema(
        self,
        schema: DatasetSchema,
        sources: Sequence[SourceId],
        model_input: ModelInputSpec,
        policy: FusionPolicy,
    ) -> DataPlan:
        """
        Plan the data path using only the schema, without materialising any
        data. Used by DAG-ML at PLAN phase (before any fold or fit).

        Implementations MUST raise `DatasetRequiredForPlanning` if any
        candidate adapter declares `supervised=True` or
        `fit_scope in {"train_only", "fold_train"}` and the schema alone is
        insufficient to choose deterministically. In that case the caller
        (DAG-ML) escalates to `resolve(dataset, ...)` at FIT_CV.
        """
        ...

    def resolve(
        self,
        dataset: MLDataset,
        sources: Sequence[SourceId],
        model_input: ModelInputSpec,
        policy: FusionPolicy,
    ) -> DataPlan:
        """
        Plan the data path with a materialised dataset available. Used by
        DAG-ML at FIT_CV when supervised / fold-aware adapters need to peek
        at the data to pick a path (e.g. an adapter that gates on signal
        sparsity, a feature selector that depends on label distribution).
        """
        ...

    def execute_fit(
        self,
        plan: DataPlan,
        dataset: MLDataset,
        view: DataView,
        y: TargetBlock | None,
        context: AdapterContext,
    ) -> tuple[dict[str, DataBlock], tuple[FittedAdapter, ...]]: ...

    def execute_transform(
        self,
        plan: DataPlan,
        dataset: MLDataset,
        view: DataView,
        fitted: Sequence[FittedAdapter],
        context: AdapterContext,
    ) -> dict[str, DataBlock]: ...
```

How is a `DataPlanner` instance obtained? The application layer (e.g. nirs4all)
constructs a concrete implementation and DAG-ML injects it into the
`RunContext` (DAG-ML §14.1). ML_DATA itself ships a default
`DefaultDataPlanner` that uses the `AdapterRegistry` to discover paths.

ML_DATA contract: any planner implementation must satisfy
`resolve_from_schema(schema, ...) == resolve(dataset, ...).restricted_to_schema()`
when no adapter peeks at the data. This guarantees that PLAN-time decisions
are consistent with FIT_CV-time decisions for the common case.

### 7.4 Resolution algorithm

```text
def resolve(dataset, sources, model_input, policy):
    schema   = dataset.schema()
    plan     = []
    chosen   = {}                  # source -> list of adapters chosen
    warnings = []
    user_ch  = []

    # Phase 1: per-source path discovery
    for src_id in sources:
        src = next(s for s in schema.sources if s.id == src_id)
        # candidate target representations from the ports
        candidates = []
        for port in model_input.ports:
            for rep_id in port.accepted_representations:
                rep_spec = registry.repr_spec(rep_id)
                if rep_spec.type_id in port.accepted_types or rep_spec.id in port.accepted_representations:
                    candidates.append((port, rep_spec))
        # for each candidate, attempt to find a path
        scored = []
        for port, target in candidates:
            path = adapter_registry.find_path(src, target, AdaptationPolicy(
                allow_lossy=policy.allow_lossy_adapters,
                allow_stateful=True,
                require_fit_on_train_only=True,
                max_output_features=policy.max_output_features,
            ))
            if path is None:
                continue
            cost = sum(adapter_cost(a) for a in path) + lossy_penalty(path, policy)
            scored.append((cost, port, target, path))
        if not scored:
            if any(p.optional for p in model_input.ports):
                warnings.append(f"source {src_id} has no path; skipped because optional")
                continue
            raise NoPlanFoundError(src_id, [c[1].id for c in candidates])
        # ambiguity: multiple cheapest paths within epsilon
        scored.sort(key=lambda t: t[0])
        if len(scored) > 1 and (scored[1][0] - scored[0][0]) < EPSILON:
            user_ch.append(f"source {src_id} has multiple equally cheap adapters")
        chosen[src_id] = scored[0]

    # Phase 2: build steps
    sid = 0
    def new_id():
        nonlocal sid
        sid += 1
        return f"step:{sid}"

    src_outputs = {}
    for src_id, (cost, port, target, path) in chosen.items():
        # materialize
        mat = DataPlanStep("materialize", inputs=(), output=f"src:{src_id}",
                           params={"source_id": src_id})
        plan.append(mat)
        cur = f"src:{src_id}"
        for adapter in path:
            o = new_id()
            plan.append(DataPlanStep("adapt", inputs=(cur,), output=o,
                                     adapter_id=adapter.spec.id))
            cur = o
        src_outputs.setdefault(port.name, []).append(cur)

    # Phase 3: per-port alignment and fusion
    output_ports = {}
    for port in model_input.ports:
        outs = src_outputs.get(port.name, [])
        if not outs:
            if port.optional:
                continue
            raise NoPlanFoundError(port.name)
        if len(outs) > 1 and not port.multi_source:
            raise PortArityError(port.name, len(outs))
        if len(outs) == 1 and policy.mode in ("dict_input", "list_input"):
            output_ports[port.name] = outs[0]
            continue
        align_step = DataPlanStep("align", inputs=tuple(outs),
                                  output=new_id(),
                                  params={"alignment": policy.alignment})
        plan.append(align_step)
        join_step = DataPlanStep("join", inputs=(align_step.output,),
                                 output=new_id(),
                                 adapter_id="fusion.feature_joiner",
                                 params={"policy": policy})
        plan.append(join_step)
        output_ports[port.name] = join_step.output

    # Phase 4: optional collation (depends on type plugin defaults)
    for port_name, step_id in list(output_ports.items()):
        port = next(p for p in model_input.ports if p.name == port_name)
        if any(needs_collation(rep, port) for rep in port.accepted_representations):
            coll = DataPlanStep("collate", inputs=(step_id,),
                                output=new_id(),
                                params={"policy": default_collation_for(port)})
            plan.append(coll)
            output_ports[port_name] = coll.output

    return DataPlan(
        steps=tuple(plan),
        output_ports=output_ports,
        warnings=tuple(warnings),
        requires_user_choice=tuple(user_ch),
    )
```

Key decisions encoded in the algorithm:

- **Per-source path discovery is independent**: each source picks the cheapest
  adapter chain to *any* accepted representation of *any* port.
- **Ambiguity escalation**: when two adapter chains have equal cost (within
  `EPSILON`), the planner appends a message to `requires_user_choice` so DAG-ML
  can refuse to auto-run or surface a UI prompt.
- **Refusal**: if no path exists and the port is non-optional, the planner
  raises `NoPlanFoundError`. ML_DATA refuses early; DAG-ML can catch and re-plan
  with a different `FusionPolicy`.
- **Lossy chains**: when `policy.allow_lossy_adapters=False`, lossy adapters are
  removed from candidate paths upstream (in `AdapterRegistry.find_path` via the
  policy).
- **Fusion**: the join step is created only when `policy.mode in
  ("concat_features", "stack_channels")`. For `dict_input` / `list_input` the
  per-source outputs are passed through.

### 7.5 `execute_fit` and `execute_transform`

```text
def execute_fit(plan, dataset, view, y, context):
    state    = {}                  # step output id -> DataBlock or FeatureTable
    fitted   = []
    for step in plan.steps:
        if step.kind == "materialize":
            block = dataset.materialize(step.params["source_id"], view)
            state[step.output] = block
        elif step.kind == "adapt":
            adapter = adapter_registry.get(step.adapter_id)
            block_in = state[step.inputs[0]]
            if adapter.spec.stateful:
                fit_ctx = context if adapter.spec.fit_scope == "fold_train" else (
                    context if context.phase == "fit_cv" else context  # passthrough
                )
                block_out, fit_obj = adapter.fit_transform(block_in, y, fit_ctx)
                fitted.append(fit_obj)
            else:
                block_out = adapter.transform(block_in, None, context)
            state[step.output] = block_out
        elif step.kind == "align":
            plan_obj = run_alignment(state, step)
            state[step.output] = plan_obj
        elif step.kind == "join":
            joiner = adapter_registry.get(step.adapter_id)
            tables = [as_feature_table(state[i]) for i in step.inputs]
            fit_obj = joiner.fit(tables, step.params["policy"], context)
            fused   = joiner.transform(tables, fit_obj, context)
            fitted.append(fit_obj)
            state[step.output] = fused
        elif step.kind == "collate":
            policy = step.params["policy"]
            collator = type_registry.get_type(state[step.inputs[0]].representation.type_id).default_collator()
            state[step.output] = collator.collate([state[step.inputs[0]]], view, policy)
    outputs = {port: state[oid] for port, oid in plan.output_ports.items()}
    return outputs, tuple(fitted)

def execute_transform(plan, dataset, view, fitted, context):
    state = {}
    fitted_iter = iter(fitted)
    for step in plan.steps:
        if step.kind == "materialize":
            state[step.output] = dataset.materialize(step.params["source_id"], view)
        elif step.kind == "adapt":
            adapter = adapter_registry.get(step.adapter_id)
            block_in = state[step.inputs[0]]
            if adapter.spec.stateful:
                state[step.output] = adapter.transform(block_in, next(fitted_iter), context)
            else:
                state[step.output] = adapter.transform(block_in, None, context)
        elif step.kind == "align":
            state[step.output] = run_alignment(state, step)
        elif step.kind == "join":
            joiner = adapter_registry.get(step.adapter_id)
            tables = [as_feature_table(state[i]) for i in step.inputs]
            state[step.output] = joiner.transform(tables, next(fitted_iter), context)
        elif step.kind == "collate":
            policy = step.params["policy"]
            collator = type_registry.get_type(state[step.inputs[0]].representation.type_id).default_collator()
            state[step.output] = collator.collate([state[step.inputs[0]]], view, policy)
    return {port: state[oid] for port, oid in plan.output_ports.items()}
```

Important: `execute_fit` returns `fitted` adapters **in the order they were
created**, matching the order in which `execute_transform` will consume them.
The caller (DAG-ML) is responsible for persisting them with the plan.

### 7.6 Questions ouvertes (section 7)

- Should `DataPlanner` expose `dry_run(plan, dataset, view) -> dict[step_id, shape]`
  so DAG-ML can check output shapes before fit? Tentative answer: yes, add it in
  a v1.1 once the first implementation exists.
- Should `requires_user_choice` carry structured alternatives (`list[Decision]`)
  rather than free-text strings? Tentative answer: yes, but defer to v1.1.

---

## 8. Collation and batching

### 8.1 CollationPolicy

```python
@dataclass(frozen=True)
class CollationPolicy:
    padding: Literal["none", "right", "left", "center"] = "none"
    truncate: bool = False
    batch_container: str | None = None       # "ndarray", "torch_tensor", "graph_batch"
    emit_mask: bool = True
    max_length: int | None = None
    pad_value: Any = 0
```

### 8.2 BatchCollator

```python
class BatchCollator(Protocol):
    def collate(
        self,
        blocks: Sequence[DataBlock],
        view: DataView,
        policy: CollationPolicy,
    ) -> DataBlock: ...
```

### 8.3 Rule: collation is last

Collation is the **last** step before the model receives the data. The reason is
that adapters operate on logical representations (per-sample sequences, per-graph
nodes), and padding / batching introduces a tensor shape that is hard to
unwind. The `DataPlanner` enforces this by emitting `collate` steps only at the
output ports, never in the middle of an adapter chain.

Padding behaviour by type:

- `dense_signal`: collator typically concatenates already-padded blocks (sizes
  match), so `padding="none"` is the default.
- `time_series`: padding required when sequences are variable; `emit_mask=True`
  to expose presence.
- `text`: padding to `max_length`; truncate-from-right by default.
- `graph`: builds a batched graph (`graph_batch` container) via the plugin's
  default collator; no padding required.
- `image_rgb`: requires identical (H, W) within a batch; if not, the collator
  raises `CollationError` (resampling is an adapter, not a collator concern).

---

## 9. Sample relations, repetitions, augmentation

### 9.1 Repetitions

`SampleRelation` (section 2.11) lets ML_DATA expose the mapping
`observation -> sample -> target -> group`. ML_DATA itself never reduces
observations to samples; it returns blocks at the observation level and lets
DAG-ML aggregate (via a model-side adapter) or split (via a splitter).

The methods of `MLDataset` always operate at the *observation level when a
source's granularity is `per_sample_repeated`*. The relation lets DAG-ML
reconstruct the higher-level mapping.

Concretely:

```text
materialize(source_id="nir", view=v)
    -> DataBlock(sample_ids=("S001", "S001", "S001", "S002", ...), ...)
                # 3 rows for S001 (rep_1, rep_2, rep_3), 3 for S002, ...

sample_relation(source_id="nir", view=v)
    -> SampleRelation(
           source_id="nir",
           observation_ids=("nir_S001_r1", "nir_S001_r2", "nir_S001_r3", ...),
           sample_ids=("S001", "S001", "S001", "S002", ...),
           target_ids=("y_S001", "y_S001", "y_S001", "y_S002", ...),
           group_ids=("plant_A", "plant_A", "plant_A", "plant_A", ...),
           origin_ids=(None, None, None, ...),                # unless augmented
       )
```

### 9.2 Alignment when observation_id != sample_id

When fusing two sources where one has multiple observations per sample (NIRS reps)
and another has one (chemistry / image), the alignment policy applies at the
**sample** level. ML_DATA broadcasts the singleton source's row across the N
observations of the repeated source:

```text
nir source:    samples ("S001", "S001", "S001", "S002")
chem source:   samples ("S001", "S002")
inner join:    canonical ("S001", "S002")  (sample level)
broadcast:     when fusing, nir rows for S001 are kept x3; chem rows for S001
               are broadcast x3 to match
```

The `AlignmentPlan.per_source_positions` carries integer arrays whose entries are
the row positions in the source; broadcast is implicit (a chem row is referenced
3 times when the repeated source has 3 observations for S001).

ML_DATA never aggregates rep_1, rep_2, rep_3 into one row. Aggregation is a
model-side concern (DAG-ML decides whether to aggregate predictions or to fit
a model that consumes repetitions).

### 9.3 Exposing `group_id` to splitters

ML_DATA exposes `SampleRelation.group_ids` so DAG-ML splitters (cross-validators)
can request:

```text
relation = dataset.sample_relation("nir", view)
groups   = relation.group_ids                # tuple of GroupId
splitter.split(X=..., y=..., groups=groups)
```

ML_DATA does **not** call any splitter, does **not** know what a fold is, and
does **not** enforce that groups are kept together. It only exposes the right
information so DAG-ML can.

### 9.4 Augmentation

Augmentation is a special adapter that produces *new rows* in a source and
declares them via `SampleRelation`.

```python
@dataclass(frozen=True)
class AugmentationPlan:
    multiplier: int                       # how many augmentations per origin
    per_sample_counts: tuple[int, ...] | None = None
    seed: int | None = None

class AugmentationAdapter(Protocol):
    @property
    def spec(self) -> AdapterSpec: ...

    def plan(
        self,
        block: DataBlock,
        policy: "AugmentationPolicy",
        context: AdapterContext,
    ) -> AugmentationPlan: ...

    def transform(
        self,
        block: DataBlock,
        plan: AugmentationPlan,
        context: AdapterContext,
    ) -> tuple[DataBlock, SampleRelation]: ...

@dataclass(frozen=True)
class AugmentationPolicy:
    apply_to: Literal["train_only", "cv_only", "all_partitions"] = "train_only"
    inherit_target: bool = True
    inherit_group: bool = True
    forbid_validation_augmentation: bool = True
    store_origin_mapping: bool = True
    seed_scope: Literal["run", "variant", "fold", "node"] = "fold"
```

Semantics of `apply_to` (the actual enforcement is performed by DAG-ML, not
ML_DATA -- this is documented here only to lock down the contract that the
two libraries share):

| Value             | CV train folds | REFIT (full train) | val / test |
|-------------------|----------------|--------------------|------------|
| `train_only` (default) | augmented   | augmented          | skipped    |
| `cv_only`         | augmented      | skipped            | skipped    |
| `all_partitions`  | augmented      | augmented          | augmented  |


ML_DATA contract for augmentation:

- The adapter returns `(DataBlock, SampleRelation)`. The returned block has
  `len(sample_ids) >= len(input.sample_ids)` (new rows appended).
- `SampleRelation.origin_ids[i]` is the `SampleId` of the original sample for
  every augmented row, and `None` for original rows.
- `AugmentationPolicy.apply_to` is **passed through** by ML_DATA; the actual
  enforcement (only apply on train, not on val) is the responsibility of
  DAG-ML, which calls `transform` with the right view.
- ML_DATA *does* enforce that `origin_id != sample_id` (raising on misuse).

DAG-ML's job is to:

- ensure the adapter runs only on the train view of a fold (when
  `apply_to="train_only"`).
- propagate `origin_ids` to its leakage check: augmented copies of a validation
  sample must never leak into training.

### 9.5 Seed propagation

`AdapterContext.random_state` is the seed handed down to the adapter. The exact
contract is:

- ML_DATA expects `random_state` to be deterministic across replays.
- DAG-ML derives `random_state` from a `SeedContext` (defined in DAG-ML, not in
  ML_DATA). ML_DATA receives the derived integer.
- ML_DATA persists `random_state` inside `FittedAdapter.artifact` when the
  adapter is stateful. The next replay will skip re-seeding (the artifact is
  used directly).

ML_DATA does **not** import or implement `SeedContext`. It only assumes the
caller derives a stable integer.

---

## 10. Auxiliary inputs (operator-aware)

Some operators (NIRS Savitzky-Golay derivative, graph readout, image patcher)
need *auxiliary* data that is not the main `DataBlock` but lives in the
descriptor or schema.

### 10.1 AuxInputSpec

```python
@dataclass(frozen=True)
class AuxInputSpec:
    name: str                       # e.g. "wavelengths", "time_coords"
    kind: Literal["axis_coordinates", "source_metadata", "schema", "side_data"]
    axis: str | None = None         # the axis name when kind == "axis_coordinates"
    source_id: SourceId | None = None
    required: bool = True
```

### 10.2 Resolution rules

ML_DATA exposes:

```python
class MLDataset(Protocol):
    ...
    def auxiliary(
        self,
        spec: AuxInputSpec,
        view: DataView,
    ) -> Any: ...
```

Resolution:

- `kind="axis_coordinates"`: returns `AxisSpec.coordinate` of the axis named
  `spec.axis` in the source `spec.source_id`. Example: `wavelengths` for NIRS.
- `kind="source_metadata"`: returns `SourceDescriptor.schema[spec.name]` or
  raises if missing.
- `kind="schema"`: returns the full `RepresentationSpec` of the source.
- `kind="side_data"`: returns a custom object the plugin chose to expose (e.g.
  variant annotation table for genotype). The plugin documents the contract.

The auxiliary path is read-only and stateless. It does not produce new blocks
and does not enter the `DataPlan`.

### 10.3 Example: NIRS wavelengths

A NIRS source registers:

```python
SourceDescriptor(
    id="nir",
    name="nir",
    type_id="dense_signal",
    modality="spectroscopy",
    native_representation=RepresentationSpec(
        id="signal_with_processings",
        type_id="dense_signal",
        rank=3,
        axes=(
            AxisSpec("sample", "sample"),
            AxisSpec("processing", "processing"),
            AxisSpec("wavelength", "wavelength", unit="nm", size=512,
                     coordinate=CoordinateSpec(
                         dtype="numeric", ordered=True,
                         values={"kind": "regular_grid", "start": 800.0, "step": 3.327})),
        ),
        container="ndarray",
        dtype="float32",
    ),
    sample_key="sample_id",
    granularity="per_sample_repeated",
    schema={"instrument": "Foss-NIRSystems"},
)
```

An adapter (e.g. `SpectralDerivative`) requests:

```python
spec = AuxInputSpec(name="wavelengths", kind="axis_coordinates",
                    axis="wavelength", source_id="nir")
coord = dataset.auxiliary(spec, view)
# coord == CoordinateSpec(dtype="numeric", ordered=True,
#                         values={"kind": "regular_grid", "start": 800.0, "step": 3.327})
# The caller materialises the explicit grid when needed:
#   wls = [coord.values["start"] + i * coord.values["step"] for i in range(size)]
```

---

## 11. Serialisation, schema fingerprint, replay

### 11.1 JSON-serialisable specs

Every dataclass listed below must round-trip through JSON without loss:

| Type                  | Serialiser entry                                                |
|-----------------------|-----------------------------------------------------------------|
| `AxisSpec`            | `{name, kind, unit, size, variable, coordinate}`                |
| `RepresentationSpec`  | `{id, type_id, rank, axes:[...], container, dtype, sparse, ragged}` |
| `SourceDescriptor`    | `{id, name, type_id, modality, native_representation, sample_key, granularity, schema, tags}` |
| `DatasetSchema`       | `{dataset_id, sample_ids, sources, targets, metadata}`          |
| `DataView`            | `{sample_ids, partition, fold_id, source_ids, columns, include_augmented, include_excluded, extra}` |
| `PresenceMask`        | `{sample_ids, source_id, present}`                              |
| `SampleRelation`      | `{source_id, observation_ids, sample_ids, target_ids, group_ids, origin_ids}` |
| `AdapterSpec`         | `{id, version, input_type, input_representation, output_representation, output_type, supervised, stateful, lossy, fit_scope, cost_hint}` |
| `AlignmentPolicy`     | `{join, reference_source, on_missing_sample}`                   |
| `FusionPolicy`        | `{mode, target_representation, alignment, missing_source, namespace_columns, allow_lossy_adapters, max_output_features}` |
| `CollationPolicy`     | `{padding, truncate, batch_container, emit_mask, max_length, pad_value}` |
| `DataPlanStep`        | `{kind, inputs, output, adapter_id, params}`                    |
| `DataPlan`            | `{steps, output_ports, warnings, requires_user_choice}`         |
| `ModelInputSpec`      | `{ports, default_fusion}`                                       |
| `AuxInputSpec`        | `{name, kind, axis, source_id, required}`                       |

Binary payloads (`DataBlock.data`, `TargetBlock.y`, `FittedAdapter.artifact`)
are **not** JSON: they go through `SerializableRef` (section 11.3) or external
store.

Canonical JSON rules (used to compute fingerprints):

- Keys are sorted lexicographically at every level.
- Tuples become arrays.
- Floats are serialised with `repr` (no rounding).
- `None` becomes `null`.
- Strings are UTF-8.
- No trailing whitespace, no indentation (single-line) in canonical form.
- Plugin-specific extension fields under `extra` and `params` follow the same
  rules.

### 11.2 schema_fingerprint

```python
def schema_fingerprint(
    schema: DatasetSchema,
    fusion: FusionPolicy | None = None,
    adapter_specs: Sequence[AdapterSpec] = (),
) -> str:
    """
    Returns a deterministic SHA-256 hex digest over
        canonical_json({
            "schema": schema,
            "fusion": fusion,
            "adapters": sorted(adapter_specs, key=lambda a: a.id),
        })
    """
```

Canonical ordering for the input:

1. `schema.sources` is sorted by `SourceDescriptor.id` before serialisation.
2. `schema.targets` and `schema.metadata` keys are sorted.
3. `adapter_specs` is sorted by `AdapterSpec.id`.
4. Inside an `AdapterSpec.cost_hint`, keys are sorted.

The fingerprint is the stable identity of the data side of a model. DAG-ML uses
it to verify that a predict-time dataset matches the train-time dataset's data
contract (predict refuses if mismatched, unless an explicit migration spec is
supplied).

### 11.3 Fitted artifacts (`SerializableRef`)

```python
@dataclass(frozen=True)
class SerializableRef:
    registry: str         # e.g. "fitted_adapters"
    type_id: str          # plugin id of the type
    version: str          # plugin / adapter version
    object_id: str        # opaque content-addressed id (e.g. sha-256 of the joblib)

class ArtifactSerializer(Protocol):
    def serialize(self, obj: Any) -> bytes: ...
    def deserialize(self, payload: bytes, ref: SerializableRef) -> Any: ...
```

ML_DATA exposes the `ArtifactSerializer` protocol but **never** stores artifacts.
DAG-ML owns the artifact store. The fitted adapter holds an in-memory `artifact`
field at runtime; before persistence, DAG-ML calls
`ArtifactSerializer.serialize(adapter.artifact)` and replaces it with a
`SerializableRef`.

### 11.4 Plugin versioning

Every `DataTypePlugin` and `RepresentationAdapter` exposes a `version` property
(`semver` string). When DAG-ML loads a `DataPlan`, it calls:

```python
def requires_plugin_versions(plan: DataPlan) -> dict[str, str]:
    """
    Returns a mapping {plugin_id: ">=1.0, <2.0"} of compatibility ranges
    derived from the adapter_ids and type_ids used in the plan.
    """
```

The default policy is `^x.y` (compatible-within-major). Plugins that change
their wire format must bump the major version. ML_DATA refuses to load a plan
when a required plugin is missing or out of range; the message includes the
mismatched range.

---

## 12. Shared contract with DAG-ML

The shared contract types are exported from a single module so DAG-ML and other
consumers can import them without dragging in storage / adapter machinery.

### 12.1 Module layout

```python
# ml_data/contract.py
from ml_data.contract import (
    SampleId, SourceId, RepresentationId, TypeId, ObservationId, TargetId, GroupId,
    AxisKind, AxisSpec, RepresentationSpec,
    SourceGranularity, SourceDescriptor, DatasetSchema,
    DataView, PresenceMask, DataBlock, FeatureTable, TargetBlock, SampleRelation,
    InputPortSpec, ModelInputSpec, AuxInputSpec,
    AlignmentPolicy, FusionPolicy, CollationPolicy,
    AdapterSpec, AdapterContext, FittedAdapter, AdaptationPolicy,
    DataPlanStep, DataPlan,
    SerializableRef,
)
```

`ml_data.contract` is the **only** module DAG-ML imports from ML_DATA. The
implementation-side modules (`ml_data.adapters`, `ml_data.plugins`,
`ml_data.storage`, `ml_data.planner`) are not part of the contract.

### 12.2 Import convention

```python
from ml_data.contract import (
    DataView, DataBlock, ModelInputSpec, FusionPolicy, DataPlan,
)
```

DAG-ML never does `from ml_data.adapters import SpectraFlattenAdapter`. That
would couple DAG-ML to the adapter catalogue. Instead, DAG-ML calls
`adapter_registry.get(adapter_id)` through ML_DATA's registry indirection.

### 12.3 Type stability

Once `ml_data/contract.py` is published at version 1.0:

- No field is renamed or removed without a major bump.
- New fields are added with defaults (`field(default=..., kw_only=True)` style).
- Dataclasses remain frozen.
- Protocols remain structural (no required methods removed).

---

## 13. Non-buts explicites

ML_DATA explicitly refuses to do the following. Implementations that drift into
these areas violate the spec.

| Concern                                   | Owned by | ML_DATA behaviour                                          |
|-------------------------------------------|----------|------------------------------------------------------------|
| Execution graph                           | DAG-ML   | Not modelled.                                              |
| ML phases (FIT_CV / SELECT / REFIT / ...) | DAG-ML   | Receives `phase` opaque on `AdapterContext`.               |
| OOF / no-leakage invariant                | DAG-ML   | Exposes building blocks (`origin_ids`, `group_ids`, `fit_scope`). |
| Fold construction / CV splitter           | DAG-ML   | Exposes `SampleRelation`; never invents folds.             |
| Hyperparameter search                     | DAG-ML   | Not modelled.                                              |
| Variant-level parallelism                 | DAG-ML   | `MLDataset` may be queried concurrently; thread-safety is the implementation's responsibility. |
| Refit                                     | DAG-ML   | `AdapterContext.phase` may say `"refit"`; ML_DATA does not gate behaviour on it. |
| Cache of execution results                | DAG-ML   | ML_DATA may cache blocks internally (LRU on materialize), but the *execution cache* (memo of `(node, params, inputs) -> result`) is DAG-ML's. |
| Prediction store                          | DAG-ML   | Not modelled.                                              |
| Refusal of plans on ML grounds            | DAG-ML   | ML_DATA refuses only when no path / no schema match; never on ML invariants. |

---

## 14. Extension checklist

To add a new domain type, follow this checklist.

### 14.1 Generic checklist

1. **Declare the plugin**: implement `DataTypePlugin` with `type_id`, `version`,
   `capability()`, `validate()`, `infer_source()`, `known_representations()`,
   `default_collator()`.
2. **Declare the representations**: enumerate the `RepresentationSpec`s the type
   exposes, including axes, container, dtype.
3. **Declare the adapters**: implement at least one
   `RepresentationAdapter` to `tabular_numeric` so the type can feed
   tree-based / linear models. Additional adapters target tensor / sequence /
   graph representations.
4. **Register the loader**: implement a `MLDataset` backend or a
   `SourceDescriptor` factory that reads from disk / memory and emits
   `DataBlock` instances conforming to the representation.
5. **Add a unit-test pack**: a small dataset, a `materialize` round-trip, a
   path-to-`tabular_numeric` test, an alignment test with another source.

### 14.2 Concrete examples

| Domain                  | Type plugin candidate           | Native representation                                  | Adapter(s) to `tabular_numeric`                                       | Axes / container                                                |
|-------------------------|----------------------------------|---------------------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------|
| Genotype variant matrix | `genotype_matrix` (core)         | `variant_matrix` (int8 0/1/2/-1) or `dosage_matrix` (float) | `genotype.dosage` (lossless), `genotype.pca` (lossy, stateful)         | `(sample, variant)`; ndarray int8 / float32                     |
| Hyperspectral cube      | `hyperspectral_cube` (new)       | `cube_hwb`                                              | `hsi.spectral_flatten`, `hsi.spatial_mean`, `hsi.pca_per_pixel`        | `(sample, h, w, band)`; ndarray float32                         |
| Mass spectrometry       | `mass_spec` (new)                | `centroided_spectrum` (ragged) or `binned_spectrum`     | `ms.bin_to_table`, `ms.peak_extract`, `ms.embedding`                   | `(sample, mz)` ragged or `(sample, bin)` dense                  |
| Electronic nose         | `e_nose` (new)                   | `sensor_array_series`                                   | `enose.aggregate` (mean/peak/area), `enose.sequence`                   | `(sample, time, sensor)`; ndarray float32                       |
| Raman spectrum          | `dense_signal` (reuse)           | `signal_with_processings`                               | reuse `spectra.flatten`, `spectra.resample`                            | identical to NIRS                                               |
| Fluorescence (EEM)      | `eem_matrix` (new)               | `excitation_emission_matrix`                            | `eem.unfold`, `eem.parafac` (lossy, stateful)                          | `(sample, excitation, emission)`; ndarray float32               |
| Multispectral satellite | `multichannel_image` (core)      | `mc_image`                                              | reuse `image.embedding`, add `satellite.index_features` (NDVI, NDWI, ...) | `(sample, h, w, band)`                                          |
| IR thermography         | `multichannel_image` (core)      | `mc_image` (single-band float)                          | `image.embedding`, `thermal.stat_features`                             | `(sample, h, w)` or `(sample, h, w, 1)`                         |

### 14.3 Built-in representation catalogue

The Rust `builtin_models` module exposes the supported baseline catalogue below.
Each row is a canonical `RepresentationSpec`; alternate storage technologies
such as Arrow, xarray, JSON metadata payloads or image-batch wrappers are host
import/export profiles unless the contract later adds an explicit multi-container
field.

| Domain | `type_id` | `representation_id` | Axes | Canonical container |
|--------|-----------|---------------------|------|---------------------|
| NIRS / spectra | `dense_signal` | `signal_1d` | `(sample, wavelength)` | `ndarray` |
| NIRS / spectra with processings | `dense_signal` | `signal_with_processings` | `(sample, processing, wavelength)` | `ndarray` |
| Raman | `dense_signal` | `raman_signal` | `(sample, wavenumber)` | `ndarray` |
| FTIR | `dense_signal` | `ftir_signal` | `(sample, wavenumber)` | `ndarray` |
| Numeric tabular | `table` | `tabular_numeric` | `(sample, feature)` | `dataframe` |
| Mixed tabular | `table` | `tabular_mixed` | `(sample, column)` | `dataframe` |
| Named feature blocks | `multi_block` | `feature_block_set` | `(sample, block, feature)` | `feature_block_set` |
| Time/climate series | `time_series` | `series_mv` | `(sample, time, variable)` | `ndarray` |
| Genotype variants | `genotype_matrix` | `variant_matrix` | `(sample, variant)` | `ndarray` |
| Genotype dosage | `genotype_matrix` | `dosage_matrix` | `(sample, variant)` | `ndarray` |
| RGB image | `image_rgb` | `rgb_image` | `(sample, height, width, channel)` | `ndarray` |
| Grayscale image | `gray_image` | `gray_image` | `(sample, height, width)` | `ndarray` |
| Multichannel image | `multichannel_image` | `mc_image` | `(sample, height, width, channel)` | `ndarray` |
| Multispectral image | `multichannel_image` | `multispectral_image` | `(sample, height, width, band)` | `ndarray` |
| Hyperspectral cube | `hyperspectral_cube` | `cube_hwb` | `(sample, height, width, band)` | `ndarray` |
| Segmentation mask | `label_mask` | `segmentation_mask` | `(sample, height, width)` | `ndarray` |
| ROI mask | `label_mask` | `roi_mask` | `(sample, height, width)` | `ndarray` |
| Sample metadata | `metadata` | `sample_metadata` | `(sample, field)` | `dataframe` |
| Numeric target | `target` | `target_numeric` | `(sample)` | `array` |
| Categorical target | `target` | `target_categorical` | `(sample)` | `array` |
| Multivariate numeric target | `target` | `target_numeric_matrix` | `(sample, target)` | `array` |
| Multivariate categorical target | `target` | `target_categorical_matrix` | `(sample, target)` | `array` |
| Mass spectrum | `mass_spec` | `mass_spectrum` | `(sample, mz)` | `ragged_array` |
| Raw text | `text` | `text_raw` | `(sample)` | `list` |
| Token ids | `text` | `text_token_ids` | `(sample, token)` | `ragged_array` |

`target_numeric`, `target_categorical`, `target_numeric_matrix` and
`target_categorical_matrix` are target-side contracts. The default tabular
model-input helper does not accept them as feature sources.

### 14.4 Anti-patterns

- **Do not** add a domain-specific knob (e.g. `wavelengths=`) directly on
  `MLDataset`. Add it to `AxisSpec.coordinate` and request it via
  `AuxInputSpec(kind="axis_coordinates")`.
- **Do not** flatten a multidimensional structure inside the type plugin and
  expose only `tabular_numeric`. Always expose the native representation;
  flattening is an adapter.
- **Do not** alter `SampleId` semantics per type. Sample id is a global,
  type-agnostic notion. If a type has rows that are not per-sample, use
  `granularity` to declare so.
- **Do not** mutate `DataBlock.data` in adapters. Always produce a new block.

---

## 15. Implementation notes

### 15.1 Performance

- **Block immutability + copy-on-write**: when an adapter only reorders rows,
  the implementation should use views (`numpy` reshape / slicing) rather than
  copies. The immutability rule says "no mutation by *consumers*", not "no
  in-place op by *producers*".
- **Block cache (LRU)**: `MLDataset.materialize` is the only expensive call.
  Implementations may cache the last K (`source_id, view_hash`) tuples in
  memory. Cache hit rate must be reported as a metric for observability.
- **Lazy axis coordinates**: large coordinate tuples (e.g. 4096 wavelengths)
  should be stored once per source, not per block.

### 15.2 Thread safety

- `MLDataset` must be safe for concurrent reads when used by DAG-ML's
  variant-level parallelism. Implementations should not hold per-call mutable
  state.
- Adapters' `transform()` must be thread-safe when called with the same
  `fitted` argument from multiple threads.
- Registries (`DataTypeRegistry`, `AdapterRegistry`) are populated at startup
  and treated as immutable at run-time; runtime registration is allowed but
  must be guarded by a lock owned by the registry.

### 15.3 Errors

ML_DATA exposes a small, named set of exceptions:

- `MLDataError` (root).
- `SchemaError`, `AlignmentError`, `RepresentationError`, `AdaptationError`,
  `NoPlanFoundError`, `PortArityError`, `CollationError`, `FusionError`,
  `StatefulAdapterMisuse`, `PluginVersionError`,
  `DatasetRequiredForPlanning` (raised by `resolve_from_schema` when an
  adapter cannot be picked without inspecting the materialised dataset; the
  caller is expected to retry via `resolve(dataset, ...)`).

Every exception carries a structured `payload` dict (JSON-serialisable) so
DAG-ML can surface it without parsing strings.

---

## 16. Questions ouvertes (cross-section)

These are explicit unknowns the implementer must close in v1.1:

1. **Streaming / out-of-core datasets**: should `MLDataset.materialize` return
   a lazy block (e.g. a `dask` array) instead of a materialised `ndarray`? The
   current spec assumes eager materialisation. A `LazyDataBlock` companion
   could be introduced without breaking the contract.
2. **Multi-target sources**: should `TargetBlock` support multi-output regression
   natively (rank-2 `y`) or rely on multiple separate `TargetBlock`s? The
   current spec allows rank-2 via `representation.axes`, but the convention
   is not enforced.
3. **Partial-fit adapters**: do we need a `partial_fit` capability on
   `RepresentationAdapter` for incremental learning? Currently no; DAG-ML can
   express it as a sequence of `fit` calls with adapter-specific state.
4. **Cross-source adapters**: can an adapter consume two `DataBlock`s and
   produce one (e.g. CCA between NIRS and image embeddings)? Currently no:
   the spec defines `RepresentationAdapter.transform` over a single block.
   A `JointAdapter` protocol may be added in v1.1.
5. **Multi-row predictions and aggregation**: should ML_DATA provide an
   `AggregationAdapter` for observation -> sample reduction? Currently no;
   it is part of DAG-ML's prediction layer. ML_DATA only provides the
   `SampleRelation` so DAG-ML can reduce predictions itself.
