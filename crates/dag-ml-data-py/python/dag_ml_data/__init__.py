"""Typed Python facade for dag-ml-data JSON contracts."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version as _distribution_version
from os import PathLike
from pathlib import Path
from typing import Any

from ._dag_ml_data import (
    DagMlDataCompatibilityError,
    DagMlDataContractError,
    DagMlDataError,
    DagMlDataValidationError,
    build_coordinator_data_plan_envelope_json,
    contract_manifest_json as _native_contract_manifest_json,
    data_plan_fingerprint_json,
    dataset_schema_fingerprint_json,
    fold_set_fingerprint_json,
    plan_model_input_json,
    sample_relation_table_fingerprint_json,
    validate_adapter_registry_json,
    validate_coordinator_data_plan_envelope_json,
    validate_data_plan_json,
    validate_dataset_schema_json,
    validate_fold_set_against_sample_relations_json,
    validate_fold_set_json,
    validate_model_input_spec_json,
    validate_sample_relation_table_json,
    version as _native_version,
)

try:
    __version__ = _distribution_version("dag-ml-data")
except PackageNotFoundError:
    __version__ = _native_version()


_FACADE_EXPORTS = [
    "JsonContract",
    "DatasetSchema",
    "ModelInputSpec",
    "AdapterRegistry",
    "DataPlan",
    "SampleRelationTable",
    "FoldSet",
    "CoordinatorDataPlanEnvelope",
    "plan_model_input",
    "build_coordinator_data_plan_envelope",
    "validate_fold_set_against_sample_relations",
]


def _coerce_json(value: Any) -> str:
    if isinstance(value, JsonContract):
        return value.json()
    if isinstance(value, PathLike):
        return Path(value).read_text(encoding="utf-8")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class JsonContract:
    """Immutable validated JSON contract wrapper."""

    __slots__ = ("_json",)

    def __init__(self, value: Any) -> None:
        json_text = _coerce_json(value)
        self._validate_json(json_text)
        self._json = json_text

    @classmethod
    def from_path(cls, path: str | PathLike[str]) -> "JsonContract":
        return cls(Path(path))

    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        json.loads(json_text)

    def json(self) -> str:
        return self._json

    def to_dict(self) -> Any:
        return json.loads(self._json)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._json!r})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self._json == other._json


class DatasetSchema(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_dataset_schema_json(json_text)

    def fingerprint(self) -> str:
        return dataset_schema_fingerprint_json(self._json)


class ModelInputSpec(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_model_input_spec_json(json_text)


class AdapterRegistry(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_adapter_registry_json(json_text)


class DataPlan(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_data_plan_json(json_text)

    def fingerprint(self) -> str:
        return data_plan_fingerprint_json(self._json)


class SampleRelationTable(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_sample_relation_table_json(json_text)

    def fingerprint(self) -> str:
        return sample_relation_table_fingerprint_json(self._json)


class FoldSet(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_fold_set_json(json_text)

    def fingerprint(self) -> str:
        return fold_set_fingerprint_json(self._json)

    def validate_against(self, sample_relations: Any) -> None:
        validate_fold_set_against_sample_relations_json(
            self._json,
            _coerce_json(sample_relations),
        )


class CoordinatorDataPlanEnvelope(JsonContract):
    @classmethod
    def _validate_json(cls, json_text: str) -> None:
        validate_coordinator_data_plan_envelope_json(json_text)


def version() -> str:
    """Return the installed Python package version."""
    return __version__


def contract_manifest_json() -> str:
    """Return the native contract manifest plus Python packaging metadata."""
    manifest = json.loads(_native_contract_manifest_json())
    manifest["python_package_version"] = __version__
    manifest["python_facade_exports"] = _FACADE_EXPORTS
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def plan_model_input(
    schema: Any,
    model_input: Any,
    adapter_registry: Any,
    request: Any,
) -> DataPlan:
    """Plan a model input and return a validated data-plan wrapper."""
    return DataPlan(
        plan_model_input_json(
            _coerce_json(schema),
            _coerce_json(model_input),
            _coerce_json(adapter_registry),
            _coerce_json(request),
        )
    )


def build_coordinator_data_plan_envelope(
    schema: Any,
    data_plan: Any,
    sample_relations: Any | None = None,
) -> CoordinatorDataPlanEnvelope:
    """Build a validated coordinator data-plan envelope wrapper."""
    relations_json = None
    if sample_relations is not None:
        relations_json = _coerce_json(sample_relations)
    return CoordinatorDataPlanEnvelope(
        build_coordinator_data_plan_envelope_json(
            _coerce_json(schema),
            _coerce_json(data_plan),
            relations_json,
        )
    )


def validate_fold_set_against_sample_relations(
    fold_set: Any,
    sample_relations: Any,
) -> None:
    """Validate fold boundaries against group and augmentation-origin relations."""
    validate_fold_set_against_sample_relations_json(
        _coerce_json(fold_set),
        _coerce_json(sample_relations),
    )


__all__ = [
    "__version__",
    "AdapterRegistry",
    "CoordinatorDataPlanEnvelope",
    "DataPlan",
    "DagMlDataCompatibilityError",
    "DagMlDataContractError",
    "DagMlDataError",
    "DagMlDataValidationError",
    "DatasetSchema",
    "FoldSet",
    "JsonContract",
    "ModelInputSpec",
    "SampleRelationTable",
    "build_coordinator_data_plan_envelope",
    "build_coordinator_data_plan_envelope_json",
    "contract_manifest_json",
    "data_plan_fingerprint_json",
    "dataset_schema_fingerprint_json",
    "fold_set_fingerprint_json",
    "plan_model_input",
    "plan_model_input_json",
    "sample_relation_table_fingerprint_json",
    "validate_adapter_registry_json",
    "validate_coordinator_data_plan_envelope_json",
    "validate_data_plan_json",
    "validate_dataset_schema_json",
    "validate_fold_set_against_sample_relations_json",
    "validate_fold_set_json",
    "validate_model_input_spec_json",
    "validate_sample_relation_table_json",
    "version",
]
