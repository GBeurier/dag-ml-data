"""Optional package self-test for the ctypes provider shim.

Skips when the C ABI cdylib cannot be located or the shared fixtures are absent,
so it never fails in an environment that has not built the cdylib. Build it with
`cargo build -p dag-ml-data-capi --lib`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dag_ml_data_provider import InMemoryProvider, find_capi_library

_WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
_ENVELOPE = _WORKSPACE_ROOT / "examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
_REQUEST = _WORKSPACE_ROOT / "examples/fixtures/oof_campaign/materialization_request_model_base_x.json"

_F64_FEATURE_MATRICES = [
    {
        "feature_set_id": "x",
        "representation_id": "tabular_numeric",
        "feature_names": ["f0", "f1"],
        "observation_ids": [
            "obs.S001.base",
            "obs.S001.rep1",
            "obs.S001.aug0",
            "obs.S002.base",
        ],
        "values": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
    }
]


def _provider() -> InMemoryProvider:
    try:
        find_capi_library()
    except FileNotFoundError:
        pytest.skip("dag-ml-data C ABI cdylib not found; build with `cargo build -p dag-ml-data-capi --lib`")
    if not _ENVELOPE.is_file() or not _REQUEST.is_file():
        pytest.skip("shared oof_campaign fixtures not found")
    return InMemoryProvider.from_files(
        _ENVELOPE,
        f64_feature_matrices=_F64_FEATURE_MATRICES,
    )


def test_rejects_both_feature_inputs() -> None:
    # The guard fires before the cdylib is loaded, so this needs no library.
    with pytest.raises(ValueError):
        InMemoryProvider(b"{}", feature_tables=[], f64_feature_matrices=[])


def test_discovery_returns_existing_library() -> None:
    try:
        path = find_capi_library()
    except FileNotFoundError:
        pytest.skip("dag-ml-data C ABI cdylib not found")
    assert path.is_file()


def test_materialize_view_and_feature_values() -> None:
    with _provider() as provider:
        data_handle = provider.materialize_file(_REQUEST)
        view_handle = provider.make_view(
            data_handle,
            {"sample_ids": ["S002", "S001"], "include_augmented": False},
        )
        features = provider.feature_values(view_handle, "x")
        observations = [row["observation_id"] for row in features]
        assert observations == ["obs.S002.base", "obs.S001.base", "obs.S001.rep1"]
        assert features[0]["features"]["f0"] == 4.0
        assert features[1]["features"]["f1"] == 10.0


def test_feature_tensor_emits_mask() -> None:
    with _provider() as provider:
        data_handle = provider.materialize_file(_REQUEST)
        view_handle = provider.make_view(
            data_handle,
            {"sample_ids": ["S002", "S001"], "include_augmented": False},
        )
        tensor = provider.feature_tensor(view_handle, {"feature_set_id": "x", "policy": {"emit_mask": True}})
        assert tensor["shape"] == [3, 2]
        assert tensor["feature_names"] == ["f0", "f1"]
