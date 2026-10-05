"""Optional package self-test for the ctypes provider shim.

Skips when the C ABI cdylib cannot be located or the shared fixtures are absent,
so it never fails in an environment that has not built the cdylib. Build it with
`cargo build -p dag-ml-data-capi --lib`.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import struct
import weakref
import zipfile
from pathlib import Path

import pytest
from dag_ml_data_provider import (
    __version__,
    InMemoryProvider,
    NativeLibraryIntegrityError,
    NativeLibraryNotFoundError,
    PackageProvider,
    find_capi_library,
)
from dag_ml_data_provider import _library as library_module
from dag_ml_data_provider import _provider as implementation
from dag_ml_data_provider import _provider as provider_module

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


def _write_n4d_fixture(path: Path) -> None:
    """Create the small deterministic v1 buffer store used by this test."""
    payload = bytearray(b"N4DF")
    payload.extend(struct.pack("<II", 1, 1))

    def write_string(value: str) -> None:
        encoded = value.encode("utf-8")
        payload.extend(struct.pack("<I", len(encoded)))
        payload.extend(encoded)

    write_string("x")
    write_string("tabular_numeric")
    payload.extend(struct.pack("<IIB", 4, 2, 0))
    for value in ("f0", "f1", "obs.S001.base", "obs.S001.rep1", "obs.S001.aug0", "obs.S002.base"):
        write_string(value)
    payload.extend(struct.pack("<8d", 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0))
    payload.extend(hashlib.sha256(payload).digest())
    path.write_bytes(payload)


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


def test_synchronous_buffers_are_released_after_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    refs = []
    original = implementation._u8_buffer

    def observed(payload):
        buffer, pointer = original(payload)
        refs.append(weakref.ref(buffer))
        return buffer, pointer

    monkeypatch.setattr(implementation, "_u8_buffer", observed)
    provider = _provider()
    for _ in range(100):
        data = provider.materialize_file(_REQUEST)
        view = provider.make_view(data, {"sample_ids": ["S001"]})
        assert provider.feature_values(view, "x")
        provider.release(view)
        provider.release(data)
    gc.collect()
    assert len(refs) >= 303
    assert all(ref() is None for ref in refs)
    provider.close()
    provider.close()  # destruction is idempotent
    assert all(ref() is None for ref in refs)
    with pytest.raises(RuntimeError, match="provider is closed"):
        provider.materialize_file(_REQUEST)
    with pytest.raises(RuntimeError, match="provider is closed"):
        provider.release(1)


def test_unknown_or_unresolved_view_does_not_select_all_samples() -> None:
    with _provider() as provider:
        data = provider.materialize_file(_REQUEST)
        for view in [{"sampl_ids": ["S001"]}, {"partition": "unknown"}, {"fold_id": "unknown"}]:
            with pytest.raises(RuntimeError, match="make_view failed"):
                provider.make_view(data, view)
        selected = provider.make_view(data, {"sample_ids": ["S002"], "partition": "fold_validation", "fold_id": "fold0"})
        assert {row["sample_id"] for row in provider.view_identity(selected)} == {"S002"}


@pytest.mark.parametrize(
    "resource",
    ["", " ../features.n4d", "../features.n4d", "/features.n4d", "a//b", "a\\b"],
)
def test_package_provider_refuses_unsafe_resource_names(resource: str) -> None:
    # Resource validation happens before native-library discovery.
    with pytest.raises(ValueError):
        PackageProvider.from_package_resources(
            "dag_ml_data_provider",
            envelope_resource=resource,
            feature_store_resource="features.n4d",
        )


def test_package_provider_refuses_resource_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.n4d"
    outside.write_bytes(b"outside")
    package = tmp_path / "fixture_escape"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "escape.n4d").symlink_to(outside)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    with pytest.raises(ValueError, match="symlink escapes"):
        provider_module._package_resource("fixture_escape", "escape.n4d")


def test_package_provider_refuses_namespace_package_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside-namespace.n4d"
    outside.write_bytes(b"outside")
    namespace_package = tmp_path / "fixture_namespace_escape"
    namespace_package.mkdir()
    (namespace_package / "escape.n4d").symlink_to(outside)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    with pytest.raises(ValueError, match="namespace packages are not supported"):
        provider_module._package_resource(
            "fixture_namespace_escape", "escape.n4d"
        )


def test_package_resource_accepts_single_root_zip_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "fixture-resource.zip"
    with zipfile.ZipFile(archive, "w") as fixture_zip:
        fixture_zip.writestr("fixture_zip_resource/__init__.py", "")
        fixture_zip.writestr("fixture_zip_resource/features.n4d", b"fixture")
    monkeypatch.syspath_prepend(str(archive))
    importlib.invalidate_caches()
    resource = provider_module._package_resource(
        "fixture_zip_resource", "features.n4d"
    )
    assert resource.read_bytes() == b"fixture"


def _write_bundle(root: Path, payload: bytes = b"native-test-library") -> Path:
    library = root / ".libs" / library_module._library_filename()
    library.parent.mkdir(parents=True)
    library.write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "package": "dag-ml-data-provider",
        "package_version": __version__,
        "cargo_package": "dag-ml-data-capi",
        "library": library.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "rust_target": "test-target",
        "rustc": "rustc test",
        "source_commit": "0" * 40,
        "source_dirty": False,
    }
    (root / "native-library.json").write_text(json.dumps(manifest), encoding="utf-8")
    return library


def test_discovery_verifies_package_local_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _write_bundle(tmp_path)
    monkeypatch.setattr(library_module, "_package_root", lambda: tmp_path)
    monkeypatch.setenv("DAG_ML_DATA_CAPI_LIB", str(tmp_path / "outside.so"))
    assert find_capi_library() == library


def test_discovery_refuses_missing_bundled_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _write_bundle(tmp_path)
    library.unlink()
    monkeypatch.setattr(library_module, "_package_root", lambda: tmp_path)
    with pytest.raises(NativeLibraryNotFoundError):
        find_capi_library()


def test_discovery_refuses_altered_bundled_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _write_bundle(tmp_path)
    library.write_bytes(b"altered")
    monkeypatch.setattr(library_module, "_package_root", lambda: tmp_path)
    with pytest.raises(NativeLibraryIntegrityError):
        find_capi_library()


def test_discovery_refuses_altered_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bundle(tmp_path)
    manifest_path = tmp_path / "native-library.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(library_module, "_package_root", lambda: tmp_path)
    with pytest.raises(NativeLibraryIntegrityError):
        find_capi_library()


def test_package_provider_releases_buffers_and_rejects_use_after_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        find_capi_library()
    except FileNotFoundError:
        pytest.skip("dag-ml-data C ABI cdylib not found")
    store = tmp_path / "features.n4d"
    _write_n4d_fixture(store)
    refs = []
    original = provider_module._u8_buffer

    def observed(payload):
        buffer, pointer = original(payload)
        refs.append(weakref.ref(buffer))
        return buffer, pointer

    monkeypatch.setattr(provider_module, "_u8_buffer", observed)
    provider = PackageProvider.from_files(_ENVELOPE, store)
    for _ in range(30):
        data = provider.materialize_file(_REQUEST)
        view = provider.make_view(data, {"sample_ids": ["S001"]})
        assert provider.feature_values(view, "x")
        provider.release(view)
        provider.release(data)
    gc.collect()
    assert len(refs) >= 93
    assert all(ref() is None for ref in refs)
    provider.close()
    with pytest.raises(RuntimeError, match="provider is closed"):
        provider.materialize_file(_REQUEST)
    with pytest.raises(RuntimeError, match="provider is closed"):
        provider.release(1)


def test_scientific_relations_survive_c_arrow_transport():
    envelope = json.loads(_ENVELOPE.read_text())
    row = envelope["coordinator_relations"]["records"][0]
    row.update(unit_level="combo", unit_id="S001.combo", rep_id="rep.0",
               derived_unit_id="derived.S001", component_observation_ids=["obs.S001.base"],
               sample_influence_weight=0.25, quality_flag="ok", metadata={"z": {"b": 2, "a": 1}}, tags=["rich"])
    with InMemoryProvider(json.dumps(envelope).encode()) as provider:
        data = provider.materialize(json.loads(_REQUEST.read_text()))
        view = provider.make_view(data, {"sample_ids": ["S001"]})
        exported = next(item for item in provider.view_identity(view) if item["observation_id"] == row["observation_id"])
        for key in ("unit_level", "unit_id", "rep_id", "derived_unit_id", "component_observation_ids",
                    "sample_influence_weight", "quality_flag", "metadata", "tags"):
            assert exported[key] == row[key]
