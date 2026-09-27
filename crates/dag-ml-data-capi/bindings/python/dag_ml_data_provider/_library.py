"""Discovery and loading of the dag-ml-data C ABI cdylib.

The platform wheel bundles and integrity-checks `dag_ml_data_capi`. Explicit
paths and the existing source-checkout discovery order remain supported.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from ._abi import configure_library

_MANIFEST_NAME = "native-library.json"
_MANIFEST_SCHEMA_VERSION = 1


class NativeLibraryError(RuntimeError):
    """Base error for failures that happen before the native ABI is used."""


class NativeLibraryNotFoundError(NativeLibraryError):
    """Raised when the platform library is absent from the installed package."""


class NativeLibraryIntegrityError(NativeLibraryError):
    """Raised when the bundled library or its provenance manifest is invalid."""


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libdag_ml_data_capi.dylib"
    if sys.platform == "win32":
        return "dag_ml_data_capi.dll"
    return "libdag_ml_data_capi.so"


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise NativeLibraryNotFoundError(
            f"bundled native-library manifest is missing from the installed package: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeLibraryIntegrityError(
            f"bundled native-library manifest is unreadable or invalid JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise NativeLibraryIntegrityError(
            "bundled native-library manifest must be a JSON object"
        )
    required = {
        "schema_version",
        "package",
        "package_version",
        "cargo_package",
        "library",
        "sha256",
        "size_bytes",
        "rust_target",
        "rustc",
        "source_commit",
        "source_dirty",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - required)
        raise NativeLibraryIntegrityError(
            "bundled native-library manifest fields are not the closed v1 contract "
            f"(missing={missing}, unknown={unknown})"
        )
    if payload["schema_version"] != _MANIFEST_SCHEMA_VERSION:
        raise NativeLibraryIntegrityError(
            "unsupported bundled native-library manifest schema_version "
            f"{payload['schema_version']!r}"
        )
    if (
        payload["package"] != "dag-ml-data-provider"
        or payload["cargo_package"] != "dag-ml-data-capi"
    ):
        raise NativeLibraryIntegrityError(
            "bundled native-library manifest identifies the wrong package"
        )
    if payload["library"] != _library_filename():
        raise NativeLibraryIntegrityError(
            f"bundled native-library manifest names {payload['library']!r}, "
            f"expected {_library_filename()!r} for {sys.platform}"
        )
    digest = payload["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise NativeLibraryIntegrityError(
            "bundled native-library manifest sha256 is not lowercase hexadecimal"
        )
    if not isinstance(payload["size_bytes"], int) or payload["size_bytes"] <= 0:
        raise NativeLibraryIntegrityError(
            "bundled native-library manifest size_bytes must be positive"
        )
    for field in ("package_version", "rust_target", "rustc", "source_commit"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise NativeLibraryIntegrityError(
                f"bundled native-library manifest {field} must be a non-empty string"
            )
    if not isinstance(payload["source_dirty"], bool):
        raise NativeLibraryIntegrityError(
            "bundled native-library manifest source_dirty must be a boolean"
        )
    return payload


def _candidate_paths() -> list[Path]:
    filename = _library_filename()
    candidates: list[Path] = []

    env_path = os.environ.get("DAG_ML_DATA_CAPI_LIB")
    if env_path:
        candidates.append(Path(env_path))

    # The platform wheel contains a native library and a provenance manifest.
    candidates.append(_package_root() / ".libs" / filename)

    # Source-checkout convenience: locate the cdylib under the Cargo target dir.
    # parents[5] of .../crates/dag-ml-data-capi/bindings/python/dag_ml_data_provider/_library.py
    # is the workspace root; when pip-installed this is not a checkout and the
    # fallback is skipped (guarded by the workspace `Cargo.toml`).
    resolved = Path(__file__).resolve()
    workspace_root = resolved.parents[5] if len(resolved.parents) >= 6 else None

    target_dirs: list[Path] = []
    cargo_target = os.environ.get("CARGO_TARGET_DIR")
    if cargo_target:
        cargo_target_path = Path(cargo_target)
        if not cargo_target_path.is_absolute() and workspace_root is not None:
            # Cargo resolves a relative CARGO_TARGET_DIR against the workspace root,
            # not the current working directory.
            cargo_target_path = workspace_root / cargo_target_path
        target_dirs.append(cargo_target_path)
    if workspace_root is not None and (workspace_root / "Cargo.toml").is_file():
        target_dirs.append(workspace_root / "target")

    for target_dir in target_dirs:
        candidates.append(target_dir / "debug" / filename)
        candidates.append(target_dir / "release" / filename)

    return candidates


def _verified_bundled_library(package_root: Path) -> Path:
    """Validate the wheel's native library against its package-local manifest."""
    manifest = _read_manifest(package_root / _MANIFEST_NAME)
    library = package_root / ".libs" / _library_filename()
    if not library.is_file():
        raise NativeLibraryNotFoundError(
            f"bundled native library is missing from the installed package: {library}"
        )
    try:
        size = library.stat().st_size
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
    except OSError as exc:
        raise NativeLibraryIntegrityError(
            f"bundled native library cannot be read for integrity verification: {library}"
        ) from exc
    if size != manifest["size_bytes"] or digest != manifest["sha256"]:
        raise NativeLibraryIntegrityError(
            "bundled native library failed integrity verification "
            f"(expected size={manifest['size_bytes']} sha256={manifest['sha256']}, "
            f"actual size={size} sha256={digest})"
        )
    return library


def find_capi_library() -> Path:
    """Returns the first existing candidate path for the C ABI cdylib.

    Raises ``FileNotFoundError`` with the searched locations if none is found.
    """
    candidates = _candidate_paths()
    package_root = _package_root()
    bundled = package_root / ".libs" / _library_filename()
    for candidate in candidates:
        if candidate == bundled and ((package_root / _MANIFEST_NAME).exists() or bundled.exists()):
            return _verified_bundled_library(package_root)
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "could not locate the dag-ml-data C ABI cdylib "
        f"({_library_filename()}). Set DAG_ML_DATA_CAPI_LIB, pass "
        "library_path=..., or build it with "
        "`cargo build -p dag-ml-data-capi --lib`. Searched:\n  " + searched
    )


def load_library(library_path: str | Path | None = None) -> ctypes.CDLL:
    """Loads and configures the C ABI cdylib.

    When ``library_path`` is ``None`` the library is discovered via
    :func:`find_capi_library`.
    """
    path = Path(library_path) if library_path is not None else find_capi_library()
    try:
        return configure_library(ctypes.CDLL(str(path)))
    except OSError as exc:
        bundled = _package_root() / ".libs" / _library_filename()
        if library_path is None and path == bundled:
            raise NativeLibraryIntegrityError(
                f"bundled native library passed hashing but could not be loaded: {path}: {exc}"
            ) from exc
        raise
