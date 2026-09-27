"""Hatch wheel hook that embeds the native provider library and provenance."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libdag_ml_data_capi.dylib"
    if sys.platform == "win32":
        return "dag_ml_data_capi.dll"
    return "libdag_ml_data_capi.so"


def _run_text(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _platform_tag(rust_target: str) -> str:
    """Tag the compiled architecture, never a broader Python interpreter ABI."""
    python_platform = sysconfig.get_platform().replace("-", "_").replace(".", "_").lower()
    arch = rust_target.split("-", 1)[0]
    if sys.platform.startswith("linux"):
        expected = f"linux_{arch}"
        if python_platform != expected:
            raise RuntimeError(f"Python platform {python_platform} differs from Rust target {rust_target}")
        # This is a local Linux wheel, not a manylinux portability claim.
        return expected
    if sys.platform == "win32":
        expected = {"x86_64": "win_amd64", "aarch64": "win_arm64", "i686": "win32"}.get(arch)
        if expected is None or python_platform != expected:
            raise RuntimeError(f"Python platform {python_platform} differs from Rust target {rust_target}")
        return expected
    if sys.platform == "darwin":
        rust_arch = {"aarch64": "arm64", "x86_64": "x86_64"}.get(arch)
        parts = python_platform.split("_")
        if rust_arch is None or not python_platform.startswith("macosx_"):
            raise RuntimeError(f"unsupported macOS Rust target {rust_target}")
        python_arch = "_".join(parts[3:])
        if python_arch not in {rust_arch, "universal2"}:
            raise RuntimeError(f"Python platform {python_platform} differs from Rust target {rust_target}")
        deployment = os.environ.get("MACOSX_DEPLOYMENT_TARGET", f"{parts[1]}.{parts[2]}")
        major, _, minor = deployment.partition(".")
        if not major.isdecimal() or not minor.isdecimal():
            raise RuntimeError(f"invalid MACOSX_DEPLOYMENT_TARGET {deployment!r}")
        minimum = (11, 0) if rust_arch == "arm64" else (10, 12)
        version = max((int(major), int(minor)), minimum)
        return f"macosx_{version[0]}_{version[1]}_{rust_arch}"
    raise RuntimeError(f"unsupported platform for provider wheel: {sys.platform}")


def _build_cdylib(workspace_root: Path, rust_target: str) -> Path:
    command = [
        os.environ.get("CARGO", "cargo"),
        "build",
        "--release",
        "--locked",
        "--target",
        rust_target,
        "-p",
        "dag-ml-data-capi",
        "--lib",
        "--message-format=json-render-diagnostics",
    ]
    result = subprocess.run(
        command,
        cwd=workspace_root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    filename = _library_filename()
    matches: list[Path] = []
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = message.get("target", {})
        if (
            message.get("reason") != "compiler-artifact"
            or target.get("name") != "dag_ml_data_capi"
        ):
            continue
        if "cdylib" not in target.get("crate_types", []):
            continue
        matches.extend(
            Path(path)
            for path in message.get("filenames", [])
            if Path(path).name == filename
        )
    if len(matches) != 1 or not matches[0].is_file():
        raise RuntimeError(
            f"cargo did not report exactly one dag-ml-data-capi cdylib named {filename}: {matches}"
        )
    return matches[0]


class CustomBuildHook(BuildHookInterface):
    """Builds one native, platform-specific wheel without source-tree copies."""

    PLUGIN_NAME = "native-provider"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name == "sdist":
            raise RuntimeError("provider source archives cannot build the native wheel; build --wheel from a Git checkout")
        if self.target_name != "wheel" or version == "editable":
            return
        project_dir = Path(self.root).resolve()
        workspace_root = project_dir.parents[3]
        if not (workspace_root / "Cargo.toml").is_file() or not (workspace_root / ".git").exists():
            raise RuntimeError("provider wheel requires the dag-ml-data Git checkout and Cargo workspace")

        rustc_verbose = _run_text(
            [os.environ.get("RUSTC", "rustc"), "-vV"], cwd=workspace_root
        )
        rust_target = next(
            (
                line.partition(":")[2].strip()
                for line in rustc_verbose.splitlines()
                if line.startswith("host:")
            ),
            "unknown",
        )
        if os.environ.get("CARGO_BUILD_TARGET", rust_target) != rust_target:
            raise RuntimeError("cross-target provider wheels are not supported by this build hook")
        platform_tag = _platform_tag(rust_target)
        source_commit = _run_text(["git", "rev-parse", "HEAD"], cwd=workspace_root)
        source_dirty = bool(
            _run_text(
                ["git", "status", "--porcelain"],
                cwd=workspace_root,
            )
        )
        if (
            source_dirty
            and os.environ.get("DAG_ML_DATA_PROVIDER_ALLOW_DIRTY_BUILD") != "1"
        ):
            raise RuntimeError(
                "refusing to build a provider wheel from tracked or untracked sources; "
                "commit the tree first (or set DAG_ML_DATA_PROVIDER_ALLOW_DIRTY_BUILD=1 "
                "for a non-release development artifact)"
            )
        library = _build_cdylib(workspace_root, rust_target)
        staging = Path(tempfile.mkdtemp(prefix="dag-ml-data-provider-wheel-"))
        self._staging = staging
        staged_library = staging / library.name
        shutil.copy2(library, staged_library)
        library_bytes = staged_library.read_bytes()
        manifest = {
            "schema_version": 1,
            "package": "dag-ml-data-provider",
            "package_version": self.metadata.version,
            "cargo_package": "dag-ml-data-capi",
            "library": library.name,
            "sha256": hashlib.sha256(library_bytes).hexdigest(),
            "size_bytes": len(library_bytes),
            "rust_target": rust_target,
            "rustc": rustc_verbose.splitlines()[0],
            "source_commit": source_commit,
            "source_dirty": source_dirty,
        }
        manifest_path = staging / "native-library.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform_tag}"
        build_data["force_include"][str(staged_library)] = (
            f"dag_ml_data_provider/.libs/{library.name}"
        )
        build_data["force_include"][str(manifest_path)] = (
            "dag_ml_data_provider/native-library.json"
        )

    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        staging = getattr(self, "_staging", None)
        if staging is not None:
            shutil.rmtree(staging)
