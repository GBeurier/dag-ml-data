#!/usr/bin/env python3
"""Validate and runtime-smoke the autonomous ctypes provider wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import venv
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256_record(payload: bytes) -> str:
    return (
        "sha256="
        + base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
        .rstrip(b"=")
        .decode()
    )


def validate_wheel(wheel_path: Path) -> dict[str, Any]:
    require(wheel_path.is_file(), f"wheel does not exist: {wheel_path}")
    require(
        "-none-any.whl" not in wheel_path.name,
        "provider wheel must be platform-specific",
    )
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        manifest_name = "dag_ml_data_provider/native-library.json"
        require(
            manifest_name in names, f"{wheel_path}: missing native provenance manifest"
        )
        manifest = json.loads(wheel.read(manifest_name))
        library_name = f"dag_ml_data_provider/.libs/{manifest['library']}"
        require(
            library_name in names,
            f"{wheel_path}: missing bundled library {library_name}",
        )
        library = wheel.read(library_name)
        require(
            len(library) == manifest["size_bytes"],
            f"{wheel_path}: native size mismatch",
        )
        require(
            hashlib.sha256(library).hexdigest() == manifest["sha256"],
            f"{wheel_path}: native provenance SHA-256 mismatch",
        )
        require(
            manifest["schema_version"] == 1, f"{wheel_path}: wrong provenance version"
        )
        require(
            manifest["package"] == "dag-ml-data-provider",
            f"{wheel_path}: wrong package provenance",
        )
        require(
            manifest["cargo_package"] == "dag-ml-data-capi",
            f"{wheel_path}: wrong Cargo provenance",
        )
        require(
            re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]) is not None,
            f"{wheel_path}: source_commit is not a full Git SHA",
        )
        require(
            manifest["source_dirty"] is False,
            f"{wheel_path}: release provenance is dirty",
        )

        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        require(
            len(metadata_names) == len(wheel_names) == len(record_names) == 1,
            "invalid dist-info layout",
        )
        metadata = Parser().parsestr(wheel.read(metadata_names[0]).decode())
        wheel_metadata = Parser().parsestr(wheel.read(wheel_names[0]).decode())
        require(
            metadata["Name"] == "dag-ml-data-provider",
            f"{wheel_path}: metadata name mismatch",
        )
        require(
            metadata["Version"] == manifest["package_version"],
            f"{wheel_path}: version mismatch",
        )
        require(
            metadata["Requires-Python"] == ">=3.11",
            f"{wheel_path}: Requires-Python mismatch",
        )
        license_expression = metadata["License-Expression"] or metadata["License"]
        require(
            license_expression is not None
            and license_expression.casefold()
            == "CeCILL-2.1 OR AGPL-3.0-or-later".casefold(),
            f"{wheel_path}: license expression mismatch",
        )
        require(
            any(name.endswith(".dist-info/licenses/LICENSE") for name in names),
            f"{wheel_path}: packaged license missing",
        )
        require(
            wheel_metadata["Root-Is-Purelib"] == "false",
            f"{wheel_path}: wheel marked pure",
        )
        tags = wheel_metadata.get_all("Tag") or []
        require(
            tags and all(not tag.endswith("-any") for tag in tags),
            f"{wheel_path}: non-platform tag",
        )

        rows = list(csv.reader(io.StringIO(wheel.read(record_names[0]).decode())))
        record = {row[0]: row[1:] for row in rows}
        require(
            set(record) == set(names),
            f"{wheel_path}: RECORD does not cover every member exactly once",
        )
        for name in names:
            digest, size = record[name]
            if name == record_names[0]:
                require(
                    digest == size == "",
                    f"{wheel_path}: RECORD self-entry must be unhashed",
                )
                continue
            payload = wheel.read(name)
            require(
                digest == sha256_record(payload),
                f"{wheel_path}: RECORD digest mismatch for {name}",
            )
            require(
                size == str(len(payload)),
                f"{wheel_path}: RECORD size mismatch for {name}",
            )
    return manifest


def venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_n4d_fixture(path: Path) -> None:
    """Write the smallest deterministic `.n4d` v1 package fixture.

    This test-only encoder intentionally covers one dense buffer. Product code
    reads and validates the bytes in Rust; it does not implement the format in
    Python.
    """
    payload = bytearray(b"N4DF")
    payload.extend(struct.pack("<II", 1, 1))

    def write_string(value: str) -> None:
        encoded = value.encode("utf-8")
        payload.extend(struct.pack("<I", len(encoded)))
        payload.extend(encoded)

    write_string("x")
    write_string("tabular_numeric")
    payload.extend(struct.pack("<IIB", 4, 2, 0))
    for value in ("f0", "f1"):
        write_string(value)
    for value in (
        "obs.S001.base",
        "obs.S001.rep1",
        "obs.S001.aug0",
        "obs.S002.base",
    ):
        write_string(value)
    payload.extend(struct.pack("<8d", 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0))
    payload.extend(hashlib.sha256(payload).digest())
    path.write_bytes(payload)


def smoke_package_provider(
    python: Path,
    temp: Path,
    repo: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    """Run the package-resource provider using no sibling checkout/import."""
    fixture_package = temp / "fixture_dataset"
    fixture_package.mkdir()
    (fixture_package / "__init__.py").write_text("", encoding="utf-8")
    (fixture_package / "envelope.json").write_bytes(
        (
            repo
            / "examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        ).read_bytes()
    )
    (fixture_package / "request.json").write_bytes(
        (
            repo
            / "examples/fixtures/oof_campaign/materialization_request_model_base_x.json"
        ).read_bytes()
    )
    write_n4d_fixture(fixture_package / "features.n4d")
    site_packages = Path(
        run_checked(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            cwd=temp,
            env=env,
        )
    )
    installed_fixture = site_packages / fixture_package.name
    shutil.copytree(fixture_package, installed_fixture)
    shutil.rmtree(fixture_package)
    probe_cwd = temp / "package-provider-probe"
    probe_cwd.mkdir()
    probe = """
import importlib.resources as resources
import json
from pathlib import Path
import sys
import fixture_dataset
from dag_ml_data_provider import PackageProvider

request = json.loads(resources.files("fixture_dataset").joinpath("request.json").read_text())
with PackageProvider.from_package_resources(
    "fixture_dataset",
    envelope_resource="envelope.json",
    feature_store_resource="features.n4d",
) as provider:
    manifests = provider.feature_buffer_manifests()
    data_handle = provider.materialize(request)
    view_handle = provider.make_view(
        data_handle,
        {"sample_ids": ["S002", "S001"], "include_augmented": False},
    )
    identity = provider.view_identity(view_handle)
    features = provider.feature_values(view_handle, "x")
    provider.release(data_handle)
    released_invalidates = False
    try:
        provider.feature_values(view_handle, "x")
    except RuntimeError:
        released_invalidates = True
destroyed = not bool(provider._vtable.user_data)
print(json.dumps({
    "package_origin": str(Path(fixture_dataset.__file__).resolve()),
    "package_under_prefix": Path(sys.prefix).resolve() in Path(fixture_dataset.__file__).resolve().parents,
    "manifest_feature_set": manifests[0]["feature_set_id"],
    "identity": [row["observation_id"] for row in identity],
    "features": [row["features"] for row in features],
    "released_invalidates": released_invalidates,
    "destroyed": destroyed,
}, sort_keys=True))
"""
    result = json.loads(
        run_checked([str(python), "-c", probe], cwd=probe_cwd, env=env)
    )
    require(
        result["package_under_prefix"] is True,
        f"fixture package was not imported below the fresh venv: {result['package_origin']}",
    )
    require(
        str(fixture_package) not in result["package_origin"],
        "fixture package remained importable from its source directory",
    )
    require(result["manifest_feature_set"] == "x", "package store was not loaded")
    require(
        result["identity"]
        == ["obs.S002.base", "obs.S001.base", "obs.S001.rep1"],
        "package provider did not preserve requested identity order",
    )
    require(
        result["features"]
        == [{"f0": 4.0, "f1": 40.0}, {"f0": 1.0, "f1": 10.0}, {"f0": 2.0, "f1": 20.0}],
        "package provider returned unexpected feature values",
    )
    require(result["released_invalidates"] is True, "released package view stayed live")
    require(result["destroyed"] is True, "package provider was not destroyed")
    return result


def smoke_installed_wheel(wheel_path: Path, repo: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="dag-ml-data-provider-wheel-smoke-"
    ) as temp_name:
        temp = Path(temp_name)
        environment = temp / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = venv_python(environment)
        clean_env = os.environ.copy()
        for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            clean_env.pop(key, None)
        # Exercise automatic package discovery without the explicit legacy override.
        clean_env.pop("DAG_ML_DATA_CAPI_LIB", None)
        run_checked(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel_path)],
            cwd=temp,
            env=clean_env,
        )

        package_report = smoke_package_provider(
            python,
            temp,
            repo,
            clean_env,
        )

        smoke = run_checked(
            [
                str(python),
                str(repo / "examples/python/provider_contract_spike.py"),
                "--envelope",
                str(
                    repo
                    / "examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
                ),
                "--request",
                str(
                    repo
                    / "examples/fixtures/oof_campaign/materialization_request_model_base_x.json"
                ),
            ],
            cwd=temp,
            env=clean_env,
        )
        report = json.loads(smoke)
        require(
            set(report["native_branch_modes"])
            == {"by_source", "by_metadata", "by_tag", "by_filter"},
            "installed runtime did not execute every native branch view",
        )
        require(
            report["lifecycle"]["released_invalidates"] is True,
            "release lifecycle was not proved",
        )
        require(
            report["lifecycle"]["destroyed"] is True,
            "destroy lifecycle was not proved",
        )

        package_root = Path(
            run_checked(
                [
                    str(python),
                    "-c",
                    "import pathlib, dag_ml_data_provider as p; print(pathlib.Path(p.__file__).parent)",
                ],
                cwd=temp,
                env=clean_env,
            )
        )
        # macOS exposes /var through /private/var; compare canonical paths.
        require(
            environment.resolve() in package_root.resolve().parents,
            "provider import did not come from the fresh venv",
        )
        library_name = json.loads((package_root / "native-library.json").read_text())[
            "library"
        ]
        library_path = package_root / ".libs" / library_name
        original = library_path.read_bytes()

        missing_path = library_path.with_suffix(library_path.suffix + ".missing")
        library_path.rename(missing_path)
        missing_probe = run_checked(
            [
                str(python),
                "-c",
                (
                    "from dag_ml_data_provider import NativeLibraryNotFoundError, find_capi_library; "
                    "\ntry: find_capi_library()"
                    "\nexcept NativeLibraryNotFoundError: print('typed-missing')"
                    "\nelse: raise SystemExit('missing library was accepted')"
                ),
            ],
            cwd=temp,
            env=clean_env,
        )
        missing_path.rename(library_path)
        require(
            missing_probe == "typed-missing",
            "missing library did not raise the typed error",
        )

        library_path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        integrity_probe = run_checked(
            [
                str(python),
                "-c",
                (
                    "from dag_ml_data_provider import NativeLibraryIntegrityError, find_capi_library; "
                    "\ntry: find_capi_library()"
                    "\nexcept NativeLibraryIntegrityError: print('typed-integrity')"
                    "\nelse: raise SystemExit('altered library was accepted')"
                ),
            ],
            cwd=temp,
            env=clean_env,
        )
        library_path.write_bytes(original)
        require(
            integrity_probe == "typed-integrity",
            "altered library did not raise the typed error",
        )
        report["package_provider"] = package_report
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    repo = Path(__file__).resolve().parents[1]
    manifest = validate_wheel(wheel)
    report = None if args.metadata_only else smoke_installed_wheel(wheel, repo)
    print(
        json.dumps(
            {
                "wheel": wheel.name,
                "native_sha256": manifest["sha256"],
                "rust_target": manifest["rust_target"],
                "runtime": report,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
