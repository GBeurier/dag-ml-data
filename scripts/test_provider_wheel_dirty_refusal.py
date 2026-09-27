#!/usr/bin/env python3
"""Prove that an untracked package source makes the wheel build fail closed."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    package_dir = repo / "crates/dag-ml-data-capi/bindings/python"
    probe = package_dir / "dag_ml_data_provider/_untracked_build_probe.py"
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise SystemExit(
            "dirty-refusal test requires a clean checkout before creating its probe:\n"
            + status
        )
    if probe.exists():
        raise SystemExit(f"refusing to overwrite existing probe path: {probe}")

    env = os.environ.copy()
    env.pop("DAG_ML_DATA_PROVIDER_ALLOW_DIRTY_BUILD", None)
    try:
        probe.write_text("UNTRACKED_BUILD_PROBE = True\n", encoding="utf-8")
        with tempfile.TemporaryDirectory(
            prefix="dag-ml-data-provider-dirty-refusal-"
        ) as out:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--outdir",
                    out,
                    str(package_dir),
                ],
                cwd=repo,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
        output = result.stdout + result.stderr
        if result.returncode == 0:
            raise SystemExit("wheel build accepted an untracked package source")
        expected = (
            "refusing to build a provider wheel from tracked or untracked sources"
        )
        if expected not in output:
            raise SystemExit(
                "wheel build failed for an unexpected reason; missing fail-closed diagnostic:\n"
                + output
            )
    finally:
        probe.unlink(missing_ok=True)

    final_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if final_status:
        raise SystemExit(
            "dirty-refusal test did not restore a clean checkout:\n" + final_status
        )
    print(
        "provider wheel refused an untracked package source and restored the clean checkout"
    )


if __name__ == "__main__":
    main()
