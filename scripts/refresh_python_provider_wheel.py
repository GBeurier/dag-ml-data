#!/usr/bin/env python3
"""Refresh the native hash after auditwheel repair, then regenerate RECORD."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="provider-wheel-repair-") as directory:
        unpacked = Path(directory)
        subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", str(wheel), "--dest", str(unpacked)],
            check=True,
        )
        roots = list(unpacked.iterdir())
        if len(roots) != 1:
            raise SystemExit("expected one unpacked provider wheel")
        root = roots[0]
        manifest_path = root / "dag_ml_data_provider/native-library.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        library = root / "dag_ml_data_provider/.libs" / manifest["library"]
        payload = library.read_bytes()
        manifest["sha256"] = hashlib.sha256(payload).hexdigest()
        manifest["size_bytes"] = len(payload)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # wheel pack regenerates RECORD for the repaired bytes and manifest.
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", str(root), "--dest-dir", str(outdir)],
            check=True,
        )


if __name__ == "__main__":
    main()
