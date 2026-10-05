"""Regression checks for dependency changes and immutable release checkouts."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from publish_crates import workspace_crates


class PublishPlanTest(unittest.TestCase):
    def test_facade_follows_its_provider_dependency(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        _, crates = workspace_crates(repo)
        published: set[str] = set()
        for crate in crates:
            self.assertTrue(set(crate.internal_deps).issubset(published), crate.name)
            published.add(crate.name)
        names = [crate.name for crate in crates]
        self.assertLess(names.index("dag-ml-data-provider"), names.index("dag-ml-data"))

    def test_tooling_can_validate_a_separate_release_checkout(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        version, _ = workspace_crates(repo)
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "publish.py"
            shutil.copyfile(Path(__file__).with_name("publish_crates.py"), script)
            result = subprocess.run(
                [sys.executable, str(script), "--repo", str(repo), "--tag", f"v{version}", "--plan-only"],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn(f"7 crate(s) at {version}", result.stdout)


if __name__ == "__main__":
    unittest.main()
