"""
Build the mphot wheel that the web demo loads.

The demo runs mphot in the browser under Pyodide, which installs the package
from a wheel. The wheel is built from the working tree, so the demo always
shows the code you have checked out rather than the last release.

The precision grids under ``src/mphot/grids`` are left out. They are caches,
they are the largest part of the package, and the demo rebuilds the ones it
needs in a fraction of a second.

The wheel's name carries the project version, so this also writes
``web/dist/wheel.json`` naming the file. The page reads that rather than
hard-coding a version that a release would invalidate.

Usage:
    python web/build.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIST = Path(__file__).resolve().parent / "dist"


def build() -> Path:
    """Build the wheel into web/dist and return its path."""
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "mphot"
        (stage / "src").mkdir(parents=True)

        for name in ("pyproject.toml", "README.md"):
            shutil.copy(REPO / name, stage / name)
        shutil.copytree(
            REPO / "src" / "mphot",
            stage / "src" / "mphot",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

        # Drop the grid caches; the demo regenerates what it needs.
        for npy in (stage / "src" / "mphot" / "grids").glob("*.npy"):
            npy.unlink()

        if DIST.exists():
            shutil.rmtree(DIST)
        DIST.mkdir(parents=True)

        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(DIST), "."],
            cwd=stage,
            check=True,
            stdout=subprocess.DEVNULL,
        )

    wheels = sorted(DIST.glob("mphot-*.whl"))
    if not wheels:
        raise RuntimeError("no wheel was produced")

    # The page reads this to find the wheel. Without it the page would have to
    # hard-code a version, and every release would break it.
    (DIST / "wheel.json").write_text(
        json.dumps({"wheel": wheels[0].name}, indent=2) + "\n"
    )
    return wheels[0]


if __name__ == "__main__":
    wheel = build()
    size = wheel.stat().st_size / 1e6
    print(f"built {wheel.relative_to(REPO)} ({size:.1f} MB)")
    print("now serve the repository root and open /web/:")
    print("    python -m http.server --directory . 8000")
