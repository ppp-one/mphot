"""
Assemble the web demo so that `web/` is a complete, self-contained site.

The demo runs mphot in the browser under Pyodide, which installs the package
from a wheel. The wheel is built from the working tree, so the demo always
shows the code you have checked out rather than the last release.

This writes four things into `web/`, all ignored by git:

* ``dist/`` — the wheel.
* ``resources/`` — a copy of the instrument and filter curves the page fetches,
  plus the grid ingredients taken out of the wheel. Copying them means `web/`
  can be published on its own, rather than having to serve the whole repository
  so that ``../resources`` resolves.
* ``build.json`` — the wheel's name and a content hash for every file in
  ``resources/``. The page reads it first, so nothing here has to hard-code a
  version that a release would invalidate, and every other file can be served
  with a one-year immutable cache.
* ``robots.txt`` and ``sitemap.xml``. The site address comes from the
  canonical link in ``index.html``, so it is written down in one place.

Two kinds of file are pulled out of the wheel rather than shipped inside it:

* The precision grids under ``src/mphot/grids``. They are caches, they are the
  largest part of the package, and the demo rebuilds the ones it needs in a
  fraction of a second.
* The grid ingredients in ``src/mphot/datafiles/*.pkl``, which are 15 MB of the
  16 MB wheel. They go next to the page instead, so the browser can download
  them at the same time as Pyodide rather than waiting for micropip to finish.
  The page writes them back into the installed package.

The Vega spectrum is dropped altogether. Only ``vega_mag`` reads it, and the
page never calls that.

Usage:
    python web/build.py
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

WEB = Path(__file__).resolve().parent
REPO = WEB.parent
DIST = WEB / "dist"
RESOURCES = WEB / "resources"
MANIFEST = WEB / "build.json"

# The curve families the page offers. Whole directories, so adding a filter to
# the page needs no change here.
CURVE_DIRS = ("systems", "filters")

# Moved out of the wheel and fetched beside the page instead.
HOISTED_DATAFILES = "*.pkl"

# Read only by `vega_mag`, which the page never calls.
UNUSED_DATAFILES = ("vega_03_to_3_microns.csv",)


def build_wheel() -> Path:
    """Build the wheel into web/dist and return its path.

    The grid caches and the grid ingredients are stripped from the staged tree
    first. The ingredients are written into ``web/resources/datafiles`` so the
    page can fetch them on its own.
    """

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

        staged = stage / "src" / "mphot"

        for npy in (staged / "grids").glob("*.npy"):
            npy.unlink()

        hoisted = RESOURCES / "datafiles"
        hoisted.mkdir(parents=True)
        for pickle in sorted((staged / "datafiles").glob(HOISTED_DATAFILES)):
            shutil.move(str(pickle), hoisted / pickle.name)

        for name in UNUSED_DATAFILES:
            unused = staged / "datafiles" / name
            if unused.is_file():
                unused.unlink()

        # `generate_system_response` writes into this directory, so a local run
        # leaves files there. Only the Gaia curves belong to the package;
        # `gaia.py` loads them by name. Anything else is one developer's cache
        # and must not reach the deploy.
        for cached in (staged / "datafiles" / "system_responses").glob("*.csv"):
            if not cached.name.startswith("gaia_"):
                cached.unlink()

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

    return wheels[0]


def copy_curves() -> int:
    """Copy the response curves the page fetches into web/resources."""
    if RESOURCES.exists():
        shutil.rmtree(RESOURCES)

    count = 0
    for name in CURVE_DIRS:
        source = REPO / "resources" / name
        if not source.is_dir():
            raise RuntimeError(f"missing {source.relative_to(REPO)}")
        target = RESOURCES / name
        target.mkdir(parents=True)
        for csv in sorted(source.glob("*.csv")):
            shutil.copy(csv, target / csv.name)
            count += 1
    return count


def write_manifest(wheel: Path) -> dict:
    """Write build.json: the wheel's name, and a hash for every staged file.

    The page asks for each resource as ``<path>?v=<hash>``. The address of a
    file therefore changes whenever its contents do, so the file itself can be
    served with a one-year immutable cache and a stale copy can never be used.
    """

    files = {}
    for path in sorted(RESOURCES.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        files[path.relative_to(WEB).as_posix()] = digest

    manifest = {"wheel": wheel.name, "files": files}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def site_url() -> str | None:
    """Return the address this site is served from.

    The address is read from the canonical link in index.html, so it is
    written down in one place only. ``SITE_URL`` overrides it, which is useful
    when you deploy the same page somewhere else.
    """
    override = os.environ.get("SITE_URL", "").strip()
    if override.startswith("http"):
        return override.rstrip("/")

    page = (WEB / "index.html").read_text()
    found = re.search(r'<link\s+rel="canonical"\s+href="(https?://[^"]+)"', page)
    return found.group(1).rstrip("/") if found else None


def write_robots_and_sitemap() -> str | None:
    """Write robots.txt, and sitemap.xml when the address is known."""
    url = site_url()

    lines = [
        "User-agent: *",
        "Allow: /",
        "",
        "# Build output, of no use to a crawler.",
        "Disallow: /dist/",
        "Disallow: /resources/",
    ]
    if url:
        lines += ["", f"Sitemap: {url}/sitemap.xml"]
    (WEB / "robots.txt").write_text("\n".join(lines) + "\n")

    sitemap = WEB / "sitemap.xml"
    if url:
        sitemap.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            f"  <url><loc>{url}/</loc>"
            f"<lastmod>{date.today().isoformat()}</lastmod>"
            "<changefreq>monthly</changefreq></url>\n"
            "</urlset>\n"
        )
    elif sitemap.exists():
        sitemap.unlink()
    return url


def build() -> tuple[Path, int, dict]:
    """Build the wheel and stage the curves.

    ``copy_curves`` runs first because it clears ``web/resources``, and
    ``build_wheel`` then writes the grid ingredients into it.
    """

    curves = copy_curves()
    wheel = build_wheel()
    return wheel, curves, write_manifest(wheel)


if __name__ == "__main__":
    wheel, curves, manifest = build()
    url = write_robots_and_sitemap()

    staged = sum((WEB / name).stat().st_size for name in manifest["files"])
    print(f"built {wheel.relative_to(REPO)} ({wheel.stat().st_size / 1e6:.1f} MB)")
    print(
        f"staged {curves} response curves and "
        f"{len(manifest['files']) - curves} data files "
        f"in {RESOURCES.relative_to(REPO)} ({staged / 1e6:.1f} MB)"
    )
    print(f"wrote {MANIFEST.relative_to(REPO)}")
    if url:
        print(f"wrote robots.txt and sitemap.xml for {url}")
    else:
        print(
            "wrote robots.txt; no sitemap, because index.html has no "
            "absolute canonical link and SITE_URL is unset"
        )
    print("now serve web/ and open it:")
    print("    python -m http.server --directory web 8000")
