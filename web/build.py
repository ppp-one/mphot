"""
Assemble the web demo so that `web/` is a complete, self-contained site.

The demo runs mphot in the browser under Pyodide, which installs the package
from a wheel. The wheel is built from the working tree, so the demo always
shows the code you have checked out rather than the last release.

This writes two things into `web/`, both ignored by git:

* ``dist/`` — the wheel, plus a ``wheel.json`` naming it. The wheel's name
  carries the project version, so the page reads the manifest rather than
  hard-coding a version that a release would invalidate.
* ``robots.txt`` and ``sitemap.xml``. The site address comes from the
  canonical link in ``index.html``, so it is written down in one place.
* ``resources/`` — a copy of the instrument and filter curves the page fetches.
  Copying them means `web/` can be published on its own, rather than having to
  serve the whole repository so that ``../resources`` resolves.

The precision grids under ``src/mphot/grids`` are left out of the wheel. They
are caches, they are the largest part of the package, and the demo rebuilds the
ones it needs in a fraction of a second.

Usage:
    python web/build.py
"""

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

# The curve families the page offers. Whole directories, so adding a filter to
# the page needs no change here.
CURVE_DIRS = ("systems", "filters")


def build_wheel() -> Path:
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

    (DIST / "wheel.json").write_text(
        json.dumps({"wheel": wheels[0].name}, indent=2) + "\n"
    )
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


def build() -> tuple[Path, int]:
    """Build the wheel and stage the curves. Returns the wheel and curve count."""
    return build_wheel(), copy_curves()


if __name__ == "__main__":
    wheel, curves = build()
    url = write_robots_and_sitemap()
    size = wheel.stat().st_size / 1e6
    print(f"built {wheel.relative_to(REPO)} ({size:.1f} MB)")
    print(f"staged {curves} response curves in {RESOURCES.relative_to(REPO)}")
    if url:
        print(f"wrote robots.txt and sitemap.xml for {url}")
    else:
        print(
            "wrote robots.txt; no sitemap, because index.html has no "
            "absolute canonical link and SITE_URL is unset"
        )
    print("now serve web/ and open it:")
    print("    python -m http.server --directory web 8000")
