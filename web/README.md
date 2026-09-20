# Web demo

An exposure time calculator built on `mphot`, running entirely in the browser
through [Pyodide](https://pyodide.org) — the same package as on the command
line, with no server doing the work. Gaia DR3 is queried live from VizieR.

## Run it

```bash
python web/build.py                        # build the wheel the page loads
python -m http.server --directory . 8000   # serve the repository root
```

Then open <http://localhost:8000/web/>.

Serve the repository root, not `web/`: the page reads the response curves from
`resources/`.

## What it does

**Build an instrument.** Pair any of the three efficiency curves (telescope ×
optics × detector QE) with any of the thirteen filters, then set the telescope
and camera parameters — primary and secondary radius, plate scale, dark
current, read noise, well depth, well fill, read time and aperture radius.
These are the same keys `get_precision` takes, as in the notebooks. Three
presets fill them in; editing any field switches to Custom.

**Read the result.** Exposure time and binned precision lead, with the
single-frame precision beside them, the number of frames per bin, and how full
the peak pixel gets. The noise budget can be shown per frame or per bin.

**See the optics.** The system response panel plots the detector-plus-optics
curve, the filter, and their product against wavelength, updating as soon as you
change either. Hovering reads off all three. It warns when a filter passes light
where the detector barely responds — a J filter on a silicon CCD, say — which
would otherwise show up only as an absurd exposure time further down.

**Distance** spans 0.01 pc to 1 Mpc. The slider is logarithmic; the field beside
it takes an exact value, and the two stay in step.

**See what matters.** The sensitivity panel sweeps each parameter on its own
while the rest stay put, after `examples/Sensitivity plot.ipynb`. Ranges are set
relative to the current design rather than fixed, so a panel stays informative
whether the camera has 0.2 or 110 e⁻/pix/s of dark current. All panels share one
vertical scale, so a flat curve really does mean that parameter does not matter
here.

## Why nothing is pre-computed

A precision grid is per system response, so the tempting move is to ship a grid
for every (detector, filter) pairing. Measured in the browser:

| | |
|---|---|
| Build a new pairing (response + both grids) | 20–230 ms |
| One cached `get_precision` | 9 ms |
| One grid on disk | 1.87 MB |
| …of which the coords array | 1.40 MB, **identical for every instrument** |

Three detectors × thirteen filters is 39 pairings, about 18 MB of unique grid
data — roughly doubling the 16 MB wheel to save a fifth of a second, once per
pairing per session. So the demo builds them on demand and caches them for the
session. The same holds for the package itself: the grids are caches, and
`build.py` leaves them out of the wheel entirely.

Worth noting separately: every `*_coords.npy` in `src/mphot/grids` is byte
identical, because `_grid_coords()` does not depend on the instrument. That is
1.4 MB duplicated per instrument in the repo and in the released wheel.

## How it works

`build.py` makes a wheel from the working tree, so the page always runs the code
you have checked out rather than the last release. The page then loads Pyodide
with numpy, pandas and scipy, installs the wheel with `micropip`, copies the
response curves into Pyodide's filesystem, and calls `generate_system_response`
followed by `get_precision` or `get_precision_gaia`. Results are bit-identical
to running mphot natively.

## Notes and limits

**Why it works at all.** mphot depends only on numpy, pandas and scipy, all of
which ship with Pyodide. A package needing a catalogue client such as
`astroquery` could not run here — it is not built for Pyodide, and its HTTP
transport needs sockets, which WebAssembly does not have.

**The Gaia query.** WebAssembly has no sockets, so `urllib` cannot reach the
network. `mphot.gaia` detects `sys.platform == "emscripten"` and uses the
browser's own HTTP stack instead, which means the archive must allow
cross-origin requests. VizieR sends `Access-Control-Allow-Origin: *` and works.
The ESA archive sends no such header, so a browser blocks it; reaching ESA from
a page needs a proxy. mphot tries VizieR first, so the default path is the one
that works.

**First load** pulls roughly 45 MB: Pyodide with numpy/pandas/scipy, plus a
16 MB wheel. Most of the wheel is the two atmosphere and stellar-spectra
pickles in `datafiles/`. The browser caches all of it after the first visit.

**The sweep blocks.** Pyodide runs on the main thread, so the ~1.5 s sensitivity
sweep freezes the page while it runs. Moving Pyodide into a web worker would fix
that.
