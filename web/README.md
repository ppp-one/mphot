# Web demo

This page is an exposure time calculator. It runs the `mphot` package in the
browser with [Pyodide](https://pyodide.org). No server does the work. The page
reads Gaia DR3 data from VizieR.

## Run it

```bash
python web/build.py                          # build the wheel, copy the curves
python -m http.server --directory web 8000   # serve web/
```

Then open <http://localhost:8000/>.

`build.py` makes `web/` complete on its own. It writes the wheel and a manifest
to `web/dist/`, and it copies the instrument and filter curves to
`web/resources/`. Git ignores both directories. `build.py` rebuilds them from
the source tree.

## Deploy to Netlify

`netlify.toml` in the repository root holds the settings. You can leave the
fields in the Netlify user interface empty.

| Setting | Value |
|---|---|
| Base directory | *(empty)* |
| Package directory | *(empty)* |
| Build command | `python web/build.py` |
| Publish directory | `web` |
| Functions directory | *(not used)* |

`PYTHON_VERSION` is set to 3.11. mphot needs Python 3.11 or later. The default
version in the build image is older. The deploy is about 18 MB: a 16 MB wheel,
2.7 MB of curves, and the page.

## What you can do

**Build an instrument.** Choose one of three efficiency curves and one of
thirteen filters. The efficiency curve covers the telescope, the optics and the
detector. Then set the telescope and camera values: primary and secondary
radius, plate scale, dark current, read noise, well depth, well fill, read time
and aperture radius. These are the same keys that `get_precision` takes. Three
presets fill them in. If you change a value, the preset becomes Custom.

**Read the result.** The exposure time and the binned precision stay at the top
of the page as you scroll. Beside them is the precision of one frame. Below
them are the number of frames in each bin, and how full the brightest pixel
gets. The noise chart shows one bin or one frame.

**See the optics.** The system response panel draws three curves against
wavelength: the detector with the optics, the filter, and the two multiplied
together. It redraws when you change either choice. Point at it to read the
values. It warns you when a filter passes light where the detector cannot see
it. A J filter on a silicon CCD is one example. Without the warning you would
notice this only later, as an exposure time of several hours.

**Distance** covers 0.01 pc to 1 Mpc. The slider is logarithmic. The field
beside it takes an exact value. The two stay in step.

## Everything updates as you move a control

Every control recomputes the result. One update takes about 10 ms, so the page
waits 80 ms after your last change and then runs.

Gaia mode reads a star once and keeps the answer. Without this cache the page
would query VizieR on every slider move. Press Enter in the source_id field to
load a different star.

## Search engines

A crawler will not wait for 45 MB of Pyodide, so the page carries its meaning in
plain HTML. With JavaScript switched off it still gives a title, a description,
a heading structure and about 440 words that explain what the calculator does,
how the model works and who it is for.

The head holds the title and description, a canonical link, Open Graph and
Twitter tags, a `WebApplication` record in JSON-LD, an icon and a share image.

The site address is <https://etc.withastra.io/>. It is written down once, in
the canonical link in `index.html`. `build.py` reads it from there to write
`robots.txt` and `sitemap.xml`, so the three can never disagree. To deploy the
same page elsewhere, set `SITE_URL` and it wins over the canonical link.

Cloudflare sits in front of the site and serves its own managed `robots.txt`.
It prepends that block to the file from the deploy, so the rules and the
`Sitemap:` line below still reach crawlers. Check
<https://etc.withastra.io/robots.txt> after a deploy to confirm both parts are
there.

## Why the page builds nothing in advance

A precision grid belongs to one system response. It is tempting to ship a grid
for every detector and filter pair. These are the measured times in the
browser:

| | |
|---|---|
| Build a new pair: response and both grids | 20–230 ms |
| One cached `get_precision` | 9 ms |
| One grid on disk | 1.87 MB |
| The coords array inside it | 1.40 MB, **the same for every instrument** |

Three detectors and thirteen filters make 39 pairs. That is about 18 MB of grid
data. It would roughly double the 16 MB wheel, and it would save a fifth of a
second once per pair per session. The page therefore builds each pair when you
first ask for it, and keeps it for the session.

The same holds for the package. The grids are caches, so `build.py` leaves them
out of the wheel.

One more point: every `*_coords.npy` file in `src/mphot/grids` holds the same
bytes, because `_grid_coords()` does not depend on the instrument. That is
1.4 MB repeated for each instrument, in the repository and in the released
wheel.

## How it works

`build.py` builds a wheel from the source tree. The page therefore runs the
code you have checked out, not the last release. The page then:

1. loads Pyodide with numpy, pandas and scipy,
2. installs the wheel with `micropip`,
3. copies the response curves into the Pyodide file system,
4. calls `generate_system_response`, then `get_precision` or
   `get_precision_gaia`.

The results are the same bits as mphot gives on the command line.

## Limits

**Why this works at all.** mphot needs only numpy, pandas and scipy. Pyodide
supplies all three. A package that needs a catalogue client such as
`astroquery` cannot run here. Pyodide has no build of it, and its HTTP
transport needs sockets. WebAssembly has no sockets.

**The Gaia query.** WebAssembly has no sockets, so `urllib` cannot reach the
network. `mphot.gaia` sees `sys.platform == "emscripten"` and uses the browser
to make the request instead. The archive must then allow cross-origin requests.
VizieR sends `Access-Control-Allow-Origin: *`, so it works. The ESA archive
sends no such header, so the browser blocks it. To reach ESA from a page you
need a proxy. mphot tries VizieR first, so the normal path works.

**First load** downloads about 45 MB: Pyodide with numpy, pandas and scipy,
plus the 16 MB wheel. Most of the wheel holds the two files of atmosphere and
stellar spectra in `datafiles/`. The browser caches all of it after the first
visit.
