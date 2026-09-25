# Web demo

This page is an exposure time calculator. It runs the `mphot` package in the
browser with [Pyodide](https://pyodide.org). No server does the work. The page
reads Gaia DR3 data from VizieR.

## Run it

```bash
python web/build.py                          # build the wheel, copy the curves
python -m http.server --directory web 8000   # serve web/
```

`uv run web/build.py` works too. A uv environment has no pip, so `build.py`
then builds the wheel with `uv build`. Both give the same wheel.

Then open <http://localhost:8000/>.

`build.py` makes `web/` complete on its own. It writes the wheel to
`web/dist/`, copies the instrument and filter curves and the grid ingredients to
`web/resources/`, and writes `web/build.json`. Git ignores all three.
`build.py` rebuilds them from the source tree.

`build.json` names the wheel and holds a content hash for every file in
`resources/`. The page reads it first and asks for each resource as
`<path>?v=<hash>`, so an address changes whenever its contents do. That is what
lets `netlify.toml` serve everything else with a one-year immutable cache.

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
version in the build image is older. The deploy is about 19 MB: 16.5 MB of grid
ingredients, 2.7 MB of curves, a 0.3 MB wheel, and the page.

## What you can do

**Build an instrument.** Choose one of three efficiency curves and one of
thirteen filters. The efficiency curve covers the telescope, the optics and the
detector. Then set the telescope and camera values: primary and secondary
radius, plate scale, dark current, read noise, well depth, well fill, read time
and aperture radius. These are the same keys that `get_precision` takes. Three
presets fill them in. If you change a value, the preset becomes Custom.

**Set the site.** A preset also sets the site: altitude, water vapour and
seeing. The two SPECULOOS presets are at Paranal (2440 m, 2.5 mm, 1.35″). The
ETH 0.5 m preset is at 569 m, with 30 mm and 3″. You can change all three
without leaving the preset. The altitude goes to `get_precision` as `h`.

The sky model is for Paranal. For a lower site, `get_precision` multiplies the
airmass by `exp((2440 - h) / 8000)`, so the extra air counts as extra airmass.
The table stops at airmass 3, so the airmass slider stops at `3 / exp((2440 -
h) / 8000)`: 2.37 at 569 m. Past that point, the model returns NaN. A site above
2440 m would need an airmass below 1, so the altitude field stops at 2440 m.
The charts use the same converted airmass. Scintillation uses the altitude
directly.

**Bound the exposure.** The bin length is in the strip at the top. The shortest
and longest exposure are in the Exposure limits card; leave them empty to let
the ETC pick any exposure time.

**Read the result.** The exposure time and the binned precision stay at the top
of the page as you scroll, at every width. Beside them is the precision of one
frame. Below them are the number of frames in each bin, and how full the
brightest pixel gets. The noise chart shows one bin or one frame. Below it, a
table that is always open lists every value the model calculated, such as the
star and sky rates, the aperture and the collecting area. It leaves out the
values that only repeat a control. In Gaia mode it also shows the temperature
and distance from the archive, and the weights of the Gaia bands.

Below 720px the strip shrinks to one line and grows a stacked bar of the noise
budget, so you can watch the budget shift while you drag a slider further down
the page without scrolling back. Each source keeps the same colour in the strip
and in the chart. The squares of the sources are the parts of the total, because
they add in quadrature, so the stacked bar is exact rather than indicative.

The bin length lives in the strip as well, because the headline binned precision
is a precision over it. It is markup of its own rather than part of what the
strip redraws, so a drag is never cut short by the update it causes.

The two-column layout now survives down to 721px on a narrower control column.
The instrument card is the longest and the one you touch least once a preset is
chosen, so it sits last.

**See the optics, the sky and the star.** A row of three charts sits under the
strip at the top, so they are the first thing below the numbers:

* **Transmission and system response**: the detector with the optics, the
  filter, the atmosphere, and all three multiplied together;
* **Sky radiance** on a log axis, and the part of it the detector records;
* **Star** above the atmosphere on a log axis, and the part of it the detector
  records.

They redraw when you change the instrument, the sky or the star. A toggle
switches all three between the filter band and the whole model range of 0.3 to
3 µm. Point at a chart to read the values, or download all eight curves as a CSV
file. The first chart warns you when a filter passes light where the detector
cannot see it. A J filter on a silicon CCD is one example. Without the warning
you would notice this only later, as an exposure time of several hours.

The three plots line up across the row even when one legend takes two lines,
because each chart is a subgrid of the same four rows. Each chart is drawn at
the width of its column, one unit to one pixel, so the labels keep their size
at any width. Below 1024px there is room for one chart at a time, and tabs above
it pick which.

These are the curves the model integrates. They come from the two grid
ingredient files, which the page downloads for the model anyway, so the charts
add nothing to the download. The files hold 13 water vapour values and 21
airmass values. Between them the page interpolates the curves linearly, while
the model interpolates its integrals with a cubic method, so the two can differ
by about 1%. At a tabulated value they agree exactly. The star is scaled so
that its detected curve integrates to the star rate the model reports. In Gaia
mode that rate includes the calibration to the archive, so the same rule covers
both modes.

The charts draw every 1 nm point. Taking every nth point would drop the sky
emission lines and the absorption lines, which are about one point wide.

**Bring your own curves.** Under **Use your own curves** you can load a filter
transmission curve, an efficiency curve for the telescope, optics and detector,
or both. A curve is two columns: wavelength, then the fraction of light that
passes. The page works out whether the wavelengths are microns, nanometres or
ångströms, and whether the throughput is a fraction or a percentage, by scoring
each reading against the 0.3 to 3 micron window the model uses. Only one reading
of a real instrument curve lands there. It also copes with a header row, with
commas, semicolons, tabs or spaces, with a byte order mark, and with rows out of
order or repeated.

It then says what it did: how many points it kept, what span they cover, the
peak, and which units it read. It warns when a curve covers less than the whole
window, because outside the file the ETC sees no light through it. It refuses a
curve with a throughput above 100, one that does not reach the window at all,
and one over 5 MB or 200000 rows.

Your curves are kept in the browser, so they are still there next visit, and
each has a button to remove it. Nothing is uploaded anywhere; the page has no
server.

**Start again.** A **Reset all** button appears next to the Preset heading as
soon as any value differs from the one the page opened with. It puts every
control back, including the sky, the star and the exposure limits. It hides
itself again when nothing differs, so it stays out of the way until you need
it.

**Distance** covers 0.01 pc to 1 Mpc. The slider is logarithmic. The field
beside it takes an exact value. The two stay in step.

## How the controls move

Every number field has the same pair of buttons, and the arrow keys go through
the same code, so the keyboard and the buttons can never disagree. A phone gets
a stepper where the native spinner gave it none.

What one press changes is the only difference between fields. Most carry a
`step`: 0.01 for the plate scale, 0.05 m for the primary radius, 0.1 s for the
read time, 1 s for the exposure limits. Without one the browser steps by 1,
which took the plate scale from 0.35 to 1.35 in a single press. A typed value
that misses the step is still kept and still computes; the next press snaps it
onto the grid, which is what a spinner does.

Dark current, read noise, well depth and distance each cover several decades, so
no single step works at both ends. They step by a tenth of their own leading
digit instead, which is about 1% to 10% of the value and always lands on a round
number: 64000 to 65000, and 0.2 to 0.21.

Between 721px and 1023px the control column is at its narrowest, and two
steppers side by side would leave the value about 36px, which clips a well
depth. The paired fields stack there instead. Vertical space is cheap in that
band, because the results are beside the controls rather than below them.

The water vapour and temperature sliders are logarithmic, because the model's
own axes are. Water vapour is tabulated at 13 values from 0.05 mm to 30 mm, most
of them below 5 mm; the slider marks them. Temperature runs from 450 K to
36500 K with nodes 50 K apart at the bottom and 1500 K apart at the top, so a
linear slider gave the dwarf stars most people come here for about 5% of its
travel. They now get 16%. Both converters clamp and round, so the value shown
and the value computed are the same one, at the same granularity as before.

## Everything updates as you move a control

Every control recomputes the result. One update takes about 10 ms, so the page
waits 80 ms after your last change and then runs. When the sky, the star or the
instrument changes, the spectra add about 12 ms. Any other change only redraws
them, which takes 1 to 7 ms.

Gaia mode reads a star once and keeps the answer. Without this cache the page
would query VizieR on every slider move. Press Enter in the source_id field to
load a different star.

When the archive answers, the page prints what it gave: the temperature, the
parallax and the distance that follows from it. This matters, because
`get_precision_gaia` quietly substitutes 3000 K when Gaia holds no temperature
for a star, and 10 pc when it holds no usable parallax. Both substitutions used
to reach only a log that the page never showed.

The temperature comes from Gaia unless you tick **Set the temperature myself**.
A temperature passed to `get_precision_gaia` always wins over the archive, so
the page sends one only when you have asked it to, and says so when it does.

## Search engines

A crawler will not wait for 47 MB of WebAssembly, so the page carries its meaning in
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
data, more than the 16.5 MB of ingredients the page already fetches, and it
would save a fifth of a second once per pair per session. The page therefore
builds each pair when you first ask for it, and keeps it for the session. It has
to build them anyway, because an uploaded curve is a pair nobody could have
shipped in advance.

The same holds for the package. The grids are caches, so `build.py` leaves them
out of the wheel.

One more point: every `*_coords.npy` file in `src/mphot/grids` holds the same
bytes, because `_grid_coords()` does not depend on the instrument. That is
1.4 MB repeated for each instrument, in the repository and in the released
wheel.

## How it works

`build.py` builds a wheel from the source tree. The page therefore runs the
code you have checked out, not the last release. The page then:

0. reads `build.json` and starts every download at once,
1. loads Pyodide with numpy, pandas and scipy,
2. installs the wheel with `micropip`, then writes the grid ingredients back
   into the installed package,
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

**First load** downloads about 47 MB: 29.5 MB of Pyodide with numpy, pandas and
scipy from the CDN, and 17.5 MB of ours.

Ours used to wait for theirs. The page fetched Pyodide, then the packages, then
the wheel, then all sixteen curves one at a time. On a 20 Mbit link the wheel
did not start until 13.7 s, and the page was ready at 25.2 s.

Three changes remove that wait, and none of them touches the model:

* `build.py` takes the two grid ingredient files out of the wheel and stages
  them beside the page. They are 15.5 MB of a 16.2 MB wheel, and holding them
  inside it meant nothing could download until micropip had finished. The wheel
  is now 0.3 MB, and the page writes the files back into the installed package
  at a path it reads from `mphot.paths`. It also drops the Vega spectrum, which
  only `vega_mag` reads and the page never calls.
* Every download starts while the module is still being evaluated, so all of it
  runs beside Pyodide instead of after it.
* Only the two curves in use are fetched before the first result. The other
  fourteen follow in the background.

On the same link everything of ours now starts at 0.12 s and is finished at
15.7 s, comfortably before the CDN finishes at 19.0 s, and the page is ready at
22.3 s. What is left is the CDN download and the grid build, neither of which
the page can shorten.

The wheel and the staged files also go into Cache Storage, keyed by the same
versioned addresses. The browser's own cache may drop a file this size whenever
it likes; this one is explicit, so a second visit is reliably quick. A second
visit downloads only `build.json`.
