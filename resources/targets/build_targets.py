"""
Build the data files that `mphot.targets` and `mphot.extended` read.

Run it from the repository root. It needs network access, and it takes about
half an hour, because it cuts out three survey images for each nebula:

    python resources/targets/build_targets.py

It writes:

* ``src/mphot/datafiles/targets/openngc.csv`` — the NGC, IC and addendum
  objects of OpenNGC, trimmed to the columns mphot uses. OpenNGC is released
  under CC-BY-SA-4.0 (https://github.com/mattiaverga/OpenNGC), and so is this
  derived file.
* ``src/mphot/datafiles/targets/nebula_brightness.csv`` — the mean and peak
  surface brightness of each nebula in that file, in B, V and R.
* ``src/mphot/datafiles/flux_calibration/{bessell_b,bessell_v,bessell_r,2mass_j,2mass_h,2mass_ks}.csv``
  — the catalogue bands, as photon-counting response curves.

Sources:

* OpenNGC, pinned to one commit so that the build can be repeated.
* Nebulae: the B, V and R images of the CTA-FRAM Sky Survey (Janecek et al.
  2019, EPJ Web Conf. 197, 02008), read through the CDS hips2fits service.
  Gaia DR3 synthetic photometry (Gaia Collaboration, Montegriffo et al. 2023,
  VizieR I/360) sets the zero point of each image and gives the stars that are
  removed from it.
* Filter curves: the SVO Filter Profile Service (Bessell 1990 B, V and R;
  Cohen et al. 2003 2MASS J, H and Ks).
"""

import io
import math
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from mphot.gaia import _tap_sync_csv  # noqa: E402

TARGETS_DIR = REPO / "src" / "mphot" / "datafiles" / "targets"
FLUX_CALIBRATION_DIR = REPO / "src" / "mphot" / "datafiles" / "flux_calibration"

OPENNGC_COMMIT = "da90466031b0372c896588b85be6016c617e205b"
OPENNGC_URL = (
    "https://raw.githubusercontent.com/mattiaverga/OpenNGC/"
    f"{OPENNGC_COMMIT}/database_files/{{}}"
)

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
SVO_FPS = "http://svo2.cab.inta-csic.es/theory/fps/getdata.php"

# SVO name, output file, and whether the curve is for an energy-counting
# detector. SVO lists Bessell's curves as energy counting (DetectorType 0) and
# the 2MASS curves as photon counting (DetectorType 1).
FILTERS = (
    ("Generic/Bessell.B", "bessell_b.csv", True),
    ("Generic/Bessell.V", "bessell_v.csv", True),
    ("Generic/Bessell.R", "bessell_r.csv", True),
    ("2MASS/2MASS.J", "2mass_j.csv", False),
    ("2MASS/2MASS.H", "2mass_h.csv", False),
    ("2MASS/2MASS.Ks", "2mass_ks.csv", False),
)

GALAXY_TYPES = {"G", "GPair", "GTrpl", "GGroup"}
NEBULA_TYPES = {"PN", "HII", "EmN", "Cl+N", "Neb", "SNR", "RfN"}

FRAM_HIPS = "fzu.cz/P/CTA-FRAM/survey/{}"
FRAM_BANDS = ("B", "V", "R")
FRAM_PIXEL = 14.0
"""Pixel of the CTA-FRAM cameras [arcsec]. The cutouts use the same pixel."""

FRAM_FWHM = 36.0
"""Typical FWHM of a star in the CTA-FRAM images [arcsec], in all three bands."""

STARS_TABLE = "I/360/syntphot"
"""Gaia DR3 synthetic photometry, with Johnson-Cousins B, V and R."""

STAR_LIMIT = 16.0
"""Stars brighter than this, in the band of the image, are subtracted [mag]."""

BRIGHT_TABLE = "I/239/hip_main"
"""Hipparcos. The Gaia synthetic photometry has no values for some of the
brightest stars, for example iota Ori A (V = 2.8) near M42."""

BRIGHT_LIMIT = 7.0
"""Hipparcos stars brighter than this that have no Gaia value are masked [V]."""

PEAK_PIXELS = 10
"""The peak is the median of the 10 brightest pixels. This is approximately the
area of two star images, so one star that is not fully removed cannot set the
peak."""

MIN_FIELD_ARCMIN = 40.0
"""Smallest cutout. It holds enough stars to set the zero point."""


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "mphot build"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def sexagesimal_to_degrees(value: str, hours: bool) -> float:
    if not isinstance(value, str) or not value.strip():
        return float("nan")
    sign = -1.0 if value.strip().startswith("-") else 1.0
    parts = [abs(float(p)) for p in value.strip().lstrip("+-").split(":")]
    degrees = parts[0] + parts[1] / 60 + parts[2] / 3600
    return sign * degrees * (15.0 if hours else 1.0)


def build_openngc() -> pd.DataFrame:
    frames = [
        pd.read_csv(io.BytesIO(fetch(OPENNGC_URL.format(f))), sep=";", dtype=str)
        for f in ("NGC.csv", "addendum.csv")
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["Type"] != "NonEx"]

    def number(column):
        return pd.to_numeric(raw[column], errors="coerce")

    table = pd.DataFrame(
        {
            "name": raw["Name"],
            "type": raw["Type"],
            "ra": [sexagesimal_to_degrees(v, hours=True) for v in raw["RA"]],
            "dec": [sexagesimal_to_degrees(v, hours=False) for v in raw["Dec"]],
            "major_axis": number("MajAx"),  # arcmin
            "minor_axis": number("MinAx"),  # arcmin
            "position_angle": number("PosAng"),  # deg, north through east
            "b": number("B-Mag"),
            "v": number("V-Mag"),
            "j": number("J-Mag"),
            "h": number("H-Mag"),
            "k": number("K-Mag"),
            "radial_velocity": number("RadVel"),  # km/s
            "messier": number("M").astype("Int64"),
            # A duplicate entry names its main object in these two columns.
            "ngc": raw["NGC"],
            "ic": raw["IC"],
            "common_names": raw["Common names"],
        }
    )

    # OpenNGC takes J, H and K from SIMBAD. For a nebula they belong to a star
    # in it: for M57, J = 16.4 is the central star. Only galaxies keep them.
    not_galaxy = ~table["type"].isin(GALAXY_TYPES)
    table.loc[not_galaxy, ["j", "h", "k"]] = np.nan
    # B and V of a nebula often include its stars too: for M42, V = 4.0 is
    # mostly the Trapezium. Nebulae get their brightness from CTA-FRAM instead.
    table.loc[table["type"].isin(NEBULA_TYPES), ["b", "v"]] = np.nan
    return table.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Surface brightness of nebulae from the CTA-FRAM images


def read_fits_image(data: bytes) -> np.ndarray:
    """Read the one image of a simple FITS file, without astropy."""
    header = {}
    offset = 0
    while True:
        card = data[offset : offset + 80].decode("ascii", "replace")
        offset += 80
        if card.startswith("END"):
            break
        if card[8:10] == "= ":
            header[card[:8].strip()] = card[10:].split("/")[0].strip()
    start = ((offset + 2879) // 2880) * 2880
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    dtype = {"-32": ">f4", "-64": ">f8"}[header["BITPIX"]]
    image = np.frombuffer(data[start:], dtype=dtype, count=nx * ny)
    return image.reshape(ny, nx).astype(float)


def fram_cutout(band: str, ra: float, dec: float, n: int) -> np.ndarray:
    """A square CTA-FRAM image, north up and east left."""
    query = urllib.parse.urlencode(
        {
            "hips": FRAM_HIPS.format(band),
            "width": n,
            "height": n,
            "fov": n * FRAM_PIXEL / 3600,
            "ra": ra,
            "dec": dec,
            "projection": "TAN",
            "format": "fits",
        }
    )
    # FITS rows go from south to north.
    return read_fits_image(fetch(f"{HIPS2FITS}?{query}"))[::-1]


def gnomonic(ra, dec, ra0: float, dec0: float):
    """Standard coordinates [deg]: xi to the east, eta to the north."""
    ra, dec, ra0, dec0 = map(np.radians, (ra, dec, ra0, dec0))
    cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cos_c
    eta = (
        np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)
    ) / cos_c
    return np.degrees(xi), np.degrees(eta)


def field_stars(ra: float, dec: float, radius: float) -> pd.DataFrame:
    """Gaia DR3 stars with synthetic B, V or R brighter than STAR_LIMIT."""
    # The table also has Sloan "rmag". ADQL ignores case unless a name is
    # quoted, so the Johnson-Cousins names are quoted.
    rows = _tap_sync_csv(
        VIZIER_TAP,
        f'SELECT RA_ICRS, DE_ICRS, "Bmag", "Vmag", "Rmag" FROM "{STARS_TABLE}" '
        f"WHERE 1 = CONTAINS(POINT('ICRS', RA_ICRS, DE_ICRS), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius})) "
        f'AND ("Bmag" < {STAR_LIMIT} OR "Vmag" < {STAR_LIMIT} '
        f'OR "Rmag" < {STAR_LIMIT})',
        timeout=600,
    )
    columns = ["RA_ICRS", "DE_ICRS", "Bmag", "Vmag", "Rmag"]
    return pd.DataFrame(rows, columns=columns).apply(pd.to_numeric, errors="coerce")


def bright_stars(ra: float, dec: float, radius: float) -> pd.DataFrame:
    """Hipparcos stars brighter than BRIGHT_LIMIT, which Gaia can miss."""
    rows = _tap_sync_csv(
        VIZIER_TAP,
        f'SELECT RAICRS, DEICRS, Vmag FROM "{BRIGHT_TABLE}" '
        f"WHERE 1 = CONTAINS(POINT('ICRS', RAICRS, DEICRS), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius})) AND Vmag < {BRIGHT_LIMIT}",
        timeout=300,
    )
    columns = ["RAICRS", "DEICRS", "Vmag"]
    return pd.DataFrame(rows, columns=columns).apply(pd.to_numeric, errors="coerce")


def _gaussian(xy, amplitude, x0, y0, sigma, background):
    x, y = xy
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return amplitude * np.exp(-r2 / (2 * sigma**2)) + background


def calibrate(image, x, y, mag, keep_out) -> dict | None:
    """
    Fit isolated stars with a Gaussian.

    Returns the star width, the shift of the image against the catalogue, and
    two zero points. The Gaussian zero point turns a catalogue magnitude into
    the amplitude of the Gaussian that removes the star. The aperture zero
    point (4 pixels) turns the image into surface brightness.
    """

    n = image.shape[0]
    yy, xx = np.indices(image.shape)
    fits = []
    for lo, hi in ((8.0, 12.0), (7.0, 13.0)):
        for i in np.flatnonzero((mag > lo) & (mag < hi) & ~keep_out):
            col, row = int(round(x[i])), int(round(y[i]))
            if not (6 <= row < n - 6 and 6 <= col < n - 6):
                continue
            near = np.hypot(x - x[i], y - y[i])
            if ((near > 0) & (near < 4) & (mag < mag[i] + 4)).any():
                continue
            box = (slice(row - 4, row + 5), slice(col - 4, col + 5))
            data = image[box]
            try:
                p, _ = curve_fit(
                    _gaussian,
                    (xx[box].ravel(), yy[box].ravel()),
                    data.ravel(),
                    p0=[data.max(), x[i], y[i], 1.1, np.median(data)],
                    maxfev=4000,
                )
            except (RuntimeError, ValueError):
                continue
            amplitude, x0, y0, sigma = p[0], p[1], p[2], abs(p[3])
            if (
                amplitude <= 0
                or not 0.6 < sigma < 2.5
                or math.hypot(x0 - x[i], y0 - y[i]) > 1.5
            ):
                continue
            r = np.hypot(xx - x0, yy - y0)
            aperture = (image[r < 4] - np.median(image[(r > 6) & (r < 9)])).sum()
            fits.append(
                (
                    sigma,
                    x0 - x[i],
                    y0 - y[i],
                    mag[i] + 2.5 * math.log10(2 * math.pi * sigma**2 * amplitude),
                    mag[i] + 2.5 * math.log10(aperture) if aperture > 0 else np.nan,
                )
            )
        if len(fits) >= 5:
            break
    if len(fits) < 3:
        return None
    f = np.array(fits)
    return {
        "sigma": float(np.median(f[:, 0])),
        "dx": float(np.median(f[:, 1])),
        "dy": float(np.median(f[:, 2])),
        "zp_gauss": float(np.median(f[:, 3])),
        "zp_aperture": float(np.nanmedian(f[:, 4])),
    }


def star_model(shape, x, y, flux, sigma) -> np.ndarray:
    """The image of the stars, as Gaussians of the measured width."""
    model = np.zeros(shape)
    n = shape[0]
    for xi, yi, fi in zip(x, y, flux, strict=True):
        half = 15 if fi > 1e4 else 6
        r0, c0 = int(round(yi)), int(round(xi))
        rows = slice(max(r0 - half, 0), min(r0 + half + 1, n))
        cols = slice(max(c0 - half, 0), min(c0 + half + 1, n))
        if rows.start >= rows.stop or cols.start >= cols.stop:
            continue
        yy, xx = np.mgrid[rows, cols]
        model[rows, cols] += (
            fi
            / (2 * math.pi * sigma**2)
            * np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2 * sigma**2))
        )
    return model


def clipped_mean(values: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation after removing outliers beyond 5 sigma."""
    values = values[np.isfinite(values)]
    for _ in range(3):
        centre = np.median(values)
        spread = 1.4826 * np.median(np.abs(values - centre))
        if spread == 0:
            break
        values = values[np.abs(values - centre) < 5 * spread]
    return float(values.mean()), float(values.std())


def fram_brightness(row: pd.Series) -> dict | None:
    """
    Mean and peak surface brightness of a nebula in B, V and R.

    The steps for each band:

    1. Cut out the CTA-FRAM image, at least 40' wide, with sky around the
       nebula.
    2. Fit isolated Gaia stars. This gives the star width and the zero point of
       this image.
    3. Subtract all Gaia stars brighter than STAR_LIMIT. Where a star is so
       bright that the image can be clipped, mask the pixels instead.
    4. Subtract the sky: the mean of a ring around the nebula. The mean, not
       the median, also removes the light of fainter stars, which is the same
       inside and outside on average.
    5. The mean is the mean in the catalogue ellipse. A nebula less than 2.4'
       across (4 star widths) is measured in an aperture grown by 1.5 star
       widths, and its total is divided by the catalogue area. It has no peak.
    6. The peak is the median, in each band, of the 10 brightest pixels in R
       that are at least 2 pixels from a mask.

    Returns:
        dict: "{band}_mean" and "{band}_peak" [mag/arcsec2, Vega]. A band
            without a significant detection is NaN. None if the nebula has no
            size.
    """

    a = row["major_axis"]
    b = row["minor_axis"] if np.isfinite(row["minor_axis"]) else a
    if not np.isfinite(a):
        return None

    fwhm = FRAM_FWHM / 60  # arcmin
    resolved = math.sqrt(a * b) >= 4 * fwhm
    grow = 0.0 if resolved else 1.5 * fwhm
    reach = a / 2 + grow  # largest distance of the aperture from the centre
    ring_in = reach + 2 * fwhm
    ring_out = ring_in + max(5.0, 0.3 * ring_in)
    n = int(math.ceil(max(2 * ring_out, MIN_FIELD_ARCMIN) * 60 / FRAM_PIXEL)) | 1

    centre = (n - 1) / 2
    yy, xx = np.indices((n, n))
    east = (centre - xx) * FRAM_PIXEL / 60  # arcmin
    north = (centre - yy) * FRAM_PIXEL / 60
    radius = np.hypot(east, north)
    pa = row["position_angle"]
    if np.isfinite(pa) and resolved:
        pa = math.radians(pa)
        u = east * math.sin(pa) + north * math.cos(pa)
        v = east * math.cos(pa) - north * math.sin(pa)
        inside = (u / (a / 2)) ** 2 + (v / (b / 2)) ** 2 <= 1
    else:
        # OpenNGC has no position angle for most nebulae, so the ellipse
        # becomes a circle of the same area.
        inside = radius <= math.sqrt(a * b) / 2 + grow
    ring = (radius > ring_in) & (radius < ring_out)
    area_pixels = math.pi / 4 * a * b * 3600 / FRAM_PIXEL**2

    stars = field_stars(row["ra"], row["dec"], n * FRAM_PIXEL / 3600 / math.sqrt(2))
    xi, eta = gnomonic(stars["RA_ICRS"], stars["DE_ICRS"], row["ra"], row["dec"])
    x0 = (centre - xi * 3600 / FRAM_PIXEL).to_numpy()
    y0 = (centre - eta * 3600 / FRAM_PIXEL).to_numpy()
    keep_out = np.hypot(x0 - centre, y0 - centre) * FRAM_PIXEL / 60 < reach + fwhm

    # Bright stars that Gaia has no B, V and R for are masked. The radius grows
    # with the brightness: 3 pixels at V = 7, 11 pixels at V = 3.
    missing = np.zeros((n, n), dtype=bool)
    hip = bright_stars(row["ra"], row["dec"], n * FRAM_PIXEL / 3600 / math.sqrt(2))
    hip_xi, hip_eta = gnomonic(hip["RAICRS"], hip["DEICRS"], row["ra"], row["dec"])
    for xi_h, eta_h, v_h in zip(hip_xi, hip_eta, hip["Vmag"], strict=True):
        hx = centre - xi_h * 3600 / FRAM_PIXEL
        hy = centre - eta_h * 3600 / FRAM_PIXEL
        match = (np.hypot(x0 - hx, y0 - hy) < 20 / FRAM_PIXEL) & (
            np.abs(stars["Vmag"].to_numpy() - v_h) < 1.5
        )
        if not match.any():
            missing |= np.hypot(xx - hx, yy - hy) < 3 + 2 * (BRIGHT_LIMIT - v_h)

    def surface_brightness(counts, cal):
        return cal["zp_aperture"] - 2.5 * math.log10(counts / FRAM_PIXEL**2)

    result = {}
    measured = {}
    for band in FRAM_BANDS:
        image = fram_cutout(band, row["ra"], row["dec"], n)
        mag = stars[f"{band}mag"].to_numpy()
        cal = calibrate(image, x0, y0, np.nan_to_num(mag, nan=99.0), keep_out)
        result[f"{band.lower()}_mean"] = result[f"{band.lower()}_peak"] = np.nan
        if cal is None:
            continue

        bright = np.isfinite(mag) & (mag < STAR_LIMIT)
        x, y = x0[bright] + cal["dx"], y0[bright] + cal["dy"]
        flux = 10 ** (0.4 * (cal["zp_gauss"] - mag[bright]))
        model = star_model(image.shape, x, y, flux, cal["sigma"])
        clean = image - model

        # Pixels where a star can be clipped. A small nebula keeps its own
        # pixels, so that the subtraction of its central star does not mask it.
        masked = binary_dilation(model > 0.3 * np.nanmax(image), iterations=2)
        masked |= missing
        if not resolved:
            masked &= radius > reach
        clean[masked] = np.nan

        sky, sky_spread = clipped_mean(clean[ring])
        net = clean - sky
        pixels = inside & np.isfinite(net)
        if pixels.sum() < 3:
            continue
        if resolved:
            level = float(net[pixels].mean())
            count = pixels.sum()
        else:
            level = float(np.nan_to_num(net[inside]).sum()) / area_pixels
            count = area_pixels
        beam = 2 * math.pi * cal["sigma"] ** 2  # pixels in one star image
        noise = sky_spread * math.sqrt(beam / max(count, beam) + beam / ring.sum())
        if not level > 3 * noise:
            continue

        result[f"{band.lower()}_mean"] = surface_brightness(level, cal)
        measured[band] = (net, cal, level)

    # The peak is the same pixels in every band, so that its colours belong to
    # one place. They are the brightest pixels in R, where an emission nebula is
    # brightest and the hot stars in it are faintest, else in V or B.
    if resolved and measured:
        # Next to a masked star, some of its light remains, and it would make
        # the peak too blue. So the peak keeps 2 pixels away from the masks.
        usable = inside.copy()
        for net, _, _ in measured.values():
            usable &= binary_erosion(np.isfinite(net), iterations=2, border_value=1)
        reference = next(b for b in ("R", "V", "B") if b in measured)
        if usable.sum() >= 3 * PEAK_PIXELS:
            values = np.where(usable, measured[reference][0], -np.inf)
            brightest = np.argsort(values, axis=None)[-PEAK_PIXELS:]
            for band, (net, cal, level) in measured.items():
                peak = float(np.median(net.ravel()[brightest]))
                result[f"{band.lower()}_peak"] = surface_brightness(
                    max(peak, level), cal
                )
    return result


def nebula_brightness(openngc: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    rows = openngc[openngc["type"].isin(NEBULA_TYPES) & openngc["major_axis"].notna()]
    out = []

    def measure(row):
        try:
            return row["name"], fram_brightness(row), None
        except Exception as e:  # one failed cutout should not stop the build
            return row["name"], None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(workers) as pool:
        for i, (name, result, error) in enumerate(
            pool.map(measure, (r for _, r in rows.iterrows()))
        ):
            print(f"\r  CTA-FRAM {i + 1}/{len(rows)} {name:<10}", end="", flush=True)
            if error:
                print(f"\n  {name}: {error}")
            if result and any(np.isfinite(v) for v in result.values()):
                out.append({"name": name, **result})
    print()
    return pd.DataFrame(out)


def build_filters() -> None:
    for svo_id, filename, energy_counting in FILTERS:
        query = urllib.parse.urlencode({"format": "ascii", "id": svo_id})
        curve = np.loadtxt(io.BytesIO(fetch(f"{SVO_FPS}?{query}")))
        wavelength = curve[:, 0] / 1e4  # Angstrom -> micron
        response = curve[:, 1]
        if energy_counting:
            # A photon-counting curve weights the photon flux. For the same
            # magnitudes it equals the energy-counting curve divided by the
            # wavelength.
            response = response / wavelength
        response = np.clip(response / response.max(), 0, None)
        np.savetxt(
            FLUX_CALIBRATION_DIR / filename,
            np.column_stack([wavelength, response]),
            delimiter=",",
            fmt="%.6g",
        )
        print(f"  wrote {filename}")


def main() -> None:
    TARGETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Filter curves")
    build_filters()

    print("OpenNGC")
    openngc = build_openngc()
    openngc.to_csv(TARGETS_DIR / "openngc.csv", index=False)
    print(f"  wrote openngc.csv, {len(openngc)} objects")

    print("Nebulae")
    brightness = nebula_brightness(openngc)
    brightness.to_csv(
        TARGETS_DIR / "nebula_brightness.csv", index=False, float_format="%.3f"
    )
    print(f"  wrote nebula_brightness.csv, {len(brightness)} nebulae")


if __name__ == "__main__":
    main()
