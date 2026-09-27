"""
Build the data files that `mphot.targets` and `mphot.extended` read.

Run it from the repository root. It needs network access, and it takes a few
minutes because it cuts out one Halpha map for each emission nebula:

    python resources/targets/build_targets.py

It writes:

* ``src/mphot/datafiles/targets/openngc.csv`` — the NGC, IC and addendum
  objects of OpenNGC, trimmed to the columns mphot uses. OpenNGC is released
  under CC-BY-SA-4.0 (https://github.com/mattiaverga/OpenNGC), and so is this
  derived file.
* ``src/mphot/datafiles/targets/nebula_lines.csv`` — the observed Halpha flux
  and line ratios of the planetary and emission nebulae in that file.
* ``src/mphot/datafiles/flux_calibration/{bessell_b,bessell_v,2mass_j,2mass_h,2mass_ks}.csv``
  — the catalogue bands, as photon-counting response curves.

Sources:

* OpenNGC, pinned to one commit so that the build can be repeated.
* Planetary nebulae: Hbeta fluxes and line intensities from the Strasbourg-ESO
  catalogue (Acker et al. 1992, VizieR V/84), and Hbeta fluxes and extinction
  constants from Cahn, Kaler & Stanghellini (1992, VizieR J/A+AS/94/399).
* Other emission nebulae: the all-sky Halpha map of Finkbeiner (2003,
  ApJS 146, 407), read through the CDS hips2fits service.
* Filter curves: the SVO Filter Profile Service (Bessell 1990 B and V; Cohen
  et al. 2003 2MASS J, H and Ks).
"""

import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

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
    ("2MASS/2MASS.J", "2mass_j.csv", False),
    ("2MASS/2MASS.H", "2mass_h.csv", False),
    ("2MASS/2MASS.Ks", "2mass_ks.csv", False),
)

GALAXY_TYPES = {"G", "GPair", "GTrpl", "GGroup"}
EMISSION_TYPES = {"HII", "EmN", "Cl+N", "Neb", "SNR"}

# 1 Rayleigh is 1e10 / (4 pi) photons/s/m2/sr. Per arcsec2, times the energy of
# an Halpha photon, and converted from W/m2 to erg/s/cm2 (a factor 1e3), that
# is 5.66e-18 erg/s/cm2/arcsec2.
ARCSEC2_PER_SR = 4.25452e10
HALPHA_PHOTON_J = 6.62607e-34 * 2.99792e8 / 0.65628e-6
RAYLEIGH_ERG = 1e10 / (4 * np.pi) / ARCSEC2_PER_SR * HALPHA_PHOTON_J * 1e3

# Beam of the Halpha map. SHASSA gives the map a 6' beam south of +15 deg.
# Further north it is 6' only where VTSS covers the sky, and 1 deg (WHAM)
# elsewhere.
SHASSA_DEC_LIMIT = 15.0
BEAM_ARCMIN = 6.0


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "mphot build"})
    with urllib.request.urlopen(request, timeout=180) as response:
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

    # Kept only for the joins below, not written out.
    table["png"] = raw["Identifiers"].str.extract(r"PN G(\d{3}\.\d[+-]\d{2}\.\d)")[0]
    return table.reset_index(drop=True)


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
    if header["BITPIX"] != "-32":
        raise ValueError(f"expected 32-bit float image, got BITPIX {header['BITPIX']}")
    image = np.frombuffer(data[start : start + nx * ny * 4], dtype=">f4")
    return image.reshape(ny, nx).astype(float)


def halpha_map_flux(row: pd.Series) -> tuple[float, str] | None:
    """
    Total observed Halpha flux of a nebula from the Finkbeiner map.

    A nebula more than three beams across is measured as the mean inside its
    ellipse. A smaller one is measured by adding up the flux in a circle grown
    by 1.5 beams, because the beam spreads the flux beyond the nebula. Both
    subtract the median of a surrounding annulus.

    Two cases are left out, because a check against the catalogue magnitudes
    showed they fail. A nebula smaller than the beam cannot be told apart from
    the emission around it. And a small nebula north of the SHASSA survey may
    sit in a 1 deg beam, where an aperture wide enough for its flux collects
    the emission for degrees around: such values came out about 10 times too
    bright, and up to several hundred times for the smallest.

    Returns:
        (flux in erg/s/cm2, method label), or None when the map cannot measure
        the nebula.
    """

    a = row["major_axis"]
    b = row["minor_axis"] if np.isfinite(row["minor_axis"]) else a
    if not np.isfinite(a) or a < BEAM_ARCMIN:
        return None

    grown = a < 3 * BEAM_ARCMIN
    if grown and row["dec"] >= SHASSA_DEC_LIMIT:
        return None

    beam = BEAM_ARCMIN
    pix = 1.0  # arcmin
    r_in = (a / 2 + 1.5 * beam) if grown else a / 2
    r_out = r_in + 15.0
    npix = int(np.ceil(2 * r_out / pix)) + 2

    query = urllib.parse.urlencode(
        {
            "hips": "CDS/P/Finkbeiner",
            "width": npix,
            "height": npix,
            "fov": npix * pix / 60,
            "ra": row["ra"],
            "dec": row["dec"],
            "projection": "TAN",
            "format": "fits",
        }
    )
    image = read_fits_image(fetch(f"{HIPS2FITS}?{query}"))

    y, x = np.indices(image.shape)
    centre = (npix - 1) / 2
    r = np.hypot(x - centre, y - centre) * pix
    annulus = image[(r > r_in) & (r < r_out)]
    background = np.nanmedian(annulus)
    net = image - background

    if grown:
        inside = r <= r_in
        flux_r_arcmin2 = np.nansum(net[inside]) * pix**2
        method = "Finkbeiner 2003 map, flux in a grown aperture"
    else:
        # OpenNGC has no position angle for most nebulae, so the ellipse is
        # replaced by a circle of the same area.
        inside = r <= np.sqrt(a * b) / 2
        flux_r_arcmin2 = np.nanmean(net[inside]) * np.pi / 4 * a * b
        method = "Finkbeiner 2003 map, mean inside the nebula"

    # The annulus scatter, scaled to the number of independent beams inside,
    # gives a rough error. Nebulae below three times that are left out.
    beams_inside = max(inside.sum() * pix**2 / (np.pi / 4 * beam**2), 1.0)
    noise = np.nanstd(annulus) * np.sqrt(beams_inside) * np.pi / 4 * beam**2
    if not flux_r_arcmin2 > 3 * noise:
        return None

    return flux_r_arcmin2 * 3600 * RAYLEIGH_ERG, method


def vizier(table: str, columns: str) -> pd.DataFrame:
    rows = _tap_sync_csv(VIZIER_TAP, f'SELECT {columns} FROM "{table}"', timeout=180)
    return pd.DataFrame(rows)


def normalise_pn_name(name: str) -> str:
    """'NGC  650-1' -> 'NGC0650', 'IC 418' -> 'IC0418'."""
    name = name.strip().split("-")[0].replace(" ", "")
    for prefix in ("NGC", "IC"):
        if name.startswith(prefix) and name[len(prefix) :].isdigit():
            return f"{prefix}{int(name[len(prefix) :]):04d}"
    return name


def planetary_lines(openngc: pd.DataFrame) -> pd.DataFrame:
    """Observed Halpha flux and line ratios of the planetary nebulae."""

    num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731

    hbeta = vizier("V/84/hbeta", 'PNG, "log(Fbeta)" AS lf')
    hbeta["PNG"] = hbeta["PNG"].str.strip()
    hbeta["lf"] = num(hbeta["lf"])
    hbeta = hbeta.groupby("PNG")["lf"].median()

    lines = ["I5007", "I5876", "I6563", "I6584", "I6717", "I6731"]
    intens = vizier("V/84/intens", "PNG, LineRef, n_I5007, " + ", ".join(lines))
    intens["PNG"] = intens["PNG"].str.strip()
    for line in lines:
        intens[line] = num(intens[line])
    # A star marks a measurement of 4959 because 5007 was saturated.
    saturated = intens["n_I5007"].fillna("").str.strip() == "*"
    intens.loc[saturated, "I5007"] *= 2.98

    cks1 = vizier("J/A+AS/94/399/table1", 'Name, "log(FHb)" AS lf')
    cks1["name"] = cks1["Name"].map(normalise_pn_name)
    cks1 = cks1.groupby("name")["lf"].apply(lambda s: num(s).median())
    cks2 = vizier("J/A+AS/94/399/table2", "Name, calpha")
    cks2["name"] = cks2["Name"].map(normalise_pn_name)
    cks2 = cks2.groupby("name")["calpha"].apply(lambda s: num(s).median())

    out = []
    for _, row in openngc[openngc["type"] == "PN"].iterrows():
        png = row["png"]
        c = cks2.get(row["name"], np.nan)

        log_hbeta, source = np.nan, None
        if isinstance(png, str) and np.isfinite(hbeta.get(png, np.nan)):
            log_hbeta, source = hbeta[png], "Acker et al. 1992 Hbeta"
        elif np.isfinite(cks1.get(row["name"], np.nan)):
            log_hbeta, source = cks1[row["name"]], "Cahn et al. 1992 Hbeta"
        if not np.isfinite(log_hbeta):
            continue

        # Line intensities relative to Halpha, as the median of all
        # observations. An observation made relative to Halpha gives no Hbeta.
        obs = intens[intens["PNG"] == png] if isinstance(png, str) else intens[:0]
        ratios = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            to_halpha = obs[lines].div(obs["I6563"], axis=0)
            hb_rel = (100 / obs["I6563"]).where(obs["LineRef"].str.strip() == "b")
        for line, key in (
            ("I5007", "oiii_5007"),
            ("I5876", "hei_5876"),
            ("I6584", "nii_6584"),
            ("I6717", "sii_6717"),
            ("I6731", "sii_6731"),
        ):
            ratios[key] = to_halpha[line].median()
        hbeta_to_halpha = hb_rel.median()

        # Observed Halpha/Hbeta: measured if possible, otherwise the case B
        # value reddened by the extinction constant c, since
        # F(Ha)/F(Hb) = 2.86 * 10**(-c * f(Ha)) with f(Ha) = -0.323.
        if np.isfinite(hbeta_to_halpha):
            halpha_to_hbeta = 1 / hbeta_to_halpha
        else:
            halpha_to_hbeta = 2.86 * 10 ** (0.323 * (c if np.isfinite(c) else 0.0))
            hbeta_to_halpha = 1 / halpha_to_hbeta

        out.append(
            {
                "name": row["name"],
                "halpha_flux": 10**log_hbeta * halpha_to_hbeta,  # erg/s/cm2
                "halpha_source": source,
                "c_hbeta": c,
                "hbeta": hbeta_to_halpha,
                **ratios,
                "ratio_source": "Acker et al. 1992" if len(obs) else None,
            }
        )
    return pd.DataFrame(out)


def emission_lines(openngc: pd.DataFrame) -> pd.DataFrame:
    """Observed Halpha flux of the other emission nebulae, from the map."""

    out = []
    rows = openngc[openngc["type"].isin(EMISSION_TYPES)]
    for i, (_, row) in enumerate(rows.iterrows()):
        print(
            f"\r  Halpha map {i + 1}/{len(rows)} {row['name']:<10}", end="", flush=True
        )
        try:
            result = halpha_map_flux(row)
        except Exception as e:  # one failed cutout should not stop the build
            print(f"\n  {row['name']}: {type(e).__name__}: {e}")
            continue
        if result is None:
            continue
        flux, method = result
        out.append({"name": row["name"], "halpha_flux": flux, "halpha_source": method})
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
    openngc.drop(columns="png").to_csv(TARGETS_DIR / "openngc.csv", index=False)
    print(f"  wrote openngc.csv, {len(openngc)} objects")

    print("Planetary nebulae")
    pn = planetary_lines(openngc)
    print(f"  {len(pn)} of {(openngc['type'] == 'PN').sum()} with an Hbeta flux")

    print("Emission nebulae")
    em = emission_lines(openngc)
    print(f"  {len(em)} of {openngc['type'].isin(EMISSION_TYPES).sum()} measured")

    lines = pd.concat([pn, em], ignore_index=True)
    lines.to_csv(TARGETS_DIR / "nebula_lines.csv", index=False, float_format="%.5g")
    print(f"  wrote nebula_lines.csv, {len(lines)} nebulae")


if __name__ == "__main__":
    main()
