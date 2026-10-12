"""
Exposure times for images of galaxies and nebulae.

A galaxy or nebula covers many pixels. Thus mphot uses the surface brightness
(e/s per pixel), not the total flux. By default, this is the mean over the
catalogue ellipse.

The calculation has three steps:

1. Find the target, sky and dark rates per pixel.
2. Find the sub-exposure time. This is the shortest time that is
   background-limited (the sky and dark variance is 10 times the read-noise
   variance) and that uses no more than 10% of the time for the readout. The
   brightest part of the target must stay below well_fill.
3. Find the number of sub-exposures that gives the requested SNR.

Galaxies get their brightness from the catalogue magnitudes. Nebulae get it
from B, V and R images of the CTA-FRAM Sky Survey, with the stars removed: the
mean in the catalogue ellipse and the brightest part.

Galaxies and reflection nebulae have a stellar spectrum that agrees with their
colours. Planetary nebulae, emission nebulae and supernova remnants have
emission lines with typical ratios. The spectrum is scaled to the catalogue or
CTA-FRAM band nearest to the instrument band.
"""

import math
from functools import cache

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import brentq, nnls
from scipy.special import expit

from mphot.constants import (
    AIRMASS_VALUES,
    GRID_FLUX_INGREDIENTS_NAME,
    GRID_RADIANCE_INGREDIENTS_NAME,
    PWV_VALUES,
    TEFF_VALUES,
    VEGA_FILE,
)
from mphot.paths import DATAFILES_DIR, FLUX_CALIBRATION_DIR, system_response_path
from mphot.precision import _peak_pixel_rate, convert_airmass, image_pixel
from mphot.targets import Target, resolve_target

H_C = 6.62607015e-34 * 2.99792458e8  # J m
C_KMS = 299792.458  # km/s

CATALOGUE_BANDS = {
    # band: (response file, effective wavelength [micron])
    "B": ("bessell_b.csv", 0.438),
    "V": ("bessell_v.csv", 0.545),
    "R": ("bessell_r.csv", 0.641),
    "J": ("2mass_j.csv", 1.235),
    "H": ("2mass_h.csv", 1.662),
    "K": ("2mass_ks.csv", 2.159),
}
"""Photon-counting response curves of the OpenNGC magnitudes and of the
CTA-FRAM surface brightness."""

HALPHA_UM = 0.65628

HYDROGEN_LINES = (
    # label, rest wavelength in air [micron], case B intensity relative to
    # Halpha at 10^4 K (Osterbrock & Ferland 2006; Hummer & Storey 1987)
    ("Hgamma", 0.43405, 0.468 / 2.86),
    ("Hbeta", 0.48613, 1 / 2.86),
    ("Halpha", HALPHA_UM, 1.0),
    ("Pa10", 0.90152, 0.0185 / 2.86),
    ("Pa9", 0.92290, 0.0255 / 2.86),
    ("Paepsilon", 0.95460, 0.0367 / 2.86),
    ("Padelta", 1.00494, 0.0553 / 2.86),
    ("Pagamma", 1.09381, 0.090 / 2.86),
    ("Pabeta", 1.28181, 0.162 / 2.86),
    ("Paalpha", 1.87510, 0.332 / 2.86),
    ("Brgamma", 2.16553, 0.0275 / 2.86),
)
"""Hydrogen recombination lines. Their ratios are set by atomic physics."""

METAL_LINES = (
    # key, label, rest wavelength in air [micron]
    ("oiii_5007", "[OIII] 5007", 0.50068),
    ("hei_5876", "HeI 5876", 0.58756),
    ("oi_6300", "[OI] 6300", 0.63003),
    ("nii_6584", "[NII] 6584", 0.65835),
    ("sii_6717", "[SII] 6717", 0.67164),
    ("sii_6731", "[SII] 6731", 0.67308),
    ("ariii_7136", "[ArIII] 7136", 0.71358),
    ("oii_7325", "[OII] 7325", 0.73250),
    ("siii_9532", "[SIII] 9532", 0.95311),
    ("hei_10830", "HeI 10830", 1.08303),
)
"""Helium and forbidden lines. Their ratios to Halpha differ from nebula to
nebula. mphot uses `TYPICAL_LINE_RATIOS`."""

DOUBLET_PARTNERS = (
    # key of the strong line, label, rest wavelength in air [micron], fraction
    ("oiii_5007", "[OIII] 4959", 0.49589, 1 / 2.98),
    ("nii_6584", "[NII] 6548", 0.65480, 1 / 3.05),
    ("oi_6300", "[OI] 6364", 0.63636, 1 / 3.0),
    ("siii_9532", "[SIII] 9069", 0.90686, 1 / 2.44),
)
"""The weaker line of each doublet, a fixed fraction of the stronger one."""

TYPICAL_LINE_RATIOS = {
    "emission nebula": {
        "oiii_5007": 0.35,
        "hei_5876": 0.035,
        "oi_6300": 0.01,
        "nii_6584": 0.3,
        "sii_6717": 0.1,
        "sii_6731": 0.08,
        "ariii_7136": 0.05,
        "oii_7325": 0.03,
        "siii_9532": 0.3,
        "hei_10830": 0.15,
    },
    "planetary nebula": {
        "oiii_5007": 3.5,
        "hei_5876": 0.05,
        "oi_6300": 0.02,
        "nii_6584": 0.5,
        "sii_6717": 0.03,
        "sii_6731": 0.04,
        "ariii_7136": 0.08,
        "oii_7325": 0.05,
        "siii_9532": 0.3,
        "hei_10830": 0.3,
    },
    "supernova remnant": {
        "oiii_5007": 0.7,
        "hei_5876": 0.0,
        "oi_6300": 0.2,
        "nii_6584": 0.6,
        "sii_6717": 0.45,
        "sii_6731": 0.35,
        "ariii_7136": 0.02,
        "oii_7325": 0.1,
        "siii_9532": 0.3,
        "hei_10830": 0.1,
    },
}
"""
Rough observed line ratios to Halpha of each kind of nebula.

Real values can be 3 times higher or lower. A supernova remnant has
[SII]/Halpha above approximately 0.4. When a nebula has B, V and R, mphot fits
[OIII] to them and does not use the typical [OIII] value.
"""

OIII_LINES = ("[OIII] 5007", "[OIII] 4959")
"""The lines that the fit to B, V and R gives their own level."""

CONTINUUM_TEMPERATURE = 10000
"""The continuum of a nebula is starlight that dust scatters. mphot gives it
the spectrum of a hot star."""

BEYOND_R_UM = 0.8
"""Wavelength beyond which the CTA-FRAM bands measure almost nothing."""

SURVEY_B_RED_EDGE_UM = 0.49
"""
Red edge of the CTA-FRAM B filter [micron].

The filter passes almost no [OIII] 5007: planetary nebulae with strong [OIII],
for example NGC 3242, NGC 7009 and M27, are 0.7 to 1.3 mag fainter in it than
in the Johnson B of the catalogues. Its curve is not published. With the
Bessell B curve cut at 0.49 micron, the fit reproduces B, V and R of these
nebulae and of HII regions such as M8, M17 and M42. Without the cut, it misses
them by up to 55%.
"""

FIT_TOLERANCE = 0.1
"""Largest difference between a fitted and a measured band before a warning."""

FIT_LIMIT = 0.25
"""Largest difference between a fitted and a measured band for which mphot uses
the fit. A larger difference is mostly a star in one band."""

DEFAULT_TEMPERATURE = {"galaxy": 5000, "reflection nebula": 10000}
"""Spectrum used when a continuum source has only one catalogue magnitude."""

DEFAULT_DISK_OFFSET = 2.5
"""Centre minus mean surface brightness of a galaxy without a B magnitude."""

FAR_BAND_UM = 0.2
"""Distance from the nearest catalogue band beyond which the spectrum matters."""

NARROW_BAND_UM = 0.03
"""Equivalent width below which an instrument band counts as narrowband."""

BACKGROUND_LIMITED = 10.0
"""A sub-exposure is background-limited when the sky and dark variance is this
many times the read-noise variance. The read noise then adds less than 5% to
the noise."""

READOUT_FRACTION = 0.1
"""The largest fraction of the total time that the readout can use."""

CONTRIBUTION_WARNING = 0.1
"""Share of the counts above which an assumed line is worth a warning."""


# ----------------------------------------------------------------------------
# Spectra and bands


@cache
def _ingredients() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Wavelengths, stellar spectra with atmospheric transmission, and sky."""
    flux = pd.read_pickle(DATAFILES_DIR / GRID_FLUX_INGREDIENTS_NAME)
    radiance = pd.read_pickle(DATAFILES_DIR / GRID_RADIANCE_INGREDIENTS_NAME)
    return flux.index.to_numpy(), flux, radiance


def _wavelengths() -> np.ndarray:
    return _ingredients()[0]


def _integral(y: np.ndarray) -> float:
    return float(trapezoid(y, _wavelengths()))


@cache
def _vega() -> np.ndarray:
    """Vega photon flux [photons/s/m2/micron] on the ingredient wavelengths."""
    vega = pd.read_csv(DATAFILES_DIR / VEGA_FILE, header=None, index_col=0)[1]
    return np.interp(_wavelengths(), vega.index.to_numpy(), vega.to_numpy())


@cache
def _catalogue_band(band: str) -> np.ndarray:
    curve = np.loadtxt(FLUX_CALIBRATION_DIR / CATALOGUE_BANDS[band][0], delimiter=",")
    return np.interp(_wavelengths(), curve[:, 0], curve[:, 1], left=0, right=0)


@cache
def _vega_in_band(band: str) -> float:
    return _integral(_vega() * _catalogue_band(band))


@cache
def _survey_band(band: str) -> np.ndarray:
    """
    Photon-counting curve of a CTA-FRAM band, for emission lines.

    The zero point of each image comes from stars, so for a continuum the
    CTA-FRAM bands are the Johnson-Cousins bands. For lines, the edges of the
    filters matter, and the B filter has a red edge that Bessell's B has not.
    """
    curve = _catalogue_band(band)
    if band == "B":
        curve = curve * expit((SURVEY_B_RED_EDGE_UM - _wavelengths()) / 0.002)
    return curve


@cache
def _vega_in_survey_band(band: str) -> float:
    return _integral(_vega() * _survey_band(band))


def _stellar_spectrum(temperature: int) -> np.ndarray:
    return _ingredients()[1][f"{temperature}K"].to_numpy()


@cache
def _synthetic_magnitudes() -> pd.DataFrame:
    """Vega magnitude of each stellar spectrum in each band, up to a constant."""
    temperatures = [t for t in TEFF_VALUES if 3000 <= t <= 40000]
    return pd.DataFrame(
        {
            band: [
                -2.5
                * np.log10(
                    _integral(_stellar_spectrum(t) * _catalogue_band(band))
                    / _vega_in_band(band)
                )
                for t in temperatures
            ]
            for band in CATALOGUE_BANDS
        },
        index=temperatures,
    )


def _matching_temperature(magnitudes: dict) -> int | None:
    """
    Temperature of the stellar spectrum whose colours best match.

    Returns None with fewer than two magnitudes, since one magnitude has no
    colour.
    """

    if len(magnitudes) < 2:
        return None
    bands = list(magnitudes)
    observed = np.array([magnitudes[b] for b in bands])
    synthetic = _synthetic_magnitudes()[bands].to_numpy()
    residual = observed - synthetic
    residual -= residual.mean(axis=1, keepdims=True)
    return int(_synthetic_magnitudes().index[np.argmin((residual**2).sum(axis=1))])


def _interpolate_columns(frame: pd.DataFrame, pwv: float, airmass: float) -> np.ndarray:
    """Bilinear interpolation between the four grid columns around a point."""

    i = int(np.clip(np.searchsorted(PWV_VALUES, pwv) - 1, 0, len(PWV_VALUES) - 2))
    j = int(
        np.clip(
            np.searchsorted(AIRMASS_VALUES, airmass) - 1, 0, len(AIRMASS_VALUES) - 2
        )
    )
    p0, p1 = PWV_VALUES[i], PWV_VALUES[i + 1]
    a0, a1 = AIRMASS_VALUES[j], AIRMASS_VALUES[j + 1]
    wp = (pwv - p0) / (p1 - p0)
    wa = (airmass - a0) / (a1 - a0)

    def column(p, a):
        return frame[f"{p}_{a}"].to_numpy()

    return (
        (1 - wp) * (1 - wa) * column(p0, a0)
        + wp * (1 - wa) * column(p1, a0)
        + (1 - wp) * wa * column(p0, a1)
        + wp * wa * column(p1, a1)
    )


def _system_response(name: str) -> np.ndarray:
    """System response of an instrument on the ingredient wavelengths."""
    response = pd.read_csv(system_response_path(name), header=None, index_col=0)[1]
    response = response.dropna().sort_index()
    return np.interp(
        _wavelengths(), response.index.to_numpy(), response.to_numpy(), left=0, right=0
    )


def extinction_curve(wavelength: float) -> float:
    """
    Extinction A(lambda)/A(V) of Cardelli, Clayton & Mathis (1989), R_V = 3.1.

    Args:
        wavelength (float): Wavelength [micron], from 0.3 to 3.3.

    Returns:
        float: A(lambda)/A(V).
    """

    x = float(np.clip(1 / wavelength, 0.3, 3.3))
    if x < 1.1:
        a, b = 0.574 * x**1.61, -0.527 * x**1.61
    else:
        y = x - 1.82
        a = np.polyval(
            [0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1], y
        )
        b = np.polyval(
            [-2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0], y
        )
    return float(a + b / 3.1)


# ----------------------------------------------------------------------------
# Source models


def _surface_brightness(target: Target) -> dict:
    """
    Mean surface brightness in each band [mag/arcsec2].

    A nebula has it from the CTA-FRAM images. For a galaxy, it is the total
    magnitude spread over the catalogue ellipse.
    """

    if target.kind != "galaxy":
        if not target.surface_brightness:
            raise ValueError(
                f"The CTA-FRAM images give no surface brightness for "
                f"{target.label}. mphot cannot calculate it."
            )
        return dict(target.surface_brightness)
    if not target.magnitudes:
        raise ValueError(
            f"The catalogue gives no magnitude for {target.label}. mphot cannot "
            "calculate its surface brightness."
        )
    area = 2.5 * math.log10(target.area)
    return {band: m + area for band, m in target.magnitudes.items()}


def _scaled_to(target: Target, band: str, mu: dict) -> str:
    if target.kind == "galaxy":
        return f"{band} = {target.magnitudes[band]:.2f} over the catalogue ellipse"
    return f"{band} = {mu[band]:.2f} mag/arcsec2 in the CTA-FRAM images"


def _continuum(
    target: Target, response: np.ndarray, band_centre: float, mu: dict | None = None
) -> dict:
    """
    Surface brightness of a continuum source in the instrument band.

    `mu` is the surface brightness in each band [mag/arcsec2]. By default it is
    the mean of the target.
    """

    mu = _surface_brightness(target) if mu is None else mu
    kind = "catalogue" if target.kind == "galaxy" else "CTA-FRAM"

    warnings = []
    temperature = _matching_temperature(mu)
    how = f"from the {kind} colours"
    if temperature is None:
        default = DEFAULT_TEMPERATURE[target.kind]
        temperature = int(min(TEFF_VALUES, key=lambda t: abs(t - default)))
        how = "assumed"
        source = (
            "The catalogue gives only one magnitude"
            if target.kind == "galaxy"
            else "The CTA-FRAM images give only one band"
        )
        warnings.append(
            f"{source} for {target.label}. mphot uses a {temperature} K spectrum."
        )

    centres = {b: CATALOGUE_BANDS[b][1] for b in mu}
    band = min(centres, key=lambda b: abs(centres[b] - band_centre))
    distance = abs(centres[band] - band_centre)
    if distance > FAR_BAND_UM:
        if min(centres.values()) <= band_centre <= max(centres.values()):
            # Between two bands, the colours of the target constrain the
            # spectrum on both sides, so the error is smaller.
            lower = max(
                (b for b in centres if centres[b] <= band_centre), key=centres.get
            )
            upper = min(
                (b for b in centres if centres[b] >= band_centre), key=centres.get
            )
            warnings.append(
                f"The filter band is between the {kind} bands {lower} and "
                f"{upper}. The result depends on the spectrum between these bands."
            )
        else:
            warnings.append(
                f"The nearest {kind} band ({band}) is {distance:.2f} micron from "
                "the filter band. The error can be up to 1 mag."
            )

    spectrum = _stellar_spectrum(temperature)
    scale = (
        10 ** (-0.4 * mu[band])
        * _vega_in_band(band)
        / _integral(spectrum * _catalogue_band(band))
    )

    return {
        "rate": scale * _integral(spectrum * response),
        "spectrum": f"{temperature} K stellar spectrum ({how})",
        "scaled to": _scaled_to(target, band, mu),
        "band": band,
        "lines": {},
        "warnings": warnings,
    }


def _line_list(target: Target, av: float) -> list[tuple]:
    """(label, rest wavelength, observed energy ratio to Halpha, typical?)."""

    ratios = TYPICAL_LINE_RATIOS[target.kind]
    a_halpha = av * extinction_curve(HALPHA_UM)

    lines = []
    for label, wavelength, case_b in HYDROGEN_LINES:
        reddening = 10 ** (-0.4 * (av * extinction_curve(wavelength) - a_halpha))
        lines.append((label, wavelength, case_b * reddening, False))
    for key, label, wavelength in METAL_LINES:
        if ratios.get(key, 0) > 0:
            lines.append((label, wavelength, ratios[key], True))
    for key, label, wavelength, fraction in DOUBLET_PARTNERS:
        if ratios.get(key, 0) > 0:
            lines.append((label, wavelength, ratios[key] * fraction, True))
    return lines


@cache
def _scattered_light() -> np.ndarray:
    """Continuum of a nebula, 1 photon/s/m2/arcsec2 in the V band."""
    temperature = int(min(TEFF_VALUES, key=lambda t: abs(t - CONTINUUM_TEMPERATURE)))
    spectrum = _stellar_spectrum(temperature)
    return spectrum / _integral(spectrum * _catalogue_band("V"))


def _emission(
    target: Target,
    response: np.ndarray,
    band_centre: float,
    av: float | None,
    mu: dict | None = None,
) -> dict:
    """
    Surface brightness of an emission nebula in the instrument band.

    The spectrum has three parts: Halpha with the lines tied to it by typical
    ratios, [OIII], and a continuum of scattered starlight. With B, V and R, a
    fit gives each part the level that reproduces the three bands. With fewer
    bands, [OIII] has its typical ratio, there is no continuum, and the lines
    get the level that gives the band nearest to the instrument band.

    `mu` is the surface brightness in each band [mag/arcsec2]. By default it is
    the mean of the target.
    """

    mu = _surface_brightness(target) if mu is None else mu
    warnings = []
    extinction_known = av is not None
    av = av or 0.0

    lam = _wavelengths()
    shift = 1 + target.radial_velocity / C_KMS
    lines = _line_list(target, av)
    oiii_5007 = next((r for label, _, r, _ in lines if label == OIII_LINES[0]), 0.0)

    # Each part is (lines, continuum). The lines map a label to the observed
    # wavelength and the photons per photon of the leading line: Halpha, or
    # [OIII] 5007. An energy ratio becomes a photon ratio when multiplied by
    # the wavelength ratio.
    parts = {
        "Halpha": (
            {
                label: (wavelength * shift, ratio * wavelength / HALPHA_UM)
                for label, wavelength, ratio, _ in lines
                if label not in OIII_LINES
            },
            None,
        ),
        "[OIII]": (
            {
                label: (wavelength * shift, ratio / oiii_5007 * wavelength / 0.50068)
                for label, wavelength, ratio, _ in lines
                if label in OIII_LINES
            },
            None,
        ),
        "Continuum": ({}, _scattered_light()),
    }

    def part_rate(part, curve) -> float:
        part_lines, continuum = part
        rate = sum(n * np.interp(w, lam, curve) for w, n in part_lines.values())
        if continuum is not None:
            rate += _integral(continuum * curve)
        return float(rate)

    measured = {b: 10 ** (-0.4 * m) * _vega_in_survey_band(b) for b, m in mu.items()}
    band = min(mu, key=lambda b: abs(CATALOGUE_BANDS[b][1] - band_centre))
    fit_miss = None
    if {"B", "V", "R"} <= set(mu):
        # Each row is one band, divided by its measured value, so that the fit
        # weights the bands by their relative error.
        names = ("B", "V", "R")
        matrix = np.array(
            [
                [part_rate(p, _survey_band(b)) / measured[b] for p in parts.values()]
                for b in names
            ]
        )
        amplitudes, _ = nnls(matrix, np.ones(len(names)))
        misses = dict(zip(names, np.abs(matrix @ amplitudes - 1), strict=True))
        worst = max(misses, key=misses.get)
        fit_miss = float(misses[worst])
    fitted = fit_miss is not None and fit_miss <= FIT_LIMIT
    if fitted:
        if fit_miss > FIT_TOLERANCE:
            warnings.append(
                f"The fitted spectrum differs by {fit_miss:.0%} from the CTA-FRAM "
                f"{worst} band. Halpha lines, [OIII] and scattered starlight do not "
                f"describe {target.label} well."
            )
        spectrum = "Emission lines and scattered starlight, fitted to B, V and R"
        scaled_to = (
            ", ".join(f"{b} = {mu[b]:.2f}" for b in names)
            + " mag/arcsec2 in the CTA-FRAM images"
        )
    else:
        # [OIII] has its typical ratio, in photons per Halpha photon.
        typical = np.array([1.0, oiii_5007 * 0.50068 / HALPHA_UM, 0.0])
        rates = np.array([part_rate(p, _survey_band(band)) for p in parts.values()])
        amplitudes = typical * measured[band] / (typical @ rates)
        spectrum = "Emission lines with typical ratios"
        scaled_to = _scaled_to(target, band, mu)
        if fit_miss is None:
            warnings.append(
                f"The CTA-FRAM images give only {', '.join(mu)} for "
                f"{target.label}. mphot uses the typical [OIII] ratio and no "
                "continuum."
            )
        else:
            warnings.append(
                f"The CTA-FRAM colours of {target.label} do not fit Halpha lines, "
                f"[OIII] and scattered starlight: the {worst} band misses by "
                f"{fit_miss:.0%}. A star in the images can cause this. mphot scales "
                f"lines with typical ratios to the {band} band instead."
            )
        if target.kind == "supernova remnant":
            warnings.append(
                "mphot calculates only the line emission. Some remnants, for "
                "example M1, also have a strong continuum."
            )
    if av:
        spectrum += f", A_V = {av:.2f}"

    counts = {}
    beyond_r = 0.0
    for (name, (part_lines, continuum)), amplitude in zip(
        parts.items(), amplitudes, strict=True
    ):
        for label, (wavelength, n) in part_lines.items():
            counts[label] = amplitude * n * float(np.interp(wavelength, lam, response))
            if wavelength > BEYOND_R_UM:
                beyond_r += counts[label]
        if continuum is not None:
            counts[name] = amplitude * _integral(continuum * response)
            beyond_r += amplitude * _integral(
                np.where(lam > BEYOND_R_UM, continuum * response, 0.0)
            )
    total = sum(counts.values())

    if _integral(response) / response.max() < NARROW_BAND_UM:
        warnings.append(
            "The broad CTA-FRAM bands do not measure the line ratios inside a "
            "narrowband filter, for example [NII]/Halpha. The error can be a "
            "factor of 1.5."
        )
    if total > 0:
        typical_lines = [
            label
            for label, _, _, is_typical in lines
            if is_typical
            and not (fitted and label in OIII_LINES)
            and counts[label] > CONTRIBUTION_WARNING * total
        ]
        if typical_lines:
            warnings.append(
                f"mphot uses typical {target.kind} values for "
                f"{', '.join(typical_lines)}. Real values can be 3 times higher or "
                "lower."
            )
        if beyond_r > CONTRIBUTION_WARNING * total:
            note = (
                f"{beyond_r / total:.0%} of the signal is at wavelengths beyond the "
                "CTA-FRAM R band, where mphot uses typical line ratios and the "
                "spectrum of a hot star. The error can be a factor of 2 or more."
            )
            if not extinction_known:
                note += (
                    " The hydrogen lines also depend on the dust extinction, which "
                    "is not known. Use av to correct them."
                )
            warnings.append(note)

    return {
        "rate": total,
        "spectrum": spectrum,
        "scaled to": scaled_to,
        "band": band,
        "fit miss": fit_miss,
        "lines": {
            label: value for label, value in counts.items() if value > 1e-3 * total
        },
        "warnings": warnings,
    }


def _seconds(t: float) -> str:
    """A time for a note: 3 significant digits, but no exponent."""
    return f"{t:.0f} s" if t >= 100 else f"{t:.3g} s"


def _disk_peak_offset(target: Target) -> float:
    """
    How much brighter the centre of the galaxy is than its mean [mag/arcsec2].

    OpenNGC takes galaxy sizes from the 25 mag/arcsec2 B-band isophote. For an
    exponential disk, the mean surface brightness inside that isophote fixes
    the central surface brightness, so the one follows from the other.
    """

    b = target.magnitudes.get("B")
    if b is None:
        return DEFAULT_DISK_OFFSET
    mean = b + 2.5 * math.log10(target.area)

    def mean_inside_isophote(central: float) -> float:
        x = (25 - central) / (2.5 / math.log(10))  # isophote radius / scale length
        return central - 2.5 * math.log10(2 / x**2 * (1 - (1 + x) * math.exp(-x)))

    try:
        central = brentq(lambda c: mean_inside_isophote(c) - mean, 10, 24.99)
    except ValueError:
        return DEFAULT_DISK_OFFSET
    return mean - central


# ----------------------------------------------------------------------------
# Exposure time


def get_exposure_extended(
    target: str | Target,
    props: dict,
    props_sky: dict,
    snr: float = 10.0,
    area: str | float = "pixel",
    mu_offset: float = 0.0,
    peak_offset: float | None = None,
    h: float = 2440,
    flat_error: float = 0.0,
    av: float | None = None,
) -> dict:
    """
    Calculate the sub-exposure time and the number of sub-exposures for an SNR.

    Args:
        target (str | Target): Messier, NGC or IC name, for example "M51", or
            a Target from `resolve_target`.

        props (dict): Instrument properties per detector pixel, as for
            `get_precision`: "name", "plate_scale", "N_dc", "N_rn",
            "well_depth", "well_fill", "read_time", "r0", "r1", and optionally
            "min_exp", "max_exp", "pixel_binning" and "pixel_binning_type".

        props_sky (dict): "pwv" [mm], "airmass", and optionally "seeing"
            [arcsec] and "sky_factor", as for `get_precision`. mphot uses the
            seeing only for "saturation magnitude [mag]".

        snr (float, optional): The SNR to get. Default is 10.

        area (str | float, optional): Where the SNR applies: "pixel" for one
            image pixel, or an area [arcsec2]. Default is "pixel".

        mu_offset (float, optional): The surface brightness to measure, in
            mag/arcsec2 fainter than the catalogue mean. For example, 2 for the
            outer disk of a galaxy. Default is 0.

        peak_offset (float, optional): The surface brightness of the brightest
            part, in mag/arcsec2 brighter than the mean. By default, the centre
            of an exponential disk for a galaxy. For a nebula, the brightest
            part in the CTA-FRAM images, or 0 for a nebula they do not resolve.

        h (float, optional): Altitude of the site [m]. Default is 2440 (Paranal).

        flat_error (float, optional): The error that remains after
            flat-fielding and sky subtraction, as a fraction of the sky. This
            error is the same in all sub-exposures, so it sets a maximum SNR.
            Default is 0 (no error).

        av (float, optional): Dust extinction A_V [mag] of a nebula. It
            changes the near-infrared hydrogen lines relative to Halpha. By
            default 0.

    Returns:
        dict: The result. The units are in the keys. Values per pixel are for
            one image pixel. "warnings" gives the assumptions that are
            important for this target. The times are None if the SNR is not
            possible. "saturation magnitude [mag]" is
            the Vega magnitude, in the filter band, of a star that fills its
            brightest pixel to well_fill in one sub-exposure. The star is on
            the centre of a pixel and has a Gaussian profile with the FWHM of
            the seeing.
    """

    if isinstance(target, str):
        target = resolve_target(target)
    if not math.isfinite(target.area):
        raise ValueError(
            f"The catalogue gives no size for {target.label}. mphot cannot "
            "calculate its surface brightness."
        )

    warnings = []
    pwv = props_sky["pwv"]
    airmass = props_sky["airmass"]
    seeing = props_sky.get("seeing")
    airmass_paranal = convert_airmass(airmass, h)
    if not PWV_VALUES[0] <= pwv <= PWV_VALUES[-1]:
        raise ValueError(f"pwv must be {PWV_VALUES[0]}-{PWV_VALUES[-1]} mm, got {pwv}.")
    if not AIRMASS_VALUES[0] <= airmass_paranal <= AIRMASS_VALUES[-1]:
        raise ValueError(
            f"Airmass {airmass:.2f} at {h:.0f} m is airmass {airmass_paranal:.2f} at "
            f"Paranal, outside mphot's sky model ({AIRMASS_VALUES[0]:.0f}-"
            f"{AIRMASS_VALUES[-1]:.0f})."
        )

    lam, flux_frame, radiance_frame = _ingredients()
    system = _system_response(props["name"])
    response = system * _interpolate_columns(flux_frame, pwv, airmass_paranal)
    sky_radiance = _interpolate_columns(radiance_frame, pwv, airmass_paranal)

    throughput = _integral(response)
    if throughput <= 0:
        raise ValueError(f"The system response of {props['name']} is zero.")
    band_centre = _integral(lam * response) / throughput

    if target.kind in ("galaxy", "reflection nebula"):
        source = _continuum(target, response, band_centre)
        if target.kind == "galaxy" and throughput / response.max() < NARROW_BAND_UM:
            warnings.append(
                "mphot does not calculate the line emission of galaxies. In a "
                "narrowband filter, the signal is too low."
            )
    else:
        source = _emission(target, response, band_centre, av)
    warnings += source["warnings"]

    rate = source["rate"]  # e/s/m2/arcsec2 at the catalogue mean
    if rate <= 0:
        raise ValueError(
            f"{target.label} gives no signal in the band of {props['name']}. For a "
            "nebula, make sure that a line is in the filter band."
        )

    # The noise is per image pixel, the saturation per detector pixel.
    k, plate_scale, dark, read_noise = image_pixel(props)
    collecting_area = np.pi * (props["r0"] ** 2 - props["r1"] ** 2)
    per_pixel = collecting_area * plate_scale**2
    vega = _integral(_vega() * response)  # e/s/m2

    if peak_offset is None:
        if target.kind == "galaxy":
            peak_offset = _disk_peak_offset(target)
            warnings.append(
                "The brightest part is the centre of an exponential disk. A bulge "
                "or nucleus is brighter and fills the well sooner. Use peak_offset "
                "to change this."
            )
        elif target.peak_surface_brightness:
            # The brightest part has its own colours, so its spectrum is fitted
            # to them in the same way as the mean.
            peak_mu = target.peak_surface_brightness
            if target.kind == "reflection nebula":
                peak_source = _continuum(target, response, band_centre, peak_mu)
            else:
                peak_source = _emission(target, response, band_centre, av, peak_mu)
            ratio = peak_source["rate"] / rate
            peak_offset = max(2.5 * math.log10(ratio), 0.0) if ratio > 0 else 0.0
            warnings.append(
                "The brightest part comes from the CTA-FRAM images, which show no "
                'detail smaller than about 40". Smaller bright parts fill the '
                "well sooner. Use peak_offset to change this."
            )
            miss = peak_source.get("fit miss")
            if miss is not None and miss > FIT_TOLERANCE:
                warnings.append(
                    f"The colours of the brightest part miss the fitted spectrum by "
                    f"{miss:.0%} in one CTA-FRAM band. In this filter, the brightest "
                    "part can be up to 2 times brighter or fainter than calculated."
                )
        else:
            peak_offset = 0.0
            warnings.append(
                f"{target.label} is too small for the CTA-FRAM images to show its "
                "brightest part. mphot sets the brightest part to the mean. "
                "Brighter parts fill the well sooner. Use peak_offset to change "
                "this."
            )

    mean = rate * per_pixel  # e/s/pixel
    signal = mean * 10 ** (-0.4 * mu_offset)
    peak = mean * 10 ** (0.4 * peak_offset)
    sky = (
        _integral(sky_radiance * system) * per_pixel * props_sky.get("sky_factor", 1.0)
    )
    read_time = props["read_time"]
    full = props["well_depth"] * props["well_fill"]

    npix = 1.0 if area == "pixel" else float(area) / plate_scale**2

    # The shortest sub-exposure that is background-limited and that uses no
    # more than READOUT_FRACTION of the time for the readout. The brightest
    # part of the target must stay below well_fill, and the time must be within
    # min_exp and max_exp.
    background = sky + dark
    t_background = (
        BACKGROUND_LIMITED * read_noise**2 / background if background > 0 else math.inf
    )
    t_readout = read_time * (1 - READOUT_FRACTION) / READOUT_FRACTION
    # One detector pixel gets 1 / k^2 of the light and dark of an image pixel.
    t_saturation = full * k**2 / (peak + background)
    t_sub, t_sub_by = t_background, "background"
    if t_readout > t_sub:
        t_sub, t_sub_by = t_readout, "readout"
    if t_saturation < t_sub:
        t_sub, t_sub_by = t_saturation, "saturation"
    if t_sub > props.get("max_exp", math.inf):
        t_sub, t_sub_by = props["max_exp"], "max_exp"
    if t_sub < props.get("min_exp", 0.0):
        t_sub, t_sub_by = props["min_exp"], "min_exp"

    def one_sub(t: float) -> tuple[float, float, float]:
        """Signal, variance and flat-field error of one sub-exposure."""
        return (
            npix * signal * t,
            npix * ((signal + sky + dark) * t + read_noise**2),
            flat_error * npix * sky * t,
        )

    # n sub-exposures give signal n*s, variance n*v and flat-field error n*e.
    # The SNR is n*s / sqrt(n*v + (n*e)^2). It gets the requested value when
    # n = snr^2 v / (s^2 - snr^2 e^2), and it cannot be more than s/e.
    snr_limit = signal / (flat_error * sky) if flat_error > 0 and sky > 0 else math.inf
    subs = exposure = wall_time = None
    if snr < snr_limit:
        s, v, e = one_sub(t_sub)
        subs = math.ceil(snr**2 * v / (s**2 - snr**2 * e**2))
        if subs == 1:
            # One sub-exposure is sufficient. Its time comes from the same
            # condition, written as a t^2 = b t + c.
            a = npix**2 * (signal**2 - snr**2 * (flat_error * sky) ** 2)
            b = snr**2 * npix * (signal + sky + dark)
            c = snr**2 * npix * read_noise**2
            t_one = (b + math.sqrt(b**2 + 4 * a * c)) / (2 * a)
            if max(t_one, props.get("min_exp", 0.0)) < t_sub:
                t_sub = max(t_one, props.get("min_exp", 0.0))
                t_sub_by = "SNR"
                warnings.append(
                    f"One exposure of {_seconds(t_sub)} gives the SNR. More "
                    "sub-exposures add readout time and read noise. Use max_exp to "
                    "limit the exposure time."
                )
        exposure = subs * t_sub
        wall_time = subs * (t_sub + read_time)

        if t_sub_by == "saturation":
            cost = (
                "the read noise adds to the noise"
                if t_saturation < t_background
                else "the readout uses more than 10% of the time"
            )
            warnings.append(
                "The brightest part fills the well to well_fill after "
                f"{_seconds(t_saturation)}. Thus the sub-exposures are short, and "
                f"{cost}."
            )
        slowdown = 1 / (1 - (snr / snr_limit) ** 2)
        if slowdown > 1.5:
            warnings.append(
                f"The flat-field error makes the exposure {slowdown:.3g} times longer."
            )
    else:
        warnings.append(
            f"SNR {snr:g} is not possible. The flat-field error limits the SNR to "
            f"{snr_limit:.3g} at this surface brightness. Use a smaller flat_error "
            "or a brighter level (a negative mu_offset)."
        )

    s, v, e = one_sub(t_sub)
    snr_sub = s / math.sqrt(v + e**2)

    # The brightest star that stays below the well fill in one sub-exposure.
    saturation_magnitude = None
    if seeing is not None:
        central = _peak_pixel_rate(seeing, 1.0, 0.0, 0.0, props["plate_scale"])
        star_rate = (full / t_sub - background / k**2) / central  # e/s, whole star
        if star_rate > 0:
            saturation_magnitude = -2.5 * math.log10(
                star_rate / (collecting_area * vega)
            )

    def magnitude(electrons_per_pixel: float) -> float:
        return -2.5 * math.log10(electrons_per_pixel / per_pixel / vega)

    mu_sky = magnitude(sky)
    return {
        "target": target.label,
        "kind": target.kind,
        "instrument": props["name"],
        "ra [deg]": target.ra,
        "dec [deg]": target.dec,
        "major_axis [arcmin]": target.major_axis,
        "minor_axis [arcmin]": target.minor_axis,
        "area [arcsec2]": target.area,
        "spectrum": source["spectrum"],
        "scaled to": source["scaled to"],
        "mu_mean [mag/arcsec2]": magnitude(mean),
        "mu_target [mag/arcsec2]": magnitude(signal),
        "mu_peak [mag/arcsec2]": magnitude(peak),
        "mu_sky [mag/arcsec2]": mu_sky,
        "N_target [e/pix/s]": signal,
        "N_peak [e/pix/s]": peak,
        "N_sky [e/pix/s]": sky,
        "N_dc [e/pix/s]": dark,
        "N_rn [e_rms/pix]": read_noise,
        "line counts [e/pix/s]": {
            label: value * per_pixel * 10 ** (-0.4 * mu_offset)
            for label, value in source["lines"].items()
        },
        "pixels in area": npix,
        "t_sub [s]": t_sub,
        "t_sub set by": t_sub_by,
        "t_background [s]": t_background if math.isfinite(t_background) else None,
        "t_saturation [s]": t_saturation,
        "saturation magnitude [mag]": saturation_magnitude,
        "SNR per sub": snr_sub,
        "SNR target": snr,
        "SNR limit": snr_limit if math.isfinite(snr_limit) else None,
        "subs": subs,
        "exposure [s]": exposure,
        "wall time [s]": wall_time,
        "pwv [mm]": pwv,
        "airmass": airmass,
        "seeing [arcsec]": seeing,
        "altitude [m]": h,
        'plate_scale ["/pix]': plate_scale,
        "pixel_binning": k,
        "sky_factor": props_sky.get("sky_factor", 1.0),
        "A [m2]": collecting_area,
        "read_time [s]": read_time,
        "flat_error": flat_error,
        "warnings": warnings,
    }
