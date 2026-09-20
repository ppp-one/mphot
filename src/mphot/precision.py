"""Photometric precision budget of an observation."""

import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps

from mphot.constants import (
    AIRMASS_VALUES,
    GRID_FLUX_INGREDIENTS_NAME,
    PC,
    PWV_VALUES,
    VEGA_FILE,
    WAVELENGTHS,
)
from mphot.grid import generate_grids, interpolate_grid, load_grids
from mphot.paths import DATAFILES_DIR, grid_path, system_response_path
from mphot.utils import gaussian, interpolate_dfs


def _peak_pixel_rate(
    fwhm: float, N_star: float, N_sky: float, N_dc: float, plate_scale: float
) -> float:
    """
    Electron rate of the brightest pixel of the point spread function.

    Args:
        fwhm (float): Full width at half maximum of the point spread function.
        N_star (float): Number of star counts.
        N_sky (float): Number of sky counts.
        N_dc (float): Number of dark current counts.
        plate_scale (float): Plate scale in arcseconds per pixel.

    Returns:
        float: Electrons per second collected by the central pixel.
    """

    sigma_IR = (fwhm / plate_scale) / 2.355  # in pix

    x = np.linspace(-0.5, 0.5, 100)
    y = x

    return N_star * simps(y=gaussian(y, sigma_IR), x=y) * simps(
        y=gaussian(x, sigma_IR), x=x
    ) + (N_sky + N_dc)


def integration_time(
    fwhm: float,
    N_star: float,
    N_sky: float,
    N_dc: float,
    plate_scale: float,
    well_depth: float,
    well_fill: float,
) -> float:
    """
    Calculate the integration time for a given set of parameters.

    Args:
        fwhm (float): Full width at half maximum of the point spread function.
        N_star (float): Number of star counts.
        N_sky (float): Number of sky counts.
        N_dc (float): Number of dark current counts.
        N_rn (float): Number of read noise counts.
        plate_scale (float): Plate scale in arcseconds per pixel.
        well_depth (float): Maximum well depth of the detector.
        well_fill (float): Fraction of the well depth to be filled.

    Returns:
        float: Calculated integration time.
    """

    return (well_depth * well_fill) / _peak_pixel_rate(
        fwhm, N_star, N_sky, N_dc, plate_scale
    )


def convert_airmass(airmass: float, h: float) -> float:
    """
    Convert airmass at the observatory to an equivalent airmass at Paranal Observatory, assuming an isothermal atmospheric model.

    Args:
        airmass (float): Airmass at the observatory.
        h (float): Altitude of the observatory in meters.

    Returns:
        float: The equivalent converted airmass at Paranal Observatory.

    Reference:
        https://acp.copernicus.org/articles/7/6047/2007/
    """

    return airmass * np.exp((2440 - h) / 8000)


def scintillation_noise(
    r: float,
    t: float,
    N_star: float,
    h: float = 2440,
    C: float = 1.56,
    airmass: float = 1.5,
) -> float:
    """
    Calculate the scintillation noise for a given set of parameters.

    Args:
        r (float): Aperture radius in meters.
        t (float): Exposure time in seconds.
        N_star (float): Number of stars.
        h (float): Altitude of the observatory in meters. Default is 2440 for Paranal Observatory.
        C (float): Empirical coefficient. Default is 1.56, optimized for the 20-cm NGTS telescopes at Paranal Observatory.
        airmass (float, optional): Airmass value. Default is 1.5.

    Returns:
        float: The calculated scintillation noise.

    Reference:
        https://academic.oup.com/mnras/article/509/4/6111/6442285
    """

    return (
        np.sqrt(
            1e-5
            * C**2
            * pow(2 * r, -4 / 3)
            * t**-1
            * airmass**3
            * np.exp(-2 * h / 8000)
        )
        * N_star
        * t
    )


def get_precision(
    props: dict,
    props_sky: dict,
    Teff: float,
    distance: float,
    binning: float = 10,
    override_grid: bool = False,
    N_sky: float | None = None,
    N_star: float | None = None,
    scn: float | None = None,
    h: float = 2440,
    C: float = 1.56,
    exp_time: float | None = None,
) -> dict:
    """
    Calculate the precision of astronomical observations based on various parameters.

    Args:
        props (dict):
            Dictionary containing properties of the instrument and observation.
            Expected keys:
            - "name": str, name of the instrument
            - "plate_scale": float, plate scale of the instrument
            - "N_dc": float, dark current noise
            - "N_rn": float, read noise
            - "well_depth": float, well depth of the detector
            - "well_fill": float, well fill level
            - "read_time": float, readout time of the detector
            - "r0": float, inner radius for aperture
            - "r1": float, outer radius for aperture
            - "ap_rad": float, optional, aperture radius

        props_sky (dict):
            Dictionary containing properties of the sky.
            Expected keys:
            - "pwv": float, precipitable water vapor
            - "airmass": float, airmass of the observation
            - "seeing": float, full width at half maximum (FWHM) of the seeing

        Teff (float):
            Effective temperature of the star in Kelvin.

        distance (float):
            Distance to the star in parsecs.

        binning (float, optional):
            Binning time in minutes. Default is 10.

        override_grid (bool, optional):
            If True, override existing grid files. Default is False.

        N_sky (float, optional):
            Number of sky counts, calculated if None. Default is None.

        N_star (float, optional):
            Number of star counts, calculated if None. Default is None.

        scn (float, optional):
            Scintillation noise, calculated if None. Default is None.

        h (float, optional):
            Altitude of the observatory in meters. Default is 2440 for Paranal Observatory.

        C (float, optional):
            Empirical coefficient used in the calculation of scn. Default is 1.56, optimized for the 20-cm NGTS telescopes at Paranal Observatory.

        exp_time (float, optional):
            Exposure time in seconds, calculated if None. Default is None.

    Returns:
        tuple: A tuple containing:
            image_precision : dict
                Precision of the image
            binned_precision : dict
                Precision of the binned image
            components : dict
                Various components used in the calculation
    """

    props = props.copy()
    props_sky = props_sky.copy()

    name = props["name"]
    plate_scale = props["plate_scale"]
    N_dc = props["N_dc"]
    N_rn = props["N_rn"]
    well_depth = props["well_depth"]
    well_fill = props["well_fill"]
    read_time = props["read_time"]

    if "min_exp" in props:
        min_exp = props["min_exp"]
    else:
        min_exp = 0

    if "max_exp" in props:
        max_exp = props["max_exp"]
    else:
        max_exp = np.inf

    r0 = props["r0"]
    r1 = props["r1"]

    pwv = props_sky["pwv"]
    airmass = props_sky["airmass"]
    fwhm = props_sky["seeing"]

    airmass_paranal = convert_airmass(airmass, h)

    ap = (
        3 * (fwhm / plate_scale)
    )  ## approx pixel radius around target star ## changed on to 3* 2022/04/26 from 10/2.355*

    if "ap_rad" in props:
        ap = props["ap_rad"] * (fwhm / plate_scale)

    if override_grid or not grid_path(name, "flux", "coords").is_file():
        generate_grids(name)

    coords, data_flux, data_radiance = load_grids(name)

    # get values from grids
    flux = interpolate_grid(coords, data_flux, pwv, airmass_paranal, Teff)
    radiance = interpolate_grid(coords, data_radiance, pwv, airmass_paranal, Teff)

    # collecting area of telescope
    A = np.pi * (r0**2 - r1**2)

    if N_star is None:
        N_star = flux * A / ((distance * PC) ** 2)
    else:
        flux = N_star * ((distance * PC) ** 2) / A

    if N_sky is None:
        N_sky = radiance * A * plate_scale**2
    else:
        radiance = N_sky / (A * plate_scale**2)

    t = integration_time(
        fwhm,
        N_star,
        N_sky,
        N_dc,
        plate_scale,
        well_depth,
        well_fill,
    )

    if exp_time is not None or (t < min_exp or t > max_exp):
        if t < min_exp:
            t = min_exp
        elif t > max_exp:
            t = max_exp

        if exp_time is not None:
            t = exp_time

        well_fill_value = t * _peak_pixel_rate(fwhm, N_star, N_sky, N_dc, plate_scale)
        well_fill = well_fill_value / well_depth

    npix = np.pi * ap**2

    if scn is None:
        scn = scintillation_noise(
            r0, t, N_star, h=h, C=C, airmass=airmass
        )  # use unconverted airmass here

    precision = np.sqrt(
        N_star * t + scn**2 + npix * (N_sky * t + N_dc * t + N_rn**2)
    ) / (N_star * t)

    precision_star = 1 / np.sqrt(N_star * t)
    precision_scn = np.sqrt(scn**2) / (N_star * t)
    precision_sky = np.sqrt(npix * (N_sky * t)) / (N_star * t)
    precision_dc = np.sqrt(npix * (N_dc * t)) / (N_star * t)
    precision_rn = np.sqrt(npix * (N_rn**2)) / (N_star * t)

    image_precision = {
        "All": precision,
        "Star": precision_star,
        "Scintillation": precision_scn,
        "Sky": precision_sky,
        "Dark current": precision_dc,
        "Read noise": precision_rn,
    }

    nImages = (binning * 60) / (t + read_time)

    binned_precision = {
        "All": precision / np.sqrt(nImages),
        "Star": precision_star / np.sqrt(nImages),
        "Scintillation": precision_scn / np.sqrt(nImages),
        "Sky": precision_sky / np.sqrt(nImages),
        "Dark current": precision_dc / np.sqrt(nImages),
        "Read noise": precision_rn / np.sqrt(nImages),
    }

    components = {
        "name": name,
        "Teff [K]": Teff,
        "distance [pc]": distance,
        "N_star [e/s]": N_star,
        "star_flux [e/m2/s]": flux / ((distance * PC) ** 2),
        "scn [e_rms]": scn,  # not sure of units
        "pixels in aperture [pix]": npix,
        "ap_radius [pix]": ap,
        "N_sky [e/pix/s]": N_sky,
        "sky_radiance [e/m2/arcsec2/s]": radiance,
        "seeing [arcsec]": fwhm,
        "pwv [mm]": pwv,
        "airmass": airmass,  # unconverted airmass
        'plate_scale ["/pix]': plate_scale,
        "N_dc [e/pix/s]": N_dc,
        "N_rn [e_rms/pix]": N_rn,  # not sure of units
        "A [m2]": A,
        "r0 [m]": r0,
        "r1 [m]": r1,
        "t [s]": t,
        "well_depth [e/pix]": well_depth,
        "peak well_fill": well_fill,  # peak pixel
        "binning [mins]": binning,
        "read_time [s]": read_time,
        "binned images": nImages,
        "altitude [m]": h,
    }

    return image_precision, binned_precision, components


def vega_mag(
    SRFile: str,
    props_sky: dict,
    N_star: float,
    sky_radiance: float,
    A: float,
) -> dict:
    """
    Calculate the Vega magnitude for a given spectral response file and sky properties.

    Args:
        SRFile (str):
            Path to the spectral response CSV file.
        props_sky (dict):
            Dictionary containing properties of the sky.
            Expected keys:
            - "pwv": float, precipitable water vapor
            - "airmass": float, airmass of the observation
        N_star (float):
            Number of star counts.
        sky_radiance (float):
            Sky radiance value.
        A (float):
            Aperture area in square meters.

    Returns:
        dict:
            A dictionary containing the Vega magnitude information:
            - "star [mag]": Vega magnitude of the star.
            - "sky [mag/arcsec2]": Vega magnitude of the sky per arcsecond squared.
            - "vega_flux [e/s]": Vega flux in electrons per second.
    """

    gridIngredients = pd.read_pickle(DATAFILES_DIR / GRID_FLUX_INGREDIENTS_NAME)
    vega = pd.read_csv(DATAFILES_DIR / VEGA_FILE, header=None, index_col=0)

    rsr = pd.read_csv(SRFile, header=None, index_col=0)
    rsr = rsr[1].rename("rsr")

    vega = vega[1].rename("vega")

    gridSauce = interpolate_dfs(WAVELENGTHS, rsr, gridIngredients, vega)

    gridSauce = gridSauce[(gridSauce["rsr"] > 0)]

    pwv = props_sky["pwv"]
    airmass = props_sky["airmass"]

    # lazy way to get atmosphere profile
    pwv = min(PWV_VALUES, key=lambda x: abs(x - pwv))
    airmass = min(AIRMASS_VALUES, key=lambda x: abs(x - airmass))

    atmosphere_trans = gridSauce[str(pwv) + "_" + str(airmass)]

    simStar = gridSauce["vega"]

    vega = simps(
        y=gridSauce["rsr"] * atmosphere_trans * simStar, x=gridSauce.index
    )  # e/s/m2

    vega_dict = {
        "star [mag]": -2.5 * np.log10(N_star / (vega * A)),
        "sky [mag/arcsec2]": -2.5 * np.log10(sky_radiance / vega),
        "vega_flux [e/s]": vega * A,
    }

    return vega_dict
