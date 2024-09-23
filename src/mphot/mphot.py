import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from scipy.integrate import simpson as simps
from scipy.interpolate import griddata

gridFluxIngredientsName = "prePrecisionGrid_2400m_flux.pkl"
gridRadianceIngredientsName = "prePrecisionGrid_2400m_radiance.pkl"
pc = 3.0857e16


def interpolate_dfs(index, *data):
    """
    Interpolates panda dataframes onto an index, of same index type (e.g. wavelength in microns)

    Parameters
    ----------
    index: 1d array which data is to be interpolated onto
    data:       Pandas dataframes

    Returns
    -------
    df: Interpolated dataframe

    """

    df = pd.DataFrame({"tmp": index}, index=index)
    for dat in data:
        dat = dat[~dat.index.duplicated(keep="first")]
        df = pd.concat([df, dat], axis=1)
    df = df.sort_index()
    df = df.interpolate("index").reindex(index)
    df.drop("tmp", axis=1, inplace=True)

    return df


def generateFluxBase(sResponse: str) -> tuple:
    """
    Generates the flux grid base for Paranal, Chile. Takes a few minutes.

    Generates a base grid for:
    airmass: 1 - 3
    pwv: 0.05 - 30 mm
    Teff: 450 - 36500 K

    See arrays for base resolutions

    Ref: ...

    Parameters
    ----------
    sResponse:  csv file with two (unlabelled) columns, wavelength (in microns), system spectral response curves of telescope + filter + camera (as fraction).

    Returns
    -------
    coords, data: coordinates and data of base grid generated.

    """

    gridIngredients = pd.read_pickle(
        Path(__file__).parent / "datafiles" / gridFluxIngredientsName
    )
    rsr = pd.read_csv(sResponse, header=None, index_col=0)

    pwv_values = np.array(
        [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0]
    )
    airmass_values = np.array(
        [
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            2.6,
            2.7,
            2.8,
            2.9,
            3.0,
        ]
    )
    temperature_values = np.array(
        [
            450,
            500,
            550,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            2000,
            2100,
            2250,
            2320,
            2400,
            2440,
            2500,
            2600,
            2650,
            2710,
            2850,
            3000,
            3030,
            3100,
            3200,
            3250,
            3410,
            3500,
            3550,
            3650,
            3700,
            3800,
            3870,
            3940,
            4000,
            4070,
            4190,
            4230,
            4330,
            4410,
            4540,
            4600,
            4700,
            4830,
            4990,
            5040,
            5140,
            5170,
            5240,
            5280,
            5340,
            5490,
            5530,
            5590,
            5660,
            5680,
            5720,
            5770,
            5880,
            5920,
            6000,
            6060,
            6170,
            6240,
            6340,
            6510,
            6640,
            6720,
            6810,
            7030,
            7220,
            7440,
            7500,
            7800,
            8000,
            8080,
            8270,
            8550,
            8840,
            9200,
            9700,
            10400,
            10700,
            12500,
            14000,
            14500,
            15700,
            16700,
            17000,
            18500,
            20600,
            24500,
            26000,
            29000,
            31500,
            32000,
            32500,
            33000,
            34500,
            35000,
            36500,
        ]
    )

    wavelengths = np.arange(0.5, 2, 0.0001)

    gridSauce = interpolate_dfs(wavelengths, rsr, gridIngredients)
    gridSauce = gridSauce[(gridSauce[1] > 0)]
    atm_grid = []
    for i, pwv in enumerate(pwv_values):
        update_progress(i / (len(pwv_values) - 1))
        for airmass in airmass_values:
            for temperature in temperature_values:
                atmosphere_trans = gridSauce[str(pwv) + "_" + str(airmass)]
                simStar = gridSauce[str(temperature) + "K"]
                response = simps(
                    y=gridSauce[1] * atmosphere_trans * simStar, x=gridSauce.index
                )

                atm_grid.append((pwv, airmass, temperature, response))

    data = np.array([x[3] for x in atm_grid])
    data = data.reshape((len(pwv_values), len(airmass_values), len(temperature_values)))

    coords = np.zeros(
        (len(pwv_values), len(airmass_values), len(temperature_values), 3)
    )
    coords[..., 0] = pwv_values.reshape((len(pwv_values), 1, 1))
    coords[..., 1] = airmass_values.reshape((1, len(airmass_values), 1))
    coords[..., 2] = temperature_values.reshape((1, 1, len(temperature_values)))

    return coords, data


def generateRadianceBase(sResponse):
    """
    Generates the sky radiance grid base for Paranal, Chile. Takes a few minutes.

    Generates a base grid for:
    airmass: 1 - 3
    pwv: 0.05 - 30 mm
    Teff: 450 - 36500 K

    See arrays for base resolutions

    Ref: ...

    Parameters
    ----------
    sResponse:  csv file with two (unlabelled) columns, wavelength (in microns), system spectral response curves of telescope + filter + camera (as fraction).

    Returns
    -------
    coords, data: coordinates and data of base grid generated.

    """

    gridIngredients = pd.read_pickle(
        Path(__file__).parent / "datafiles" / gridRadianceIngredientsName
    )
    rsr = pd.read_csv(sResponse, header=None, index_col=0)

    pwv_values = np.array(
        [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0]
    )
    airmass_values = np.array(
        [
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            2.6,
            2.7,
            2.8,
            2.9,
            3.0,
        ]
    )
    temperature_values = np.array(
        [
            450,
            500,
            550,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            2000,
            2100,
            2250,
            2320,
            2400,
            2440,
            2500,
            2600,
            2650,
            2710,
            2850,
            3000,
            3030,
            3100,
            3200,
            3250,
            3410,
            3500,
            3550,
            3650,
            3700,
            3800,
            3870,
            3940,
            4000,
            4070,
            4190,
            4230,
            4330,
            4410,
            4540,
            4600,
            4700,
            4830,
            4990,
            5040,
            5140,
            5170,
            5240,
            5280,
            5340,
            5490,
            5530,
            5590,
            5660,
            5680,
            5720,
            5770,
            5880,
            5920,
            6000,
            6060,
            6170,
            6240,
            6340,
            6510,
            6640,
            6720,
            6810,
            7030,
            7220,
            7440,
            7500,
            7800,
            8000,
            8080,
            8270,
            8550,
            8840,
            9200,
            9700,
            10400,
            10700,
            12500,
            14000,
            14500,
            15700,
            16700,
            17000,
            18500,
            20600,
            24500,
            26000,
            29000,
            31500,
            32000,
            32500,
            33000,
            34500,
            35000,
            36500,
        ]
    )

    wavelengths = np.arange(0.5, 2, 0.0001)

    gridSauce = interpolate_dfs(wavelengths, rsr, gridIngredients)
    gridSauce = gridSauce[(gridSauce[1] > 0)]
    atm_grid = []
    for i, pwv in enumerate(pwv_values):
        update_progress(i / (len(pwv_values) - 1))
        for airmass in airmass_values:
            for temperature in temperature_values:
                atmosphere_flux = gridSauce[str(pwv) + "_" + str(airmass)]
                response = simps(y=gridSauce[1] * atmosphere_flux, x=gridSauce.index)

                atm_grid.append((pwv, airmass, temperature, response))

    data = np.array([x[3] for x in atm_grid])
    data = data.reshape((len(pwv_values), len(airmass_values), len(temperature_values)))

    coords = np.zeros(
        (len(pwv_values), len(airmass_values), len(temperature_values), 3)
    )
    coords[..., 0] = pwv_values.reshape((len(pwv_values), 1, 1))
    coords[..., 1] = airmass_values.reshape((1, len(airmass_values), 1))
    coords[..., 2] = temperature_values.reshape((1, 1, len(temperature_values)))

    return coords, data


def gaus(delta, sigma):
    """
    Generate Gaussian

    Ref: ...

    Parameters
    ----------
    delta: x/y variable
    sigma: sigma of gaussian

    Returns
    -------
    value: gaussian value

    """

    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(delta**2) / (2 * sigma**2))


def int_time(
    fwhm, N_star, N_sky, N_dc, N_rn, plate_scale, well_depth, well_fill, bias_level
):
    """
    Calculate integration time

    Ref: ...

    Parameters
    ----------
    Many detector properties

    Returns
    -------
    time: integration time

    """

    sigma_IR = (fwhm / plate_scale) / 2.355  # in pix

    x = np.linspace(-0.5, 0.5, 100)
    y = x

    t = (well_depth * well_fill - bias_level) / (
        N_star * simps(y=gaus(y, sigma_IR), x=y) * simps(y=gaus(x, sigma_IR), x=x)
        + (N_sky + N_dc)
    )

    return t


def scint(r, t, N_star, airmass=1.5):
    """
    Scintillation noise estimate

    Ref: ...

    Parameters
    ----------
    r: radius of telescope
    t: integration time
    N_star: Flux of star received

    Returns
    -------
    value: scintilation noise

    """

    # https://academic.oup.com/mnras/article/509/4/6111/6442285
    return (
        np.sqrt(
            1e-5
            * 1.56**2
            * pow(2 * r, -4 / 3)
            * t**-1
            * airmass**3
            * np.exp(-2 * 2440 / 8000)
        )
        * N_star
        * t
    )


def get_precision(
    props,
    props_sky,
    Teff,
    distance,
    binning=10,
    override=False,
    fixed_exp=[False, 10],
    SPCcorrection=True,
    mapping=False,
    sky_override=[False, None],
):
    """
    Calculate precision

    Ref: ...

    Parameters
    ----------
    many

    Returns
    -------
    many

    """

    name = props["name"]
    plate_scale = props["plate_scale"]
    N_dc = props["N_dc"]
    N_rn = props["N_rn"]
    well_depth = props["well_depth"]
    well_fill = props["well_fill"]
    bias_level = props["bias_level"]
    read_time = props["read_time"]
    r0 = props["r0"]
    r1 = props["r1"]

    pwv = props_sky["pwv"]
    airmass = props_sky["airmass"]
    fwhm = props_sky["seeing"]

    ap = 3 * (
        fwhm / plate_scale
    )  ## approx pixel radius around target star ## changed on to 3* 2022/04/26 from 10/2.355*

    if props["ap_rad"]:
        ap = props["ap_rad"] * (fwhm / plate_scale)

    if (
        os.path.isfile(
            Path(__file__).parent / "grids" / f"{name}_precisionGrid_flux_coords.npy"
        )
        is False
    ) or (override):
        # generate base of grid
        coords, data = generateFluxBase(
            Path(__file__).parent / "datafiles" / "SRs" / f"{name}_instrumentSR.csv"
        )

        # save output
        np.save(
            Path(__file__).parent / "grids" / f"{name}_precisionGrid_flux_data.npy",
            data,
        )
        np.save(
            Path(__file__).parent / "grids" / f"{name}_precisionGrid_flux_coords.npy",
            coords,
        )

        # generate base of grid
        coords, data = generateRadianceBase(
            Path(__file__).parent / "datafiles" / "SRs" / f"{name}_instrumentSR.csv"
        )

        # save output
        np.save(
            Path(__file__).parent
            / "grids"
            / f"{name}_precisionGrid_radiance_coords.npy",
            coords,
        )
        np.save(
            Path(__file__).parent / "grids" / f"{name}_precisionGrid_radiance_data.npy",
            data,
        )

    coords = np.load(
        Path(__file__).parent / "grids" / f"{name}_precisionGrid_flux_coords.npy"
    )
    data_flux = np.load(
        Path(__file__).parent / "grids" / f"{name}_precisionGrid_flux_data.npy"
    )
    data_radiance = np.load(
        Path(__file__).parent / "grids" / f"{name}_precisionGrid_radiance_data.npy"
    )

    flux = interp(coords, data_flux, pwv, airmass, Teff)
    radiance = interp(coords, data_radiance, pwv, airmass, Teff)

    A = np.pi * (r0**2 - r1**2)

    N_star = flux * A / ((distance * pc) ** 2)
    N_sky = radiance * A * plate_scale**2

    if sky_override[0] and sky_override[1] is None:
        fwhm = 0.3
        N_sky = N_sky * ap / (3 * (fwhm / plate_scale))
        ap = 3 * (fwhm / plate_scale)

    ## correction
    if SPCcorrection:
        if (Teff <= 3042) and (Teff >= 1278):
            poly = np.load(Path(__file__).parent / "datafiles" / "16_order_poly.npy")
            # print(Teff,(2.512**np.polyval(poly, Teff)))
            N_star = N_star / (2.512 ** np.polyval(poly, Teff))

    if fixed_exp[0]:
        t = fixed_exp[1]

        sigma_IR = (fwhm / plate_scale) / 2.355  # in pix

        x = np.linspace(-0.5, 0.5, 100)
        y = x

        well_fill = (
            t
            * (
                N_star
                * simps(y=gaus(y, sigma_IR), x=y)
                * simps(y=gaus(x, sigma_IR), x=x)
                + (N_sky + N_dc)
            )
            + bias_level
        )
        well_fill = well_fill / well_depth
    else:
        t = int_time(
            fwhm,
            N_star,
            N_sky,
            N_dc,
            N_rn,
            plate_scale,
            well_depth,
            well_fill,
            bias_level,
        )
        if t > 120:
            t = 120

            sigma_IR = (fwhm / plate_scale) / 2.355  # in pix

            x = np.linspace(-0.5, 0.5, 100)
            y = x
            well_fill = (
                t
                * (
                    N_star
                    * simps(y=gaus(y, sigma_IR), x=y)
                    * simps(y=gaus(x, sigma_IR), x=x)
                    + (N_sky + N_dc)
                )
                + bias_level
            )
            well_fill = well_fill / well_depth

    npix = np.pi * ap**2

    scn = scint(r0, t, N_star, airmass)

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
        "star_flux [e/m2/s]": flux / ((distance * pc) ** 2),
        "scn [e_rms]": scn,  # not sure of units
        "npix": npix,
        "ap_radius [pix]": ap,
        "N_sky [e/pix/s]": N_sky,
        "sky_radiance [e/m2/arcsec2/s]": radiance * 1,
        'plate_scale ["/pix]': plate_scale,
        "N_dc [e/pix/s]": N_dc,
        "N_rn [e_rms/pix]": N_rn,  # not sure of units
        "A [m2]": A,
        "r0 [m]": r0,
        "r1 [m]": r1,
        "t [s]": t,
        "preset_exp": fixed_exp[0],
        "bias_level": bias_level,
        "well_depth [e/pix]": well_depth,
        "well_fill": well_fill,  # peak pixel
        "binning [mins]": binning,
        "read_time [s]": read_time,
        "nImages": nImages,
    }

    if mapping == False:
        return image_precision, binned_precision, components
    else:
        return {
            "image_precision": image_precision,
            "binned_precision": binned_precision,
            "components": components,
        }


def vega_mag(SRFile, props_sky, N_star, sky_radiance, A):
    """
    Calculate vega magnitude

    Ref: ...

    Parameters
    ----------
    many

    Returns
    -------
    many

    """

    gridIngredients = pd.read_pickle(
        Path(__file__).parent / "datafiles" / gridFluxIngredientsName
    )

    rsr = pd.read_csv(SRFile, header=None, index_col=0)
    rsr = rsr[1].rename("rsr")

    vega = pd.read_csv(
        Path(__file__).parent / "datafiles" / "vega.csv", header=None, index_col=0
    )
    vega = vega[1].rename("vega")

    wavelengths = np.arange(0.5, 2, 0.0001)
    gridSauce = interpolate_dfs(wavelengths, rsr, gridIngredients, vega)
    gridSauce = gridSauce[(gridSauce["rsr"] > 0)]

    pwv_values = np.array(
        [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0]
    )
    airmass_values = np.array(
        [
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            2.6,
            2.7,
            2.8,
            2.9,
            3.0,
        ]
    )

    pwv = props_sky["pwv"]
    airmass = props_sky["airmass"]

    # lazy way to get atmosphere profile
    pwv = min(pwv_values, key=lambda x: abs(x - pwv))
    airmass = min(airmass_values, key=lambda x: abs(x - airmass))

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


def interp(coords, data, pwv, airmass, Teff):
    """
    Interpolates between water grid base points (waterGrid.generateBase(...)), using a cubic method.

    Parameters
    ----------
    coords, data:   coordinates and data of base grid generated.
    pwv:            precipitable water vapour value at zenith
    airmass:        airmass of target/comparison star
    Teff:           effective temperature of target/comparison star

    Returns
    -------
    interp: interpolated value of grid.

    """

    method = "cubic"
    Teffs = coords[..., 2][0, 0]
    Teff_lower = np.max(Teffs[Teffs <= Teff])
    Teff_upper = np.min(Teffs[Teffs >= Teff])

    if Teff_lower == Teff_upper:
        x = coords[..., 0][coords[..., 2] == Teff]  # pwv
        y = coords[..., 1][coords[..., 2] == Teff]  # airmass
        z = data[coords[..., 2] == Teff]  # effect

        interp = griddata(
            (x, y), z, (pwv, airmass), method=method
        )  # interpolated value
    else:
        x_lower = coords[..., 0][coords[..., 2] == Teff_lower]  # pwv
        y_lower = coords[..., 1][coords[..., 2] == Teff_lower]  # airmass
        z_lower = data[coords[..., 2] == Teff_lower]  # effect
        interp_lower = griddata(
            (x_lower, y_lower), z_lower, (pwv, airmass), method=method
        )  # interpolated value lower Teff

        x_upper = coords[..., 0][coords[..., 2] == Teff_upper]  # pwv
        y_upper = coords[..., 1][coords[..., 2] == Teff_upper]  # airmass
        z_upper = data[coords[..., 2] == Teff_upper]  # effect
        interp_upper = griddata(
            (x_upper, y_upper), z_upper, (pwv, airmass), method=method
        )  # interpolated value upper Teff

        w_lower = (Teff_upper - Teff) / (Teff_upper - Teff_lower)  # lower weight
        w_upper = (Teff - Teff_lower) / (Teff_upper - Teff_lower)  # upper weight

        interp = (
            w_lower * interp_lower + w_upper * interp_upper
        )  # final interpolated value

    return interp


def generateSR(efficiencyFile, filterFile):
    """
    Generate spectral response curve from QE and filter profiles
    formatted as microns,fractional value

    Ref: ...

    Parameters
    ----------
    many

    Returns
    -------
    many

    """

    wavelengths = np.arange(0.5, 2, 0.0001)

    eff = pd.read_csv(efficiencyFile, header=None)
    filt = pd.read_csv(filterFile, header=None)

    # name to refer to the generated file
    name = efficiencyFile.split("/")[-1][:-4] + "_" + filterFile.split("/")[-1][:-4]

    # generates a SR, saved locally as 'name1_instrumentSR.csv'
    SRFile = Path(__file__).parent / "datafiles" / "SRs" / f"{name}_instrumentSR.csv"

    effDF = pd.DataFrame({"eff": eff[1].values}, index=eff[0])

    filtDF = pd.DataFrame({"filt": filt[1].values}, index=filt[0])

    df = interpolate_dfs(wavelengths, effDF, filtDF)

    dfSR = df["eff"] * df["filt"]

    dfSR = dfSR[dfSR > 0]

    dfSR.to_csv(SRFile, header=False)

    # print(SRFile.name + " has been saved!")

    return name, dfSR


def update_progress(progress):
    """
    Progress bar

    Ref: ...

    Parameters
    ----------
    many

    Returns
    -------
    many

    """
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100
    )
    print(text)


def to_precision(x, p=3):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.0:
        return "0." + "0" * (p - 1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x / tens)

    if n < math.pow(10, p - 1):
        e = e - 1
        tens = math.pow(10, e - p + 1)
        n = math.floor(x / tens)

    if abs((n + 1.0) * tens - x) <= abs(n * tens - x):
        n = n + 1

    if n >= math.pow(10, p):
        n = n / 10.0
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append("e")
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p - 1):
        out.append(m)
    elif e >= 0:
        out.append(m[: e + 1])
        if e + 1 < len(m):
            out.append(".")
            out.extend(m[e + 1 :])
    else:
        out.append("0.")
        out.extend(["0"] * -(e + 1))
        out.append(m)

    return "".join(out)


def display_results(props_sky, r1, r2):
    """
    Display results in a nice way

    Ref: ...

    Parameters
    ----------
    many

    Returns
    -------
    many

    """

    image_precision1, binned_precision1, components1 = r1
    image_precision2, binned_precision2, components2 = r2

    SRFile1 = (
        Path(__file__).parent
        / "datafiles"
        / "SRs"
        / f"{components1['name']}_instrumentSR.csv"
    )
    SRFile2 = (
        Path(__file__).parent
        / "datafiles"
        / "SRs"
        / f"{components2['name']}_instrumentSR.csv"
    )

    vega1 = vega_mag(
        SRFile1,
        props_sky,
        components1["N_star [e/s]"],
        components1["sky_radiance [e/m2/arcsec2/s]"],
        components1["A [m2]"],
    )
    vega2 = vega_mag(
        SRFile2,
        props_sky,
        components2["N_star [e/s]"],
        components2["sky_radiance [e/m2/arcsec2/s]"],
        components2["A [m2]"],
    )

    pd.set_option("display.float_format", to_precision)

    columns = [["single", "single", "binned", "binned"], ["1", "2", "1", "2"]]
    values = (
        np.c_[
            list(image_precision1.values()),
            list(image_precision2.values()),
            list(binned_precision1.values()),
            list(binned_precision2.values()),
        ]
        * 1000
    )
    display(pd.DataFrame(values, index=image_precision1.keys(), columns=columns))

    columns = [["1", "2"]]

    for k, v in components1.items():
        if (type(v) != str) and (type(v) != bool):
            components1[k] = to_precision(v)

    for k, v in components2.items():
        if (type(v) != str) and (type(v) != bool):
            components2[k] = to_precision(v)

    values = np.c_[list(components1.values()), list(components2.values())]
    display(pd.DataFrame(values, index=components1.keys(), columns=columns))

    columns = [["1", "2"]]
    values = np.c_[list(vega1.values()), list(vega2.values())]
    display(pd.DataFrame(values, index=vega1.keys(), columns=columns))
