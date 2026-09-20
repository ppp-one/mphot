"""Precision grids over PWV, airmass and stellar effective temperature."""

import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from scipy.interpolate import griddata

from mphot.constants import (
    AIRMASS_VALUES,
    GRID_FLUX_INGREDIENTS_NAME,
    GRID_RADIANCE_INGREDIENTS_NAME,
    PWV_VALUES,
    TEFF_VALUES,
    WAVELENGTHS,
)
from mphot.paths import DATAFILES_DIR, grid_path, system_response_path
from mphot.utils import interpolate_dfs, update_progress


def _grid_coords() -> np.ndarray:
    """
    Build the coordinate array of a precision grid.

    Returns:
        np.ndarray: Array of shape
            (len(PWV_VALUES), len(AIRMASS_VALUES), len(TEFF_VALUES), 3) holding
            the PWV, airmass and temperature of every grid point.
    """

    shape = (len(PWV_VALUES), len(AIRMASS_VALUES), len(TEFF_VALUES))
    coords = np.zeros((*shape, 3))
    coords[..., 0] = PWV_VALUES.reshape((shape[0], 1, 1))
    coords[..., 1] = AIRMASS_VALUES.reshape((1, shape[1], 1))
    coords[..., 2] = TEFF_VALUES.reshape((1, 1, shape[2]))

    return coords


def _generate_grid(
    sResponse: str, ingredients_name: str, weight_by_star: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the system response against the atmosphere over the whole grid.

    Args:
        sResponse (str): Path to the CSV file containing the spectral response function.
        ingredients_name (str): File name of the precomputed ingredients pickle.
        weight_by_star (bool): If True, also weight the integrand by the stellar
            spectrum of each grid temperature. Stellar flux needs this, sky
            radiance does not.

    Returns:
        tuple: The coords and data arrays described by `generate_flux_grid`.
    """

    gridIngredients = pd.read_pickle(DATAFILES_DIR / ingredients_name)
    rsr = pd.read_csv(sResponse, header=None, index_col=0)

    gridSauce = interpolate_dfs(WAVELENGTHS, rsr, gridIngredients)
    gridSauce = gridSauce[(gridSauce[1] > 0)]

    # Pull the columns out as plain arrays. Every pandas operation rebuilds a
    # Series and realigns its index, which costs more than the integral does.
    lam = gridSauce.index.to_numpy()
    system_response = gridSauce[1].to_numpy()

    if weight_by_star:
        # One row per grid temperature, so all of them integrate in one call.
        star_spectra = np.array(
            [
                gridSauce[str(temperature) + "K"].to_numpy()
                for temperature in TEFF_VALUES
            ]
        )

    data = np.zeros((len(PWV_VALUES), len(AIRMASS_VALUES), len(TEFF_VALUES)))

    for i, pwv in enumerate(PWV_VALUES):
        update_progress(i / (len(PWV_VALUES) - 1))
        for j, airmass in enumerate(AIRMASS_VALUES):
            atmosphere = gridSauce[str(pwv) + "_" + str(airmass)].to_numpy()
            response = system_response * atmosphere

            if weight_by_star:
                data[i, j] = simps(y=response * star_spectra, x=lam, axis=-1)
            else:
                # Sky radiance carries no stellar weight, so the integrand does
                # not depend on temperature. One integral fills the whole axis.
                data[i, j] = simps(y=response, x=lam)

    return _grid_coords(), data


def generate_flux_grid(sResponse: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a base flux grid based on atmospheric parameters and response functions, with the following ranges:
        airmass: 1 - 3
        pwv: 0.05 - 30 mm
        Teff: 450 - 36500 K

    This function reads in a spectral response file and a precomputed grid of flux ingredients,
    then interpolates and integrates these data to produce a grid of stellar flux responses
    for various combinations of precipitable water vapor (PWV), airmass, and temperature values.

    Args:
        sResponse (str): Path to the CSV file containing the spectral response function.

    Returns:
        tuple: A tuple containing:
            - coords (np.ndarray): A 4D array with shape
              representing the coordinates of the grid points. The last dimension
              contains the values of PWV, airmass, and temperature respectively.
            - data (np.ndarray): A 3D array with shape
              (len(pwv_values), len(airmass_values), len(temperature_values))
              containing the computed flux values for each combination of PWV,
              airmass, and temperature.
    """

    return _generate_grid(sResponse, GRID_FLUX_INGREDIENTS_NAME, weight_by_star=True)


def generate_radiance_grid(sResponse: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a radiance base grid for atmospheric parameters, with the following ranges:
        airmass: 1 - 3
        pwv: 0.05 - 30 mm
        Teff: 450 - 36500 K

    This function reads in a spectral response file and a precomputed grid of radiance ingredients,
    then interpolates and integrates these data to produce a grid of atmospheric flux responses
    for various combinations of precipitable water vapor (PWV), airmass, and temperature values.

    Args:
        sResponse (str): Path to the spectral response CSV file.

    Returns:
        tuple: A tuple containing:
            - coords (np.ndarray): A 4D array of shape (len(pwv_values), len(airmass_values), len(temperature_values), 3)
              containing the coordinates for PWV, airmass, and temperature.
            - data (np.ndarray): A 3D array of shape (len(pwv_values), len(airmass_values), len(temperature_values))
              containing the integrated atmospheric flux responses.
    """

    return _generate_grid(
        sResponse, GRID_RADIANCE_INGREDIENTS_NAME, weight_by_star=False
    )


def generate_grids(name: str) -> None:
    """
    Build and save both precision grids of a named instrument.

    Args:
        name (str): Name of the instrument, as returned by `generate_system_response`.

    Returns:
        None
    """

    sResponse = system_response_path(name)

    for kind, generate in (
        ("flux", generate_flux_grid),
        ("radiance", generate_radiance_grid),
    ):
        coords, data = generate(sResponse)
        np.save(grid_path(name, kind, "coords"), coords)
        np.save(grid_path(name, kind, "data"), data)


def load_grids(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the saved precision grids of a named instrument.

    Args:
        name (str): Name of the instrument.

    Returns:
        tuple: A tuple containing:
            - coords (np.ndarray): Coordinates of the grid points, shared by
              both grids.
            - data_flux (np.ndarray): Stellar flux grid.
            - data_radiance (np.ndarray): Sky radiance grid.
    """

    return (
        np.load(grid_path(name, "flux", "coords")),
        np.load(grid_path(name, "flux", "data")),
        np.load(grid_path(name, "radiance", "data")),
    )


def interpolate_grid(
    coords: np.ndarray, data: np.ndarray, pwv: float, airmass: float, Teff: float
) -> float:
    """
    Interpolates between grid points, using a cubic method.

    Args:
        coords (np.ndarray): Coordinates of base grid generated.
        data (np.ndarray): Data of base grid generated.
        pwv (float): Precipitable water vapour value at zenith.
        airmass (float): Airmass of target/comparison star.
        Teff (float): Effective temperature of target/comparison star.

    Returns:
        float: Interpolated value of grid.
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

    # griddata returns a 0-d array. Weighting it above turns it into a numpy
    # scalar, but the branch that hits a grid temperature exactly does not, so
    # the return type used to depend on Teff. Callers were promised a float.
    return float(interp)
