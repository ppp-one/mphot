"""Locations of the data files that ship with the package."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).parent

DATAFILES_DIR = PACKAGE_DIR / "datafiles"
"""Spectra and atmosphere ingredients shipped with the package."""

SYSTEM_RESPONSES_DIR = DATAFILES_DIR / "system_responses"
"""System response curves, written by `generate_system_response`."""

FLUX_CALIBRATION_DIR = DATAFILES_DIR / "flux_calibration"
"""Gaia filter transmission curves used for flux calibration."""

GRIDS_DIR = PACKAGE_DIR / "grids"
"""Precision grids, written the first time a system response is used."""


def system_response_path(name: str) -> Path:
    """
    Path of the system response file of a named instrument.

    Args:
        name (str): Name of the instrument, as returned by `generate_system_response`.

    Returns:
        Path: Path of the system response CSV file.
    """

    return SYSTEM_RESPONSES_DIR / f"{name}_instrument_system_response.csv"


def grid_path(name: str, kind: str, part: str) -> Path:
    """
    Path of one precision grid file.

    Args:
        name (str): Name of the instrument.
        kind (str): Either "flux" or "radiance".
        part (str): Either "coords" or "data".

    Returns:
        Path: Path of the grid .npy file.
    """

    if kind not in ("flux", "radiance"):
        raise ValueError(f"kind must be 'flux' or 'radiance', got {kind!r}.")
    if part not in ("coords", "data"):
        raise ValueError(f"part must be 'coords' or 'data', got {part!r}.")

    return GRIDS_DIR / f"{name}_precisionGrid_{kind}_{part}.npy"
