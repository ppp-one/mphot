"""
Backwards-compatible re-exports.

The code used to live in this one module. It is now split by subject, and this
module keeps `from mphot.core import ...` working. New code should import from
the module that owns the name, or from `mphot` itself.
"""

from mphot.constants import (
    AIRMASS_VALUES,
    GRID_FLUX_INGREDIENTS_NAME,
    GRID_RADIANCE_INGREDIENTS_NAME,
    PC,
    PWV_VALUES,
    TEFF_VALUES,
    VEGA_FILE,
    WAVELENGTHS,
)
from mphot.display import display_number, display_results
from mphot.gaia import (
    DEFAULT_GAIA_TAP_SOURCES,
    DEFAULT_GAIA_TIMEOUT,
    GAIA_TAP_URLS,
    GaiaSourceNotFound,
    best_gaia_filters,
    get_precision_gaia,
    query_gaia_source,
)
from mphot.grid import (
    generate_flux_grid,
    generate_grids,
    generate_radiance_grid,
    interpolate_grid,
    load_grids,
)
from mphot.paths import grid_path, system_response_path
from mphot.precision import (
    convert_airmass,
    get_precision,
    integration_time,
    scintillation_noise,
    vega_mag,
)
from mphot.response import generate_system_response
from mphot.utils import gaussian, interpolate_dfs, update_progress

# Old lowercase names for the module-level constants.
grid_flux_ingredients_name = GRID_FLUX_INGREDIENTS_NAME
grid_radiance_ingredients_name = GRID_RADIANCE_INGREDIENTS_NAME
vega_file = VEGA_FILE
wavelengths = WAVELENGTHS
pc = PC

__all__ = [
    "AIRMASS_VALUES",
    "DEFAULT_GAIA_TAP_SOURCES",
    "DEFAULT_GAIA_TIMEOUT",
    "GAIA_TAP_URLS",
    "GRID_FLUX_INGREDIENTS_NAME",
    "GRID_RADIANCE_INGREDIENTS_NAME",
    "PC",
    "PWV_VALUES",
    "TEFF_VALUES",
    "VEGA_FILE",
    "WAVELENGTHS",
    "GaiaSourceNotFound",
    "best_gaia_filters",
    "convert_airmass",
    "display_number",
    "display_results",
    "gaussian",
    "generate_flux_grid",
    "generate_grids",
    "generate_radiance_grid",
    "generate_system_response",
    "get_precision",
    "get_precision_gaia",
    "grid_path",
    "integration_time",
    "interpolate_dfs",
    "interpolate_grid",
    "load_grids",
    "query_gaia_source",
    "scintillation_noise",
    "system_response_path",
    "update_progress",
    "vega_mag",
]
