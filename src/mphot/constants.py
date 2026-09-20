"""Shared constants: data file names, physical constants and grid axes."""

import numpy as np

# Precomputed atmosphere and stellar spectra used to build the precision grids.
GRID_FLUX_INGREDIENTS_NAME = "pre_grid_03_to_3_microns_2400m_flux.pkl"
GRID_RADIANCE_INGREDIENTS_NAME = "pre_grid_03_to_3_microns_2400m_radiance.pkl"
VEGA_FILE = "vega_03_to_3_microns.csv"

# Wavelength support points of every interpolated spectrum [microns].
WAVELENGTHS = np.arange(0.3, 3, 0.001)

PC = 3.0857e16  # parsec in meters

# Axes of the precision grids. The atmosphere ingredients are tabulated at
# these values, so a grid can only be built on them.
# fmt: off
PWV_VALUES = np.array([
    0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0,
])  # precipitable water vapour at zenith [mm]

AIRMASS_VALUES = np.array([
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3,
    2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
])  # airmass at Paranal

TEFF_VALUES = np.array([
    450, 500, 550, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
    1600, 1700, 1800, 2000, 2100, 2250, 2320, 2400, 2440, 2500, 2600, 2650,
    2710, 2850, 3000, 3030, 3100, 3200, 3250, 3410, 3500, 3550, 3650, 3700,
    3800, 3870, 3940, 4000, 4070, 4190, 4230, 4330, 4410, 4540, 4600, 4700,
    4830, 4990, 5040, 5140, 5170, 5240, 5280, 5340, 5490, 5530, 5590, 5660,
    5680, 5720, 5770, 5880, 5920, 6000, 6060, 6170, 6240, 6340, 6510, 6640,
    6720, 6810, 7030, 7220, 7440, 7500, 7800, 8000, 8080, 8270, 8550, 8840,
    9200, 9700, 10400, 10700, 12500, 14000, 14500, 15700, 16700, 17000,
    18500, 20600, 24500, 26000, 29000, 31500, 32000, 32500, 33000, 34500,
    35000, 36500,
])  # stellar effective temperature [K]
# fmt: on
