from pathlib import Path

import numpy as np
import pandas as pd

import mphot


def test_interpolate_dfs():
    # Create sample DataFrames
    df1 = pd.DataFrame({"A": [1, 3, 5]}, index=[0, 2, 4])
    df2 = pd.DataFrame({"B": [1, 4, 6]}, index=[0, 3, 5])
    index = [0, 1, 2, 3, 4, 5]

    result = mphot.interpolate_dfs(index, df1, df2)

    assert list(result.index) == index
    assert list(result["A"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]
    assert list(result["B"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_integration_time():
    t = mphot.integration_time(1.0, 1000, 100, 10, 0.5, 100000, 0.8)
    assert t > 0  # Integration time should be positive


def test_get_precision():
    props = {
        "name": "speculoos_Andor_iKon-L-936_-60_I+z",
        "plate_scale": 0.2,
        "N_dc": 0.1,
        "N_rn": 5.0,
        "well_depth": 100000,
        "well_fill": 0.8,
        "read_time": 1.0,
        "r0": 1.0,
        "r1": 0.5,
    }

    props_sky = {
        "pwv": 1.0,
        "airmass": 1.2,
        "seeing": 1.0,
    }

    Teff = 5800
    distance = 10.0

    result = mphot.get_precision(props, props_sky, Teff, distance)

    assert isinstance(result, tuple)
    assert len(result) == 3

    image_precision, binned_precision, components = result

    assert isinstance(image_precision, dict)
    assert isinstance(binned_precision, dict)
    assert isinstance(components, dict)

    assert "All" in image_precision
    assert "Star" in image_precision
    assert "Scintillation" in image_precision
    assert "Sky" in image_precision
    assert "Dark current" in image_precision
    assert "Read noise" in image_precision

    assert "All" in binned_precision
    assert "Star" in binned_precision
    assert "Scintillation" in binned_precision
    assert "Sky" in binned_precision
    assert "Dark current" in binned_precision
    assert "Read noise" in binned_precision

    assert "name" in components
    assert "Teff [K]" in components
    assert "distance [pc]" in components
    assert "N_star [e/s]" in components
    assert "star_flux [e/m2/s]" in components
    assert "scn [e_rms]" in components
    assert "pixels in aperture [pix]" in components
    assert "ap_radius [pix]" in components
    assert "N_sky [e/pix/s]" in components
    assert "sky_radiance [e/m2/arcsec2/s]" in components
    assert "seeing [arcsec]" in components
    assert "pwv [mm]" in components
    assert "airmass" in components
    assert 'plate_scale ["/pix]' in components
    assert "N_dc [e/pix/s]" in components
    assert "N_rn [e_rms/pix]" in components
    assert "A [m2]" in components
    assert "r0 [m]" in components
    assert "r1 [m]" in components
    assert "t [s]" in components
    assert "well_depth [e/pix]" in components
    assert "peak well_fill" in components
    assert "binning [mins]" in components
    assert "read_time [s]" in components
    assert "binned images" in components


def test_generate_system_response():
    instrument_efficiency_path = "resources/systems/speculoos_Andor_iKon-L-936_-60.csv"  # index in microns, efficiency of telescope+instrument as fraction
    filter_path = (
        "resources/filters/I+z.csv"  # index in microns, filter efficiency as fraction
    )

    # does instrument efficiency file exist?
    assert Path(
        instrument_efficiency_path
    ).exists(), f"Expected file {instrument_efficiency_path} to exist, but it does not"

    # does filter file exist?
    assert Path(
        filter_path
    ).exists(), f"Expected file {filter_path} to exist, but it does not"

    name, system_response = mphot.generate_system_response(
        instrument_efficiency_path, filter_path
    )

    expected_name = "speculoos_Andor_iKon-L-936_-60_I+z"
    assert name == expected_name, f"Expected name to be {expected_name}, but got {name}"

    SRFile = (
        Path(mphot.__path__[0])
        / "datafiles"
        / "system_responses"
        / f"{expected_name}_instrument_system_response.csv"
    )

    assert SRFile.exists(), f"Expected file {SRFile} to exist, but it does not"

    SR = pd.read_csv(SRFile, index_col=0, header=None)[1]

    assert np.allclose(SR.index.values, system_response.index.values), "Index mismatch"
    assert np.allclose(SR.values, system_response.values), "Values mismatch"
