import contextlib
import io
import json
import os
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mphot
import mphot.core
import mphot.gaia


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
    assert Path(instrument_efficiency_path).exists(), (
        f"Expected file {instrument_efficiency_path} to exist, but it does not"
    )

    # does filter file exist?
    assert Path(filter_path).exists(), (
        f"Expected file {filter_path} to exist, but it does not"
    )

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


def test_to_float():
    assert mphot.gaia._to_float("1") == 1.0
    assert mphot.gaia._to_float(1) == 1.0
    assert np.isnan(mphot.gaia._to_float(None))
    assert np.isnan(mphot.gaia._to_float(""))
    assert np.isnan(mphot.gaia._to_float("not a number"))


def test_to_float_float32_recovers_stored_value():
    # teff_gspphot is float32 upstream, so CSV carries its shortest decimal.
    assert mphot.gaia._to_float("3099.6", float32=True) == 3099.60009765625
    assert mphot.gaia._to_float("3099.6") == 3099.6
    assert np.isnan(mphot.gaia._to_float("", float32=True))


def test_tap_sync_csv_builds_a_sync_url_and_parses_rows(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return contextlib.closing(io.BytesIO(b"parallax,teff_gspphot\n80.2,5000\n"))

    monkeypatch.setattr(mphot.gaia.urllib.request, "urlopen", fake_urlopen)

    rows = mphot.gaia._tap_sync_csv("https://example.org/tap/", "SELECT 1", 30)

    assert rows == [{"parallax": "80.2", "teff_gspphot": "5000"}]
    assert seen["timeout"] == 30
    assert seen["url"].startswith("https://example.org/tap/sync?")
    assert "FORMAT=csv" in seen["url"]
    assert "SELECT+1" in seen["url"]


def test_tap_sync_csv_turns_a_timeout_into_timeout_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(mphot.gaia.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(TimeoutError):
        mphot.gaia._tap_sync_csv("https://example.org/tap", "SELECT 1", 1)


def test_tap_sync_csv_reports_an_http_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "https://example.org/tap/sync",
            400,
            "Bad Request",
            {},
            io.BytesIO(b"malformed ADQL"),
        )

    monkeypatch.setattr(mphot.gaia.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        mphot.gaia._tap_sync_csv("https://example.org/tap", "SELECT 1", 1)


def test_query_gaia_tap_maps_columns_and_reports_a_missing_source(monkeypatch):
    monkeypatch.setattr(
        mphot.gaia,
        "_tap_sync_csv",
        lambda url, adql, timeout: [
            {
                "PARALLAX": "80.2",
                "TEFF_GSPPHOT": "3099.6",
                "PHOT_BP_MEAN_FLUX": "341.2",
                "PHOT_G_MEAN_FLUX": "10615.1",
                "PHOT_RP_MEAN_FLUX": "18095.7",
            }
        ],
    )
    # Column names come back in whatever case the service uses.
    result = mphot.gaia._query_gaia_tap(1, "vizier", 30)
    assert result["parallax"] == 80.2
    assert result["teff_gspphot"] == 3099.60009765625

    monkeypatch.setattr(mphot.gaia, "_tap_sync_csv", lambda url, adql, timeout: [])
    with pytest.raises(mphot.GaiaSourceNotFound):
        mphot.gaia._query_gaia_tap(1, "vizier", 30)


def test_package_imports_without_ipython(monkeypatch):
    # IPython is optional; the package must import in a plain interpreter.
    import subprocess

    code = (
        "import sys;"
        "sys.modules['IPython'] = None;"
        "sys.modules['IPython.display'] = None;"
        "import mphot, mphot.display, mphot.utils;"
        "assert mphot.utils.clear_output is None, 'clear_output fallback not used';"
        "assert mphot.display.display is print, 'display fallback not used';"
        "mphot.update_progress(0.5);"
        "print('OK', mphot.get_precision.__name__)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(mphot.__path__[0]).parent)},
    )
    assert out.returncode == 0, out.stderr
    assert "OK get_precision" in out.stdout
    assert "Progress: [##########----------] 50.0%" in out.stdout


def test_query_gaia_source_rejects_unknown_service():
    with pytest.raises(ValueError, match="Unknown Gaia TAP source"):
        mphot.query_gaia_source(1, tap_sources="not_a_service")


def test_query_gaia_source_falls_back(monkeypatch):
    tried = []
    row = dict.fromkeys(mphot.gaia._GAIA_COLUMNS, 1.0)

    def fake_query(source_id, tap_source, timeout):
        tried.append(tap_source)
        if tap_source == "esa":
            raise TimeoutError("timed out")
        return row

    monkeypatch.setattr(mphot.gaia, "_query_gaia_tap", fake_query)

    result = mphot.query_gaia_source(1, timeout=1, tap_sources=("esa", "vizier"))

    assert tried == ["esa", "vizier"]
    assert result == row


def test_query_gaia_source_raises_when_all_services_fail(monkeypatch):
    def fake_query(source_id, tap_source, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(mphot.gaia, "_query_gaia_tap", fake_query)

    with pytest.raises(RuntimeError, match="any TAP service"):
        mphot.query_gaia_source(1, timeout=1, tap_sources=("esa", "vizier"))


def test_query_gaia_source_does_not_retry_missing_source(monkeypatch):
    tried = []

    def fake_query(source_id, tap_source, timeout):
        tried.append(tap_source)
        raise mphot.GaiaSourceNotFound("missing")

    monkeypatch.setattr(mphot.gaia, "_query_gaia_tap", fake_query)

    with pytest.raises(mphot.GaiaSourceNotFound):
        mphot.query_gaia_source(1, timeout=1, tap_sources=("vizier", "esa"))

    assert tried == ["vizier"]


def test_interpolate_grid_returns_a_float_on_and_off_a_grid_temperature():
    # griddata returns a 0-d array. The branch for a Teff that sits exactly on
    # a grid temperature used to pass it straight back, so the return type
    # depended on Teff and the value would not serialise to JSON.
    coords, data_flux, _ = mphot.core.load_grids("gaia_g_inverse_atmosphere_paranal")
    on_grid = float(mphot.core.TEFF_VALUES[20])

    for Teff in (on_grid, on_grid + 5):
        value = mphot.interpolate_grid(coords, data_flux, 2.5, 1.1, Teff)
        assert type(value) is float, f"Teff={Teff} gave {type(value).__name__}"


def test_get_precision_components_are_json_serialisable():
    name = "speculoos_Andor_iKon-L-936_-60_I+z"
    mphot.generate_system_response(
        "resources/systems/speculoos_Andor_iKon-L-936_-60.csv",
        "resources/filters/I+z.csv",
    )
    props = {
        "name": name,
        "plate_scale": 0.35,
        "N_dc": 0.2,
        "N_rn": 6.328,
        "well_depth": 64000,
        "well_fill": 0.7,
        "read_time": 10.5,
        "r0": 0.5,
        "r1": 0.14,
    }
    props_sky = {"pwv": 2.5, "airmass": 1.1, "seeing": 1.35}
    on_grid = float(mphot.core.TEFF_VALUES[20])

    for Teff in (on_grid, on_grid + 5):
        _, _, components = mphot.get_precision(props, props_sky, Teff, 12.5)
        json.dumps({k: v for k, v in components.items() if k != "name"})


def test_gaia_calibration_does_not_depend_on_site_altitude(monkeypatch):
    # The inverse atmosphere cancels the Paranal sky at airmass 1. The
    # calibration run used to take the site altitude too, which converted its
    # airmass to 1.26 at 569 m and made the calibrated star 1-3% too bright.
    star = {
        "teff_gspphot": 3000.0,
        "parallax": 80.0,
        "phot_bp_mean_flux": 2e4,
        "phot_g_mean_flux": 1e5,
        "phot_rp_mean_flux": 1.5e5,
    }
    monkeypatch.setattr(mphot.gaia, "query_gaia_source", lambda *a, **k: star)

    name, _ = mphot.generate_system_response(
        "resources/systems/speculoos_Andor_iKon-L-936_-60.csv",
        "resources/filters/I+z.csv",
    )
    props = {
        "name": name,
        "plate_scale": 0.35,
        "N_dc": 0.2,
        "N_rn": 6.328,
        "well_depth": 64000,
        "well_fill": 0.7,
        "read_time": 10.5,
        "r0": 0.5,
        "r1": 0.14,
    }
    props_sky = {"pwv": 2.5, "airmass": 1.1, "seeing": 1.35}

    factors = []
    for h in (2440, 569):
        _, _, raw = mphot.get_precision(props, props_sky, 3000.0, 12.5, h=h)
        _, _, cal = mphot.get_precision_gaia(
            props, props_sky, source_id=1, gaia_filter="g", h=h
        )
        factors.append(cal["N_star [e/s]"] / raw["N_star [e/s]"])

    assert factors[1] == pytest.approx(factors[0], rel=1e-9)
