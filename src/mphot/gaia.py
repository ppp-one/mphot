"""Gaia DR3 lookups and flux calibration against the Gaia bands."""

import csv
import io
import logging
import sys
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from scipy.optimize import minimize

from mphot.paths import FLUX_CALIBRATION_DIR, system_response_path
from mphot.precision import get_precision

logger = logging.getLogger(__name__)


def best_gaia_filters(
    system_name: str, min_weight_sum: float = 0, support_points: int = 8000
) -> np.ndarray:
    """
    Determine the weights of the linear combination of Gaia filters that best resembles the instrument system response.

    Args:
        system_name (str):
            Name of the instrument.

        min_weight_sum (float, optional):
            The minimum sum of weights in the linear combination of Gaia filters. Default is 0.

        support_points (int, optional):
            The number of support points in the wavelength spectrum between 0.3 and 3.0 microns used to interpolate the instrument system response and Gaia filter transmission curves. Default is 8000.

    Returns:
        tuple: A tuple containing:
            image_precision : dict
                Precision of the image
            binned_precision : dict
                Precision of the binned image
            components : dict
                Various components used in the calculation
    """

    # Load transmission curves
    path = system_response_path(system_name)
    trans = np.loadtxt(str(path), delimiter=",")

    gaia_filters = ["bp", "g", "rp"]
    N = len(gaia_filters)
    gaia_paths = [
        FLUX_CALIBRATION_DIR / f"gaia_{gaia_filter}.csv" for gaia_filter in gaia_filters
    ]
    gaia_trans = []

    for gaia_path in gaia_paths:
        g_trans = np.loadtxt(str(gaia_path), delimiter=",")
        gaia_trans.append(g_trans)

    # Get best linear combination of Gaia filters
    lam = np.linspace(0.3, 3, support_points)
    trans_interp = np.interp(lam, trans[:, 0], trans[:, 1], left=0, right=0)
    gaia_trans_interp = np.column_stack(
        [
            np.interp(lam, g_trans[:, 0], g_trans[:, 1], left=0, right=0)
            for g_trans in gaia_trans
        ]
    )

    def squared_err(weights):
        err = gaia_trans_interp @ weights - trans_interp

        return err @ err

    def jacobian(weights):
        err = gaia_trans_interp @ weights - trans_interp

        return 2 * gaia_trans_interp.T @ err

    bounds = [(0, None)] * N
    cons = ({"type": "ineq", "fun": lambda weights: np.sum(weights) - min_weight_sum},)
    x0 = np.full(N, max(1.0, min_weight_sum / N))

    res = minimize(
        squared_err,
        x0,
        jac=jacobian,
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000},
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    return res.x


GAIA_TAP_URLS = {
    "vizier": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
    "esa": "https://gea.esac.esa.int/tap-server/tap",
}
"""TAP services that serve the Gaia DR3 catalogue. VizieR hosts a copy of it."""

DEFAULT_GAIA_TAP_SOURCES = ("vizier", "esa")
"""Order in which the TAP services are tried. The first one that answers wins."""

DEFAULT_GAIA_TIMEOUT = 60
"""Seconds to wait for one TAP service before moving on to the next one."""

# Only these columns are read. Asking for all 152 columns of gaiadr3.gaia_source
# makes the query far slower for no gain.
_GAIA_COLUMNS = (
    "parallax",
    "teff_gspphot",
    "phot_bp_mean_flux",
    "phot_g_mean_flux",
    "phot_rp_mean_flux",
)

# Columns the archive stores as 32-bit floats. CSV carries the shortest decimal
# that round-trips the stored float32, so reading it straight into a float64
# lands a few ULP away from the catalogue value. Casting back through float32
# recovers the stored bits exactly.
_FLOAT32_COLUMNS = frozenset({"teff_gspphot"})

# Each service uses its own column and table names, so the results are mapped
# back to the Gaia DR3 names given in _GAIA_COLUMNS.
_GAIA_TAP_TABLES = {
    "esa": {
        "table": "gaiadr3.gaia_source",
        "id_column": "source_id",
        "columns": {column: column for column in _GAIA_COLUMNS},
    },
    "vizier": {
        "table": '"I/355/gaiadr3"',
        "id_column": '"Source"',
        "columns": {
            "parallax": "Plx",
            "teff_gspphot": "Teff",
            "phot_bp_mean_flux": "FBP",
            "phot_g_mean_flux": "FG",
            "phot_rp_mean_flux": "FRP",
        },
    },
}


class GaiaSourceNotFound(ValueError):
    """Raised when a TAP service holds no row for the given source_id."""


def _to_float(value, float32: bool = False) -> float:
    """
    Convert a CSV field to float, mapping an empty or missing one to NaN.

    Args:
        value: The raw CSV field.
        float32 (bool, optional): If True, round the value to the nearest
            float32 first, recovering the value the archive stores.

    Returns:
        float: The parsed value, or NaN if it is empty or not a number.
    """

    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return float(np.float32(number)) if float32 else number


def _read_url_emscripten(url: str) -> str:
    """
    Read a URL from inside a browser, where sockets do not exist.

    WebAssembly has no sockets, so `urllib` cannot reach the network. Pyodide
    offers a synchronous reader built on the browser's own HTTP stack, which
    keeps the calling code unchanged.

    Note that the browser applies its same-origin policy, so the service must
    send an `Access-Control-Allow-Origin` header. VizieR does; the ESA archive
    does not, and needs a proxy.

    Args:
        url (str): The URL to read.

    Returns:
        str: The response body.

    Raises:
        RuntimeError: If the request fails, usually because the service sends
            no CORS header.
    """

    from pyodide.http import open_url

    try:
        return open_url(url).getvalue()
    except Exception as e:
        raise RuntimeError(
            f"the browser could not fetch the TAP service ({type(e).__name__}). "
            "The service must allow cross-origin requests."
        ) from e


def _tap_sync_csv(base_url: str, adql: str, timeout: float | None) -> list[dict]:
    """
    Run a synchronous TAP query and return its rows.

    The TAP standard serves results over plain HTTP GET, so this needs no
    client library. Asking for CSV also avoids a VOTable parser.

    Args:
        base_url (str): Base TAP endpoint, without the ``/sync`` suffix.
        adql (str): The ADQL query.
        timeout (float, optional): Seconds to wait for each network operation.
            If None, there is no timeout. Ignored in a browser, where the
            browser controls the timeout.

    Returns:
        list[dict]: One dict per result row, keyed by column name, values as
            strings.

    Raises:
        TimeoutError: If the service does not respond within `timeout` seconds.
        RuntimeError: If the service returns an HTTP error.
    """

    query = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}
    )
    url = base_url.rstrip("/") + "/sync?" + query

    if sys.platform == "emscripten":
        return list(csv.DictReader(io.StringIO(_read_url_emscripten(url))))

    request = urllib.request.Request(url, headers={"User-Agent": "mphot"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace").strip().replace("\n", " ")
        raise RuntimeError(f"HTTP {e.code} from the TAP service: {detail[:200]}") from e
    except urllib.error.URLError as e:
        # A connect timeout arrives wrapped, a read timeout arrives bare.
        if isinstance(e.reason, TimeoutError):
            raise TimeoutError(f"no response within {timeout} s") from e
        raise RuntimeError(f"cannot reach the TAP service: {e.reason}") from e
    except TimeoutError as e:
        raise TimeoutError(f"no response within {timeout} s") from e

    return list(csv.DictReader(io.StringIO(body)))


def _query_gaia_tap(source_id, tap_source: str, timeout: float | None) -> dict:
    """Read one Gaia DR3 source from a single TAP service."""
    config = _GAIA_TAP_TABLES[tap_source]
    columns = config["columns"]
    select = ", ".join(
        f"{remote_name} AS {name}" for name, remote_name in columns.items()
    )
    adql = (
        f"SELECT TOP 1 {select} "
        f"FROM {config['table']} "
        f"WHERE {config['id_column']} = {int(source_id)}"
    )

    logger.debug(f"Gaia query to '{tap_source}': {adql}")
    rows = _tap_sync_csv(GAIA_TAP_URLS[tap_source], adql, timeout)

    if not rows:
        raise GaiaSourceNotFound(
            f"No Gaia DR3 source with source_id {source_id} in '{tap_source}'."
        )

    row = rows[0]
    # Column names come back in the case used by the service, so match them
    # without case.
    lookup = {name.lower(): name for name in row}
    return {
        name: _to_float(row[lookup[name.lower()]], name in _FLOAT32_COLUMNS)
        for name in columns
    }


def query_gaia_source(
    source_id: np.uint64,
    timeout: float | None = DEFAULT_GAIA_TIMEOUT,
    tap_sources: str | tuple = DEFAULT_GAIA_TAP_SOURCES,
) -> dict:
    """
    Get the parameters of one Gaia DR3 source used for flux calibration.

    The query is a plain HTTP GET against the service's TAP ``/sync`` endpoint
    asking for CSV, so it needs no TAP client library. It reads only the five
    columns that mphot needs. If a service fails or times out, the next service
    in `tap_sources` is tried.

    Args:
        source_id (np.uint64):
            The source_id property of the target from the Gaia DR3 catalog.

        timeout (float, optional):
            Seconds to wait for each network operation, per service. If None,
            there is no timeout. Default is 60.

        tap_sources (str or tuple, optional):
            Name or names of the TAP services to try, in order. Must be keys of
            GAIA_TAP_URLS ("vizier" or "esa"). Default is ("vizier", "esa").

    Returns:
        dict:
            The columns "parallax", "teff_gspphot", "phot_bp_mean_flux",
            "phot_g_mean_flux" and "phot_rp_mean_flux" as floats. Missing
            values are NaN.

    Raises:
        GaiaSourceNotFound: If the first service that answers holds no such source.
        RuntimeError: If every service fails or times out.
    """
    if isinstance(tap_sources, str):
        tap_sources = (tap_sources,)

    unknown = [name for name in tap_sources if name.lower() not in GAIA_TAP_URLS]
    if unknown:
        raise ValueError(
            f"Unknown Gaia TAP source(s): {unknown}. "
            f"Valid options are: {sorted(GAIA_TAP_URLS)}."
        )
    if not tap_sources:
        raise ValueError("At least one TAP source must be given.")

    failures = []
    for tap_source in tap_sources:
        tap_source = tap_source.lower()
        try:
            return _query_gaia_tap(source_id, tap_source, timeout)
        except GaiaSourceNotFound:
            # The source is missing, not the service. Trying another copy of
            # the same catalogue will not help.
            raise
        except Exception as e:
            # Keep the reason and try the next service; all reasons are
            # reported together if every service fails.
            failures.append(f"{tap_source} ({type(e).__name__}: {e})")
            logger.warning(f"Gaia query to '{tap_source}' failed: {e}")

    raise RuntimeError(
        "Could not read Gaia DR3 source "
        f"{source_id} from any TAP service. Failures: " + "; ".join(failures)
    )


def get_precision_gaia(
    props: dict,
    props_sky: dict,
    source_id: np.uint64,
    gaia_filter: str | None = None,
    min_weight_sum: float = 0,
    support_points: int = 8000,
    binning: float = 10,
    override_grid: bool = False,
    N_sky: float | None = None,
    scn: float | None = None,
    h: float = 2440,
    C: float = 1.56,
    exp_time: float | None = None,
    Teff: float | None = None,
    distance: float | None = None,
    timeout: float | None = DEFAULT_GAIA_TIMEOUT,
    tap_sources: str | tuple = DEFAULT_GAIA_TAP_SOURCES,
) -> dict:
    """
    Calculate the precision of astronomical observations based on various parameters and perform calibration of fluxes to Gaia fluxes.

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

        source_id (np.int64):
            The source_id property of the target from the Gaia DR3 catalog.

        gaia_filter (str, optional):
            The Gaia filter used for calibration. Must be one of the following:
            - "bp"
            - "g"
            - "rp"
            - If None, filters are selected automatically.
            See https://www.cosmos.esa.int/web/gaia/edr3-passbands for further information. Default is None.

        min_weight_sum (float, optional):
            The minimum sum of Gaia filter weights for determining the best linear combination of Gaia filters. Only used if gaia_filter is not specified. Default is 0.

        support_points (int, optional):
            The number of support points in the wavelength spectrum between 0.3 and 3.0 microns for determining the best linear combination of Gaia filters. Only used if gaia_filter is not specified. Default is 8000.

        binning (float, optional):
            Binning time in minutes. Default is 10.

        override_grid (bool, optional):
            If True, override existing grid files. Default is False.

        N_sky (float, optional):
            Number of sky counts, calculated if None. Default is None.

        scn (float, optional):
            Scintillation noise, calculated if None. Default is None.

        h (float, optional):
            Altitude of the observatory in meters. Default is 2440 for Paranal Observatory.

        C (float, optional):
            Empirical coefficient used in the calculation of scn. Default is 1.56, optimized for the 20-cm NGTS telescopes at Paranal Observatory.

        exp_time (float, optional):
            Exposure time in seconds, calculated if None. Default is None.

        Teff (float, optional):
            Effective temperature of the star in Kelvin. If None, it will be fetched from the Gaia catalog. Default is None.

        distance (float, optional):
            Distance to the star in parsecs. If None, it will be calculated from the parallax fetched from the Gaia catalog. Default is None.

        timeout (float, optional):
            Seconds to wait for each Gaia TAP service. If None, there is no timeout. Default is 60.

        tap_sources (str or tuple, optional):
            Name or names of the TAP services to query, in order. Must be keys of GAIA_TAP_URLS ("vizier" or "esa"). The next service is tried if one fails or times out. Default is ("vizier", "esa").

    Returns:
        tuple: A tuple containing:
            image_precision : dict
                Precision of the image
            binned_precision : dict
                Precision of the binned image
            components : dict
                Various components used in the calculation
    """
    name = props["name"]
    r0 = props["r0"]
    r1 = props["r1"]

    params = query_gaia_source(source_id, timeout=timeout, tap_sources=tap_sources)

    if Teff is None:
        Teff = params["teff_gspphot"]
    if distance is None:
        parallax = params["parallax"]
        # A zero or negative parallax gives no usable distance, so fall back on
        # the NaN branch below.
        distance = 1 / (parallax * 1e-3) if parallax > 0 else float("nan")  # pc

    if np.isnan(Teff):
        logger.warning(f"Teff value for source_id {source_id} is NaN.")
        Teff = 3000  # K
        logger.warning(f"Setting Teff to {Teff} K.")
    if np.isnan(distance):
        logger.warning(f"Distance value for source_id {source_id} is NaN.")
        distance = 10  # pc
        logger.warning(f"Setting distance to {distance} pc.")

    gaia_filters = np.array(["bp", "g", "rp"])
    gaia_fluxes = np.array([])
    mphot_fluxes = np.array([])

    # Determine used Gaia filters and weights
    if gaia_filter:
        filter_index = np.nonzero(gaia_filters == gaia_filter)[0][0]
        weights_vec = np.zeros(3)
        weights_vec[filter_index] = 1

        weights = 1
        gaia_filters = np.array([gaia_filter])
    else:
        weights = best_gaia_filters(
            name, min_weight_sum=min_weight_sum, support_points=support_points
        )
        weights_vec = weights

    for g_filter in gaia_filters:
        gaia_str = f"phot_{g_filter}_mean_flux"
        if np.isnan(params[gaia_str]):
            raise ValueError(
                f"Gaia DR3 source {source_id} has no {gaia_str} value, "
                "so the flux cannot be calibrated. Select a different "
                "gaia_filter."
            )
        gaia_flux = params[gaia_str] / 0.7278
        gaia_fluxes = np.append(gaia_fluxes, gaia_flux)

        # Get simulated Gaia flux
        props_instrument_gaia = props.copy()
        props_instrument_gaia["name"] = f"gaia_{g_filter}_inverse_atmosphere_paranal"

        ## Ideal conditions
        props_sky_gaia = {
            "pwv": 0.05,
            "airmass": 1,
            "seeing": props["plate_scale"],
        }

        # The inverse atmosphere cancels the Paranal sky at airmass 1, so this
        # run stays at the default Paranal altitude. At the site altitude, the
        # airmass would be converted, and the extra extinction would make the
        # calibrated star too bright.
        _, _, components_gaia = get_precision(
            props_instrument_gaia,
            props_sky_gaia,
            Teff,
            distance,
            binning=binning,
            override_grid=override_grid,
            N_sky=N_sky,
            scn=scn,
            C=C,
            exp_time=exp_time,
        )

        mphot_flux = components_gaia["N_star [e/s]"] / (np.pi * (r0**2 - r1**2))
        mphot_fluxes = np.append(mphot_fluxes, mphot_flux)

    factor = np.sum(weights * gaia_fluxes) / np.sum(weights * mphot_fluxes)

    # Calibrate simulated flux
    _, _, components = get_precision(
        props,
        props_sky,
        Teff,
        distance,
        binning=binning,
        override_grid=override_grid,
        N_sky=N_sky,
        scn=scn,
        h=h,
        C=C,
        exp_time=exp_time,
    )

    N_star_cal = components["N_star [e/s]"] * factor

    image_precision, binned_precision, components_final = get_precision(
        props,
        props_sky,
        Teff,
        distance,
        binning=binning,
        override_grid=False,
        N_star=N_star_cal,
        N_sky=N_sky,
        scn=scn,
        h=h,
        C=C,
        exp_time=exp_time,
    )

    components_final["Gaia-BP weight"] = weights_vec[0]
    components_final["Gaia-G weight"] = weights_vec[1]
    components_final["Gaia-RP weight"] = weights_vec[2]

    return image_precision, binned_precision, components_final
