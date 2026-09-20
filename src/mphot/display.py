"""Notebook output of precision results."""

import math

import numpy as np
import pandas as pd
from IPython.display import display

from mphot.paths import system_response_path
from mphot.precision import convert_airmass, vega_mag


def display_number(x: float, p: int = 3) -> str:
    """
    Convert a number to a string with the given precision.

    Args:
        x (float): The number to be converted.
        p (int, optional): The precision (number of significant digits). Default is 3.

    Returns:
        str: The number represented as a string with the specified precision.

    Examples:
    >>> display_number(123.456, 4)
    '123.5'
    >>> display_number(0.00123456, 2)
    '0.0012'
    >>> display_number(123456, 2)
    '1.2e+05'
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

    m = f"{n:.{p}g}"

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


def display_results(r1: tuple, r2: tuple | None = None) -> None:
    """
    Display the results of the photometric analysis.

    Args:
        props_sky (dict):
            Dictionary containing properties of the sky.
        r1 (tuple):
            A tuple containing image precision, binned precision, and components for the first set of results.
                - image_precision1 (dict):
                    Dictionary containing image precision metrics for the first set.
                - binned_precision1 (dict):
                    Dictionary containing binned precision metrics for the first set.
                - components1 (dict):
                    Dictionary containing components for the first set.
        r2 (tuple, optional):
            A tuple containing image precision, binned precision, and components for the second set of results.
                - image_precision2 (dict):
                    Dictionary containing image precision metrics for the second set.
                - binned_precision2 (dict):
                    Dictionary containing binned precision metrics for the second set.
                - components2 (dict):
                    Dictionary containing components for the second set.

    Returns:
        None
            This function displays the results using pandas DataFrames and does not return any value.
    """

    pd.set_option("display.float_format", display_number)

    image_precision1, binned_precision1, components1 = r1

    # Copy the values to avoid directly editing the original dictionaries
    image_precision1 = image_precision1.copy()
    binned_precision1 = binned_precision1.copy()
    components1 = components1.copy()
    name1 = components1["name"]
    components1.pop("name")

    props_sky1 = {
        "pwv": components1["pwv [mm]"],
        "airmass": convert_airmass(components1["airmass"], components1["altitude [m]"]),
        "seeing": components1["seeing [arcsec]"],
    }

    SRFile1 = system_response_path(name1)

    vega1 = vega_mag(
        SRFile1,
        props_sky1,
        components1["N_star [e/s]"],
        components1["sky_radiance [e/m2/arcsec2/s]"],
        components1["A [m2]"],
    )

    if r2 is not None:
        image_precision2, binned_precision2, components2 = r2

        # Copy the values to avoid directly editing the original dictionaries
        image_precision2 = image_precision2.copy()
        binned_precision2 = binned_precision2.copy()
        components2 = components2.copy()
        name2 = components2["name"]
        components2.pop("name")

        props_sky2 = {
            "pwv": components2["pwv [mm]"],
            "airmass": convert_airmass(
                components2["airmass"], components2["altitude [m]"]
            ),
            "seeing": components2["seeing [arcsec]"],
        }

        SRFile2 = system_response_path(name2)

        vega2 = vega_mag(
            SRFile2,
            props_sky2,
            components2["N_star [e/s]"],
            components2["sky_radiance [e/m2/arcsec2/s]"],
            components2["A [m2]"],
        )

        columns = [
            [
                "single frame [ppt]",
                "single frame [ppt]",
                f"{components1['binning [mins]']} minute binned [ppt]",
                f"{components2['binning [mins]']} minute binned [ppt]",
            ],
            [name1, name2, name1, name2],
        ]
        values = (
            np.c_[
                list(image_precision1.values()),
                list(image_precision2.values()),
                list(binned_precision1.values()),
                list(binned_precision2.values()),
            ]
            * 1000  # convert to ppt
        )
        display(pd.DataFrame(values, index=image_precision1.keys(), columns=columns))

        columns = [[name1, name2]]

        for k, v in components1.items():
            if not isinstance(v, (str, bool)):
                components1[k] = display_number(v)

        for k, v in components2.items():
            if not isinstance(v, (str, bool)):
                components2[k] = display_number(v)

        values = np.c_[list(components1.values()), list(components2.values())]
        display(pd.DataFrame(values, index=components1.keys(), columns=columns))

        columns = [[name1, name2]]
        values = np.c_[list(vega1.values()), list(vega2.values())]
        display(pd.DataFrame(values, index=vega1.keys(), columns=columns))

    else:
        columns = [
            [
                "single frame [ppt]",
                f"{components1['binning [mins]']} minute binned [ppt]",
            ],
            [name1, name1],
        ]
        values = (
            np.c_[
                list(image_precision1.values()),
                list(binned_precision1.values()),
            ]
            * 1000  # convert to ppt
        )
        display(pd.DataFrame(values, index=image_precision1.keys(), columns=columns))

        columns = [[name1]]

        for k, v in components1.items():
            if not isinstance(v, (str, bool)):
                components1[k] = display_number(v)

        values = np.c_[list(components1.values())]
        display(pd.DataFrame(values, index=components1.keys(), columns=columns))

        columns = [[name1]]
        values = np.c_[list(vega1.values())]
        display(pd.DataFrame(values, index=vega1.keys(), columns=columns))
