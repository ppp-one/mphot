"""Building the system response curve of an instrument."""

import logging

import pandas as pd

from mphot.constants import WAVELENGTHS
from mphot.paths import system_response_path
from mphot.utils import interpolate_dfs

logger = logging.getLogger(__name__)


def generate_system_response(
    efficiency_file: str, filter_file: str
) -> tuple[str, pd.Series]:
    """
    Generates a spectral response (SR) file by combining efficiency and filter data.

    Args:
        efficiency_file (str): Path to the CSV file containing efficiency data.
        filter_file (str): Path to the CSV file containing filter data.

    Returns:
        tuple: A tuple containing:
            - name (str): The name used to refer to the generated SR file.
            - dfSR (pd.Series): The spectral response data.
    """

    eff = pd.read_csv(efficiency_file, header=None)
    filt = pd.read_csv(filter_file, header=None)

    # name to refer to the generated file
    name = efficiency_file.split("/")[-1][:-4] + "_" + filter_file.split("/")[-1][:-4]

    # generates a SR, saved locally as 'name1_instrument_system_response.csv'
    SRFile = system_response_path(name)

    effDF = pd.DataFrame({"eff": eff[1].values}, index=eff[0])

    filtDF = pd.DataFrame({"filt": filt[1].values}, index=filt[0])

    df = interpolate_dfs(WAVELENGTHS, effDF, filtDF)

    dfSR = df["eff"] * df["filt"]

    dfSR = dfSR[dfSR >= 0]

    dfSR.to_csv(SRFile, header=False)

    logger.info(f"`{SRFile}` has been generated and saved!")

    return name, dfSR
