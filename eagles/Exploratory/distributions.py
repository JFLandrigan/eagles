from eagles.Exploratory.utils import plot_utils as pu
import pandas as pd
import numpy as np
from IPython.display import display

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

import logging

logger = logging.getLogger(__name__)


def find_caps(
    df: pd.DataFrame = None, cols: list = [], stats: list = [], plot=False
) -> pd.DataFrame:
    """
    This function finds potential cap points for distributions and returns them by feature in a pandas dataframe
    :param df: pandas dataframe containing the data to be analyzed
    :param cols: list of column names to analyze
    :param stats: list of stats to find. Keys include 'sd' and/or 'percentile'
    :param plot: boolean indicator whether to plot distribs or not
    :return: pandas dataframe with feature by potential cap points
    """
    if not isinstance(stats, list):
        logger.warning("stats arg was not list")
        return None

    if len(cols) == 0:
        cols = df.columns

    if len(stats) == 0:
        stats = ["percentile", "sd"]

    cap_dict = {"Feature": list()}
    if "percentile" in stats:
        cap_dict["75th_Percentile"] = list()
        cap_dict["90th_Percentile"] = list()
    if "sd" in stats:
        cap_dict["plus_2_SD"] = list()
        cap_dict["plus_3_SD"] = list()
        cap_dict["minus_2_SD"] = list()
        cap_dict["minus_3_SD"] = list()
        cap_dict["skew"] = list()

    for col in cols:
        cap_dict["Feature"].append(col)
        if "percentile" in stats:
            cap_dict["75th_Percentile"].append(df[col].quantile(0.75))
            cap_dict["90th_Percentile"].append(df[col].quantile(0.90))
        if "sd" in stats:
            skew = df[col].skew()
            cap_dict["skew"].append(skew)
            if skew > 0.1:
                cap_dict["plus_2_SD"].append(df[col].mean() + (df[col].std() * 2))
                cap_dict["plus_3_SD"].append(df[col].mean() + (df[col].std() * 3))
                cap_dict["minus_2_SD"].append(np.nan)
                cap_dict["minus_3_SD"].append(np.nan)
            elif skew < -0.1:
                cap_dict["plus_2_SD"].append(np.nan)
                cap_dict["plus_3_SD"].append(np.nan)
                cap_dict["minus_2_SD"].append(df[col].mean() - (df[col].std() * 2))
                cap_dict["minus_3_SD"].append(df[col].mean() - (df[col].std() * 3))
            else:
                cap_dict["plus_2_SD"].append(df[col].mean() + (df[col].std() * 2))
                cap_dict["plus_3_SD"].append(df[col].mean() + (df[col].std() * 3))
                cap_dict["minus_2_SD"].append(df[col].mean() - (df[col].std() * 2))
                cap_dict["minus_3_SD"].append(df[col].mean() - (df[col].std() * 3))

    cap_df = pd.DataFrame(cap_dict)
    display(cap_df)

    if plot:
        pu.plot_distributions(data=df, cols=cols, caps=cap_dict)

    return cap_df
