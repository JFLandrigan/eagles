import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def _plot_distribution_caps(
    df: pd.DataFrame = None, col: str = None, caps: dict = {}, stats: list = []
):
    """
    Function to plot the distributions and their potential caps
    :param df: pandas dataframe with col to plotted
    :param col: string name of the column
    :param caps: Dictionary containing the caps
    :param stats: list of stats to plot
    :return:
    """

    ind = caps["Feature"].index(col)

    _ = plt.figure(figsize=(5, 5))
    ax = sns.kdeplot(df[col], shade=True)
    if "sd" in stats:
        _ = plt.axvline(caps["2_SD"][ind])
        _ = plt.axvline(caps["3_SD"][ind])
    if "percentile" in stats:
        _ = plt.axvline(caps["75th_Percentile"][ind])
        _ = plt.axvline(caps["90th_Percentile"][ind])
    _ = plt.title(col + " Distribution Caps")

    return


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
        cap_dict["2_SD"] = list()
        cap_dict["3_SD"] = list()

    for col in cols:
        cap_dict["Feature"].append(col)
        if "percentile" in stats:
            cap_dict["75th_Percentile"].append(df[col].quantile(0.75))
            cap_dict["90th_Percentile"].append(df[col].quantile(0.90))
        if "sd" in stats:
            cap_dict["2_SD"].append(df[col].std() * 2)
            cap_dict["3_SD"].append(df[col].std() * 3)

        if plot:
            _plot_distribution_caps(df=df, col=col, caps=cap_dict, stats=stats)

    cap_df = pd.DataFrame(cap_dict)

    return cap_df
