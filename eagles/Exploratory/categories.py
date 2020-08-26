from eagles.Exploratory.utils import plot_utils as pu
import pandas as pd
from IPython.display import display
import logging

logger = logging.getLogger(__name__)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def get_sample_stats(
    data: pd.DataFrame = None, cols: list = [], disp=True, plot=False
) -> pd.DataFrame:
    """
    Function to get count and proportions of samples
    :param data: default None, expects pandas dataframe
    :param cols: default empty list, if empty defaults to all object columns
    :param disp: default True, boolean indicator to display dataframe of results
    :param plot: default False, boolean indicator to plot histograms
    :return:
    """

    if len(cols) == 0:
        cols = data.columns

    # filter out object cols
    cols = [col for col in cols if data[col].dtype == "O"]

    if len(cols) == 0:
        print("No cols detected, expects cols to be type Object")
        return None

    tmp = data.copy(deep=True)
    tmp["count"] = [i for i in range(len(data))]
    cat_df = pd.DataFrame()
    for col in cols:
        grp = tmp.groupby(col, as_index=False)["count"].agg("count")
        grp["proportion_samples"] = round((grp["count"] / len(tmp)) * 100, 2)
        grp[col] = list(map(lambda x: col + "_" + x, grp[col]))

        grp.columns = ["feature_by_category", "count", "proportion_samples"]

        cat_df = pd.concat([cat_df, grp])

    if disp:
        display(cat_df)

    if plot:
        pu.plot_category_histograms(data=data, cols=cols)

    return cat_df


def get_multi_group_stats(
    data: pd.DataFrame = None, group_cols: list = None, disp=True
) -> pd.DataFrame:
    """
    Function to get proportion samples for multiple category cols
    :param data: default None, expects pandas dataframe
    :param group_cols: default None, Expects list of cols to group by
    :param disp:
    :return:
    """

    if group_cols is None:
        logger.warning("No grouping cols passed in")
        return None

    tmp = data[group_cols].copy(deep=True)
    tmp["count"] = [i for i in range(len(data))]

    grp = tmp.groupby(group_cols, as_index=False)["count"].agg("count")
    grp["proportion_samples"] = round((grp["count"] / len(tmp)) * 100, 2)

    if disp:
        display(grp)

    return grp
