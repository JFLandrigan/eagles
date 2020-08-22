from eagles.Exploratory.utils import plot_utils as pu
import pandas as pd
from IPython.display import display

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def get_sample_stats(
    data: pd.DataFrame = None, cols: list = [], plot=False
) -> pd.DataFrame:
    """
    Function to get count and proportions of samples
    :param data:
    :param cols:
    :param plot:
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

    display(cat_df)

    if plot:
        pu.plot_category_histograms(data=data, cols=cols)

    return cat_df
