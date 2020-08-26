from eagles.Exploratory.utils import plot_utils as pu
import pandas as pd
from IPython.display import display

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def get_proportion_missing(
    data: pd.DataFrame = None, cols: list = [], plot=False
) -> pd.DataFrame:
    """
    This function finds the percent missing per column in a pandas dataframe
    :param df: pandas dataframe
    :param cols: list of column names
    :return: Dataframe with col of feature names and col for percent missing
    """

    if len(cols) == 0:
        cols = data.columns

    percent_missing = data[cols].isnull().sum() * 100 / data.shape[0]
    missing_value_df = pd.DataFrame(percent_missing).reset_index()
    missing_value_df.columns = ["feature", "percent_missing"]

    missing_value_df.sort_values("percent_missing", ascending=False, inplace=True)

    if plot:
        pu.plot_missing_values(df=data, cols=cols)

    display(missing_value_df)

    return missing_value_df
