import pandas as pd

# TODO decide if want to add in missingno
def get_proportion_missing(df: pd.DataFrame = None, cols: list = []) -> None:
    """
    This function finds the percent missing per column in a pandas dataframe
    :param df: pandas dataframe
    :param cols: list of column names
    :return: None
    """

    if len(cols) == 0:
        cols = df.columns

    percent_missing = df[cols].isnull().sum() * 100 / df.shape[0]
    missing_value_df = pd.DataFrame(
        {"column_name": cols, "percent_missing": percent_missing}
    )

    missing_value_df.sort_values("percent_missing", inplace=True)

    return None
