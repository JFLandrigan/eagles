import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


def plot_missing_values(df: pd.DataFrame = None, cols: list = []) -> None:

    if len(cols) == 0:
        cols = df.columns

    _ = plt.figure(figsize=(8, 8))
    cmap = sns.cubehelix_palette(8, start=0, rot=0, dark=0, light=0.95, as_cmap=True)
    ax = sns.heatmap(data=pd.isnull(df[cols]), cmap=cmap, cbar=False)
    _ = ax.set_title("Missing Data")

    return None


def get_proportion_missing(
    df: pd.DataFrame = None, cols: list = [], plot=False
) -> None:
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

    missing_value_df.sort_values("percent_missing", ascending=False, inplace=True)
    print(missing_value_df)

    if plot:
        plot_missing_values(df=df, cols=cols)

    return None

