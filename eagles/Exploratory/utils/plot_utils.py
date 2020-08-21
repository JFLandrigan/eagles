import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


def plot_distributions(df: pd.DataFrame = None, cols: list = []) -> None:

    if len(cols) == 0:
        cols = df.columns

    for col in cols:
        _ = plt.figure(figsize=(6, 6))
        ax = sns.kdeplot(df[col], shade=True, legend=False)
        _ = plt.title(col + " Distribution")

    return None


def plot_distribution_caps(
    df: pd.DataFrame = None, col: str = None, caps: dict = {}, stats: list = []
) -> None:
    """
    Function to plot the distributions and their potential caps
    :param df: pandas dataframe with col to plotted
    :param col: string name of the column
    :param caps: Dictionary containing the caps
    :param stats: list of stats to plot
    :return:
    """

    ind = caps["Feature"].index(col)

    _ = plt.figure(figsize=(6, 6))
    ax = sns.kdeplot(df[col], shade=True, legend=False)

    if "sd" in stats:
        if caps["skew"][ind] > 0.1:
            _ = plt.axvline(caps["plus_2_SD"][ind])
            _ = plt.axvline(caps["plus_3_SD"][ind])
        elif caps["skew"][ind] < -0.1:
            _ = plt.axvline(caps["minus_2_SD"][ind])
            _ = plt.axvline(caps["minus_3_SD"][ind])
        else:
            _ = plt.axvline(caps["plus_2_SD"][ind])
            _ = plt.axvline(caps["plus_3_SD"][ind])
            _ = plt.axvline(caps["minus_2_SD"][ind])
            _ = plt.axvline(caps["minus_3_SD"][ind])
    if "percentile" in stats:
        _ = plt.axvline(caps["75th_Percentile"][ind])
        _ = plt.axvline(caps["90th_Percentile"][ind])

    _ = plt.title(col + " Distribution Caps")
    plt.show()

    return None


def plot_missing_values(df: pd.DataFrame = None, cols: list = []) -> None:

    if len(cols) == 0:
        cols = df.columns

    _ = plt.figure(figsize=(12, 12))
    cmap = sns.cubehelix_palette(8, start=0, rot=0, dark=0, light=0.95, as_cmap=True)
    ax = sns.heatmap(data=pd.isnull(df[cols]), cmap=cmap, cbar=False)
    _ = ax.set_title("Missing Data")

    return None
