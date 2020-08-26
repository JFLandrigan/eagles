import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


def plot_distributions(
    data: pd.DataFrame = None, cols: list = [], caps: dict = None
) -> None:
    """
    Function to plot the distributions and their potential caps
    :param df: pandas dataframe with col to plotted
    :param cols: list of column names to plot
    :param caps: Dictionary containing the caps
    :return:
    """

    if len(cols) == 0:
        cols = data.columns

    # filter out object cols
    cols = [col for col in cols if data[col].dtype != "O"]

    for col in cols:
        _ = plt.figure(figsize=(6, 6))
        ax = sns.kdeplot(data[col], shade=True, legend=False)

        if caps:
            ind = caps["Feature"].index(col)

            if "plus_2_SD" in caps.keys():
                _ = plt.axvline(caps["plus_2_SD"][ind], label="plus 2 SD")
            if "plus_3_SD" in caps.keys():
                _ = plt.axvline(caps["plus_3_SD"][ind], label="plus 3 SD")
            if "minus_2_SD" in caps.keys():
                _ = plt.axvline(caps["minus_2_SD"][ind], label="minus 2 SD")
            if "minus_3_SD" in caps.keys():
                _ = plt.axvline(caps["minus_3_SD"][ind], label="minus 3 SD")
            if "75th_Percentile" in caps.keys():
                _ = plt.axvline(caps["75th_Percentile"][ind], label="75th percentile")
            if "90th_Percentile" in caps.keys():
                _ = plt.axvline(caps["90th_Percentile"][ind], label="90th percentile")

        _ = plt.title(col + " Distribution")

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


def plot_category_histograms(data: pd.DataFrame = None, cols: list = []) -> None:
    """
    Function to plot the distributions and their potential caps
    :param data: pandas dataframe with col to plotted
    :param cols: list of column names to plot
    :return:
    """

    if len(cols) == 0:
        cols = data.columns

    # filter out object cols
    cols = [col for col in cols if data[col].dtype == "O"]
    tmp = data.copy(deep=True)
    tmp["count"] = [i for i in range(len(data))]
    for col in cols:
        _ = plt.figure(figsize=(6, 6))
        ax = tmp.groupby(col)["count"].count().plot(kind="bar")
        _ = ax.set_title(col + " Counts")

    return None


def plot_correlations(data=None, cols: list = [], plot_title="Feature Correlations"):

    if len(cols) == 0:
        cols = data.columns

    # filter out object cols
    cols = [col for col in cols if data[col].dtype != "O"]

    corr = data[cols].corr()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(
        corr,
        # mask=mask,
        cmap=cmap,
        vmin=corr.values.min(),
        vmax=corr.values.max(),
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    ax.set_title(plot_title)

    return


def plot_outcome_boxes(data=None, outcome: list = [], fts: list = []) -> None:

    if len(fts) > 10:
        fts = np.array_split(fts, math.ceil(len(fts) / 10))
    else:
        fts = [fts]

    for ft in fts:
        for out in outcome:
            tmp_fts = outcome + ft
            df_long = pd.melt(data[tmp_fts], id_vars=out, var_name="Feature")

            # Set up the matplotlib figure
            _ = plt.figure(figsize=(8, 8))
            ax = sns.boxplot(x="Feature", y="value", hue=out, data=df_long)
            _ = ax.set_xticklabels((ax.get_xticklabels()), rotation=90)
            _ = ax.set_title(out)

    return None


def plot_proportions_by_outcome(data=None, outcome: str = None, fts: list = []) -> None:

    for ft in fts:
        _ = plt.figure(figsize=(5, 5))
        ax = sns.countplot(x=ft, hue=outcome, data=data)
        _ = ax.set_title(ft)
        plt.show()

    return None
