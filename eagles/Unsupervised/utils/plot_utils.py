import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


def _plot_dendogram(data):
    dendrogram = sch.dendrogram(sch.linkage(data, method="average"))
    return


def plot_score_curve(data=None, metric="max_sil", opt_n_clusters=None):
    if metric in ["max_sil"]:
        y_val = "Silhouette Scores"
    else:
        y_val = "WSS Totals"

    tmp = pd.DataFrame(
        {"Number of Clusters (k)": data["n_clusters"], "y": data["scores"]}
    )

    _ = plt.figure(figsize=(10, 10))
    ax = sns.lmplot(x="Number of Clusters (k)", y="y", data=tmp, fit_reg=False)
    _ = ax.set(xlabel="Number of Clusters (k)", ylabel=y_val)
    _ = plt.axvline(opt_n_clusters)
    _ = plt.title("Number of Clusters by " + y_val)
    _ = plt.show()

    return


def plot_mean_cluster_scores(data=None, plot_scale=None):

    dflong = pd.melt(
        data, id_vars=["Cluster"], var_name="Feature", value_name="Mean Score"
    )

    if plot_scale:
        if plot_scale == "standard":
            scaler = StandardScaler()
            dflong["Mean Score"] = scaler.fit_transform(
                np.array(dflong["Mean Score"]).reshape(-1, 1)
            )
        elif plot_scale == "minmax":
            scaler = MinMaxScaler()
            dflong["Mean Score"] = scaler.fit_transform(
                np.array(dflong["Mean Score"]).reshape(-1, 1)
            )

    num_features = len(set(dflong["Feature"]))
    if num_features <= 10:
        _ = plt.figure(figsize=(10, 10))
        featureBars = sns.barplot(
            x="Mean Score", y="Feature", hue="Cluster", data=dflong
        )
        _ = featureBars.set_xticklabels(featureBars.get_xticklabels(), rotation=90)
        _ = featureBars.set_ylabel("")
        _ = featureBars.set_title("Mean Feature by Cluster")
    else:
        fts = list(set(dflong["Feature"]))
        ft_groups = [fts[i : i + 10] for i in range(0, len(fts), 10)]
        for fts in ft_groups:
            tmp = dflong[dflong["Feature"].isin(fts)]
            _ = plt.figure(figsize=(10, 10))
            featureBars = sns.barplot(
                x="Mean Score", y="Feature", hue="Cluster", data=tmp
            )
            _ = featureBars.set_xticklabels(featureBars.get_xticklabels(), rotation=90)
            _ = featureBars.set_ylabel("")
            _ = featureBars.set_title("Mean Feature by Cluster")

    # box plot code
    # _ = plt.figure(figsize=(10, 10))
    # featureBars = sns.boxplot(x="Feature", y="Mean Score", hue="Cluster", data=dflong)
    # _ = featureBars.set_title("Mean Feature by Cluster")

    return


def plot_ft_relationships(data=None, plot_dims=[]):

    _ = plt.figure(figsize=(10, 10))
    featureRels = sns.pairplot(data[plot_dims], hue="Cluster", corner=True)
    _ = featureRels.fig.suptitle("Feature Relationships by Cluster", fontsize=16)

    return
