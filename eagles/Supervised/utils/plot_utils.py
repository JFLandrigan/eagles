import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


def plot_feature_importance(ft_df=None, mod_name=None, num_top_fts=None, plot_title=""):
    if num_top_fts:
        ft_df = ft_df.head(num_top_fts).copy(deep=True)

    if (
        ("RandomForest" in mod_name)
        or ("GradientBoosting" in mod_name)
        or ("DecisionTree" in mod_name)
        or ("ExtraTrees" in mod_name)
    ):
        plt.figure(figsize=(10, 10), dpi=80, facecolor="w", edgecolor="k")
        ax = sns.barplot(x="Importance", y="Feature", data=ft_df)
        ax.set_title(plot_title)

    elif (
        ("Regression" in mod_name)
        or (mod_name == "Lasso")
        or (mod_name == "ElasticNet")
    ):
        plt.figure(figsize=(10, 10), dpi=80, facecolor="w", edgecolor="k")
        ax = sns.barplot(x="Coef", y="Feature", data=ft_df)
        ax.set_title(plot_title)

    return


def plot_feature_correlations(df=None, plot_title=""):
    corr = df.corr()

    # Generate a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(
        corr,
        # mask=mask,
        cmap="vlag",
        vmin=corr.values.min(),
        vmax=corr.values.max(),
        center=0,
        square=True,
        # linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    ax.set_title(plot_title)

    return


def plot_confusion_matrix(cf=None, labels=None):
    annotate = True
    size = 18

    if len(labels) > 15:
        annotate = False

    if len(labels) > 8:
        size = 12

    cf_df = pd.DataFrame(cf, index=labels, columns=labels)
    cf_df = cf_df.div(cf_df.sum(axis=1), axis=0)

    plt.figure(figsize=(12, 12))
    sns.heatmap(cf_df, annot=annotate, cmap="vlag", annot_kws={"fontsize": size})

    return


def plot_roc_curve(y_true, pred_probs):
    ns_probs = [0 for _ in range(len(y_true))]
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
    clf_fpr, clf_tpr, _ = roc_curve(y_true, pred_probs)

    plt.figure(figsize=(8, 8))
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
    plt.plot(clf_fpr, clf_tpr, marker=".", label="Classifier")
    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    return


def plot_precision_recall_curve(y_true, pred_probs):
    clf_precision, clf_recall, _ = precision_recall_curve(y_true, pred_probs)

    # plot the precision-recall curves
    no_skill = len(y_true[y_true == 1]) / len(y_true)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
    plt.plot(clf_recall, clf_precision, marker=".", label="Classifier")
    # axis labels
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    return


def plot_true_pred_scatter(y_true, y_pred):

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    fig, axs = plt.subplots(nrows=2, figsize=(12, 12))
    # base linear plot looking at cor of pred to test
    sns.regplot(x="y_true", y="y_pred", data=df, ax=axs[0])

    # distribution of true and pred predictions
    num_values = len(df["y_true"])
    tmp_df = pd.DataFrame(
        {
            "type": list(np.repeat("true", num_values))
            + list(np.repeat("pred", num_values)),
            "value": list(df["y_true"]) + list(df["y_pred"]),
        }
    )
    _ = sns.kdeplot(
        tmp_df["value"], hue=tmp_df["type"], bw_method=0.2, label="true", fill=True
    )

    return


def plot_error_distrib(y_true=None, y_pred=None) -> None:
    diffs = y_true - y_pred
    tmp = pd.DataFrame({"errors": diffs})
    plt.figure(figsize=(10, 10))
    _ = sns.histplot(data=tmp, x="errors", kde=True)
    plt.title("Error Distribution (y_true - y_pred)")

    return
