import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from math import sqrt


def root_mean_squared_error(y_true, preds):
    return sqrt(mean_squared_error(y_true, preds))


def mean_absolute_percentage_error(y_true, y_pred):

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df = df[df["y_true"] != 0].copy(deep=True)

    y_true = df["y_true"]
    y_pred = df["y_pred"]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def precision_recall_auc(y_true=None, pred_probabs=None):

    clf_precision, clf_recall, _ = precision_recall_curve(y_true, pred_probabs)
    score = auc(clf_recall, clf_precision)

    return score


def init_model_metrics(metrics=[]):
    """
    Function to init dictionary that stores metric functions and metric scores
    :param metrics: list of strings for metrics to store in dictionary
    :return: dictionary that with _func _score metric pairings
    """
    metric_dictionary = {}

    # Classification Metrics
    if "accuracy" in metrics:
        metric_dictionary["accuracy_func"] = accuracy_score
        metric_dictionary["accuracy_scores"] = np.array([])

    if "f1" in metrics:
        metric_dictionary["f1_func"] = f1_score
        metric_dictionary["f1_scores"] = np.array([])

    if "precision" in metrics:
        metric_dictionary["precision_func"] = precision_score
        metric_dictionary["precision_scores"] = np.array([])

    if "recall" in metrics:
        metric_dictionary["recall_func"] = recall_score
        metric_dictionary["recall_scores"] = np.array([])

    if "roc_auc" in metrics:
        metric_dictionary["roc_auc_func"] = roc_auc_score
        metric_dictionary["roc_auc_scores"] = np.array([])

    if "precision_recall_auc" in metrics:
        metric_dictionary["precision_recall_auc_func"] = precision_recall_auc
        metric_dictionary["precision_recall_auc_scores"] = np.array([])

    # Regression Metrics
    if "mse" in metrics:
        metric_dictionary["mse_func"] = mean_squared_error
        metric_dictionary["mse_scores"] = np.array([])

    if "rmse" in metrics:
        metric_dictionary["rmse_func"] = root_mean_squared_error
        metric_dictionary["rmse_scores"] = np.array([])

    if "mae" in metrics:
        metric_dictionary["mae_func"] = mean_absolute_error
        metric_dictionary["mae_scores"] = np.array([])

    if "mape" in metrics:
        metric_dictionary["mape_func"] = mean_absolute_percentage_error
        metric_dictionary["mape_scores"] = np.array([])

    if "r2" in metrics:
        metric_dictionary["r2_func"] = r2_score
        metric_dictionary["r2_scores"] = np.array([])

    return metric_dictionary

def calc_metrics(
    metrics=None,
    metric_dictionary=None,
    y_test=None,
    preds=None,
    pred_probs=None,
    avg="binary",
):
    for metric in metrics:
        if metric not in [
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "precision_recall_auc",
        ]:
            metric_dictionary[metric + "_scores"] = np.append(
                metric_dictionary[metric + "_scores"],
                metric_dictionary[metric + "_func"](y_test, preds),
            )
        elif metric in ["f1", "precision", "recall"]:
            metric_dictionary[metric + "_scores"] = np.append(
                metric_dictionary[metric + "_scores"],
                metric_dictionary[metric + "_func"](y_test, preds, average=avg),
            )
        elif metric in ["roc_auc", "precision_recall_auc"]:
            metric_dictionary[metric + "_scores"] = np.append(
                metric_dictionary[metric + "_scores"],
                metric_dictionary[metric + "_func"](y_test, pred_probs),
            )

    return metric_dictionary