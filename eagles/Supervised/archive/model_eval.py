from Model_Tuner.Supervised import supervised_tuner as st
from Model_Tuner.Supervised.utils import tuner_utils as tu
from Model_Tuner.Supervised.utils import plot_utils as pu
from Model_Tuner.Supervised.utils import logger_utils as lu
from Model_Tuner.Supervised import config

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy as np

import time


def define_problem_type(mod=None):
    if type(mod).__name__ in config.clf_models:
        problem_type = "clf"
    elif type(mod).__name__ in config.regress_models:
        problem_type = "regress"
    elif type(mod).__name__ == "Pipeline":
        if type(mod.named_steps["clf"]).__name__ in config.clf_models:
            problem_type = "clf"
        else:
            problem_type = "regress"
    else:
        print("WARNING COULD NOT INFER PROBLEM TYPE. ENSURE MODEL IS SUPPORTED")
        return
    return problem_type


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


def model_eval(
    X=None,
    y=None,
    model=None,
    params={},
    metrics=["f1"],
    bins=None,
    pipe=None,
    scale=None,
    num_top_fts=None,
    num_cv=5,
    get_ft_imp=False,
    random_seed=None,
    binary=True,
    log=True,
    log_name=None,
    log_path=None,
    log_note=None,
    tune_test=False,
):
    """
    Model Eval function. Used to perform cross validation on model and is automatically called post tune_test_model
    :param X: pandas dataframe containing features for model training
    :param y: series or np array containing prediction values
    :param model: Model object containing fit, predict, predict_proba attributes, sklearn pipeline object or string indicator of model to eval
    :param params: dictionary containing parameters of model to fit on
    :param metrics: list of metrics to eval model on default is ['f1]
    :param bins: list of bin ranges to output the score to percent actual distribution
    :param pipe: Sklearn pipeline object without classifier
    :param scale: string Standard or MinMax indicating to scale the features during cross validation
    :param num_top_fts: int number of top features to be plotted
    :param num_cv: int number of cross validations to do
    :param get_ft_imp: boolean indicating to get and plot the feature importances
    :param random_seed: int for random seed setting
    :param binary: boolean indicating if model predictions are binary or multi-class
    :param log: boolean indicator to log out results
    :param log_name: string name of the logger doc
    :param log_path: string path to store logger doc if none data dir in model tuner dir is used
    :param log_note: string containing note to add at top of logger doc
    :param tune_test:
    :return:
    """

    if random_seed is None:
        random_seed = np.random.randint(1000, size=1)[0]
    print("Random Seed Value: " + str(random_seed))

    mod = st.init_model(model=model, params=params)
    problem_type = define_problem_type(mod=mod)

    print("Performing CV Runs: " + str(num_cv))
    kf = KFold(n_splits=num_cv, shuffle=True, random_state=random_seed)

    if binary:
        avg = "binary"
    else:
        avg = "macro"

    metric_dictionary = tu.init_model_metrics(metrics=metrics)

    cnt = 1
    for train_index, test_index in kf.split(X):
        cv_st = time.time()

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if pipe and tune_test == False:
            tmp_mod = pipe
            tmp_mod.steps.append(["clf", mod])
            mod = tmp_mod
            params = {"clf__" + k: v for k, v in params.items()}

        elif scale and tune_test == False:
            if scale == "standard":
                scaler = StandardScaler()
            elif scale == "minmax":
                scaler = MinMaxScaler()

            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        mod.fit(X_train, y_train)
        preds = mod.predict(X_test)

        if problem_type == "clf":
            pred_probs = mod.predict_proba(X_test)[:, 1]
        else:
            pred_probs = []

        metric_dictionary = get_metrics(
            metrics=metrics,
            metric_dictionary=metric_dictionary,
            preds=preds,
            pred_probs=pred_probs,
            avg=avg,
        )

        print(
            "Finished cv run: "
            + str(cnt)
            + " time: "
            + str(time.time() - cv_st)
            + " \n \n"
        )
        cnt += 1

    print("CV Run Scores")
    for metric in metrics:
        print(metric + " scores: " + str(metric_dictionary[metric + "_scores"]))
        print(metric + " mean: " + str(metric_dictionary[metric + "_scores"].mean()))
        print(
            metric
            + " standard deviation: "
            + str(metric_dictionary[metric + "_scores"].std())
            + " \n"
        )

    print(" \n")

    print("Final cv train test split")
    for metric in metrics:
        print(metric + " score: " + str(metric_dictionary[metric + "_scores"][-1]))

    if problem_type == "clf":
        print(" \n")
        cf = confusion_matrix(y_test, preds)
        cr = classification_report(
            y_test, preds, target_names=[str(x) for x in mod.classes_]
        )

        pu.plot_confusion_matrix(cf=cf, labels=mod.classes_)
        print(cr)

    if binary and problem_type == "clf":
        prob_df = pd.DataFrame({"probab": pred_probs, "actual": y_test})
        bt = tu.create_bin_table(
            df=prob_df, bins=bins, bin_col="probab", actual_col="actual"
        )
        print(bt)

    if "roc_auc" in metrics:
        pu.plot_roc_curve(y_true=y_test, pred_probs=pred_probs)
    if "precision_recall_auc" in metrics:
        pu.plot_precision_recall_curve(y_true=y_test, pred_probs=pred_probs)

    if get_ft_imp:
        ft_imp_df = tu.feature_importances(mod=mod, X=X, num_top_fts=num_top_fts)

    if log:
        log_data = {
            "features": list(X.columns),
            "random_seed": random_seed,
            "params": mod.get_params(),
            "metrics": metric_dictionary,
        }

        if problem_type == "clf":
            log_data["cf"] = cf
            log_data["cr"] = cr

        if type(mod).__name__ == "Pipeline":
            log_data["model"] = type(mod).__name__
            pipe_steps = "Pipe steps: "
            for k in mod.named_steps.keys():
                pipe_steps = pipe_steps + type(mod.named_steps[k]).__name__ + " "
            log_data["pipe_steps"] = pipe_steps
        else:
            log_data["model"] = type(mod).__name__

        if log_note:
            log_data["note"] = log_note

        if binary and problem_type == "clf":
            log_data["bin_table"] = bt

        if get_ft_imp:
            log_data["ft_imp_df"] = ft_imp_df

        if tune_test:
            return log_data
        else:
            lu.log_results(
                fl_name=log_name, fl_path=log_path, log_data=log_data, tune_test=False
            )

    return
