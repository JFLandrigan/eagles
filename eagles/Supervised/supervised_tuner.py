from eagles.Supervised import model_init as mi
from eagles.Supervised.utils import tuner_utils as tu
from eagles.Supervised.utils import plot_utils as pu
from eagles.Supervised.utils import logger_utils as lu
from eagles.Supervised.utils import metric_utils as mu

import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

import logging
logger = logging.getLogger(__name__)

def tune_test_model(
    X=None,
    y=None,
    model=None,
    params={},
    tune_metric=None,
    eval_metrics=[],
    num_cv=5,
    pipe=None,
    scale=None,
    select_features=None,
    bins=None,
    num_top_fts=None,
    tuner="random_cv",
    n_iterations=15,
    get_ft_imp=True,
    n_jobs=1,
    random_seed=None,
    binary=True,
    log="log",
    log_name=None,
    log_path=None,
    log_note=None,
):

    if random_seed is None:
        random_seed = np.random.randint(1000, size=1)[0]
        print("Random Seed Value: " + str(random_seed))

    if select_features:
        print("Selecting features")

        sub_fts, drop_fts = tu.select_features(
            X=X,
            y=y,
            methods=select_features["methods"],
            problem_type="clf",
            model_pipe=select_features["model_pipe"],
            imp_thresh=select_features["imp_thresh"],
            corr_thresh=select_features["corr_thresh"],
            bin_fts=select_features["bin_fts"],
            dont_drop=select_features["dont_drop"],
            random_seed=random_seed,
            n_jobs=n_jobs,
            plot_ft_importance=select_features["plot_ft_importance"],
            plot_ft_corr=select_features["plot_ft_corr"],
        )

        X = X[sub_fts].copy(deep=True)

        features = sub_fts.copy()
    else:
        features = list(X.columns)

    # init the model and define the problem type (linear and svr don't take random_state args)
    if model not in ["linear", "svr"] and not params:
        params = {"random_state": random_seed}
    mod_scv = mi.init_model(model=model, params=params)

    if tune_metric is None or len(eval_metrics) == 0:
        problem_type = mi.define_problem_type(mod_scv)

        if tune_metric is None and problem_type == "clf":
            tune_metric = "f1"
        else:
            tune_metric = "neg_mean_squared_error"

        if len(eval_metrics) == 0 and problem_type == "clf":
            eval_metrics = ["f1"]
        else:
            eval_metrics = ["mse"]

    if pipe and scale:
        logger.warning("ERROR CAN'T PASS IN PIPE OBJECT AND ALSO SCALE ARG")
        return

    if pipe:
        tmp_mod_scv = pipe
        tmp_mod_scv.steps.append(["clf", mod_scv])
        mod_scv = tmp_mod_scv
        params = {k if "clf__" in k else "clf__" + k: v for k, v in params.items()}

    elif scale:
        if scale == "standard":
            mod_scv = Pipeline([("scale", StandardScaler()), ("clf", mod_scv)])
            params = {k if "clf__" in k else "clf__" + k: v for k, v in params.items()}
        elif scale == "minmax":
            mod_scv = Pipeline([("scale", MinMaxScaler()), ("clf", mod_scv)])
            params = {k if "clf__" in k else "clf__" + k: v for k, v in params.items()}

    if tuner == "random_cv":
        scv = RandomizedSearchCV(
            mod_scv,
            param_distributions=params,
            n_iter=n_iterations,
            scoring=tune_metric,
            cv=num_cv,
            n_jobs=n_jobs,
            verbose=2,
            random_state=random_seed,
        )

    elif tuner == "bayes_cv":
        scv = BayesSearchCV(
            estimator=mod_scv,
            search_spaces=params,
            n_iter=n_iterations,
            cv=num_cv,
            verbose=2,
            refit=True,
            n_jobs=n_jobs,
        )

    elif tuner == "grid_cv":
        scv = GridSearchCV(
            mod_scv,
            param_grid=params,
            scoring=tune_metric,
            cv=num_cv,
            n_jobs=n_jobs,
            verbose=1,
        )

    else:
        print("TUNING SEARCH NOT SUPPORTED")
        return

    scv.fit(X, y)
    print("Best score for grid search: " + str(scv.best_score_))

    mod = scv.best_estimator_
    params = mod.get_params()
    print("Parameters of the best model: \n")
    for pr in mod.get_params():
        print(pr + ' :' + str(mod.get_params()[pr]))

    print("\n")

    print("Performing model eval on best estimator")

    log_data = model_eval(
        X=X,
        y=y,
        model=mod,
        params={},
        metrics=eval_metrics,
        bins=bins,
        pipe=pipe,
        scale=scale,
        num_top_fts=num_top_fts,
        num_cv=num_cv,
        get_ft_imp=get_ft_imp,
        random_seed=random_seed,
        binary=binary,
        log=log,
        log_name=log_name,
        log_path=log_path,
        tune_test=True,
    )

    if log:

        log_data["test_params"] = params
        log_data["tune_metric"] = tune_metric
        if log_note:
            log_data["note"] = log_note

        if isinstance(log, list):
            log_path, log_name = lu.construct_save_dir(
                fl_path=None, fl_name=None, model_name=None
            )
        else:
            log_path, log_name, timestr = lu.construct_save_path(
                fl_path=None, fl_name=None, model_name=None
            )

        if isinstance(log, list):
            for x in log:
                print("Saving out the: " + x)
                if x == "log":
                    lu.log_results(
                        fl_name=log_name,
                        fl_path=log_path,
                        log_data=log_data,
                        tune_test=True,
                    )
                elif x == "data":
                    if ~isinstance(X, pd.DataFrame):
                        X = pd.DataFrame(X)

                    tmp_data = X.copy(deep=True)
                    tmp_data["y_true"] = y
                    lu.pickle_data(
                        data=tmp_data, fl_path=log_path, fl_name=log_name, data_type=x
                    )
                elif x == "mod":
                    lu.pickle_data(data=mod, fl_path=log_path, fl_name=log_name, data_type=x)
                else:
                    logger.warning("LOG TYPE NOT SUPPORTED: " + x)

        if log == "log":
            lu.log_results(
                fl_name=log_name, fl_path=log_path, log_data=log_data, tune_test=True
            )
        else:
            logger.warning("LOG TYPE NOT SUPPORTED: " + log)

    return [mod, params, features]


def model_eval(
    X=None,
    y=None,
    model=None,
    params={},
    metrics=[],
    bins=None,
    pipe=None,
    scale=None,
    num_top_fts=None,
    num_cv=5,
    get_ft_imp=False,
    random_seed=None,
    binary=True,
    log="log",
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

    mod = mi.init_model(model=model, params=params)
    problem_type = mi.define_problem_type(mod=mod)
    if len(metrics) == 0:
        if problem_type == "clf":
            metrics = ["f1"]
        else:
            metrics = ["mse"]

    print("Performing CV Runs: " + str(num_cv))
    kf = KFold(n_splits=num_cv, shuffle=True, random_state=random_seed)

    if binary:
        avg = "binary"
    else:
        avg = "macro"

    metric_dictionary = mu.init_model_metrics(metrics=metrics)

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

        metric_dictionary = mu.calc_metrics(
            metrics=metrics,
            metric_dictionary=metric_dictionary,
            y_test=y_test,
            preds=preds,
            pred_probs=pred_probs,
            avg=avg,
        )

        print(
            "Finished cv run: "
            + str(cnt)
            + " time: "
            + str(time.time() - cv_st)
            + " \n"
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


        # if called from tune test then return the log data for final appending before logout
        # else log out the data and then return the final dictionary
        # TODO add in funcitonality to log out the model and the data in a dir just like the tune and tester
        if tune_test:
            return log_data
        
        else:
            lu.log_results(
                fl_name=log_name, fl_path=log_path, log_data=log_data, tune_test=False
            )

            return log_data

    return
