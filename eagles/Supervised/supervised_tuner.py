from eagles.Supervised import model_init as mi
from eagles.Supervised.utils import tuner_utils as tu
from eagles.Supervised.utils import plot_utils as pu
from eagles.Supervised.utils import logger_utils as lu
from eagles.Supervised.utils import metric_utils as mu

import time
import pandas as pd
import numpy as np
from IPython.display import display

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
    params=None,
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
    """
    Takes in input and output data, performs selected parameter tuning and then calls the model eval function to test
    model with the best parameters on the desired outcomes
    :param X: default None, expects pandas dataframe with names columns
    :param y: default None, expects pandas series, list or numpy array with same number of samples as X
    :param model: string or sklearn model object, Expects the string abbreviation of the model or a callable sklearn
    model object. Note can also be a Pipeline object with a sklearn model as one of the steps.
    :param params: dict default {}, Expects dictionary containing appropriatly prefixed paramters to search across
    relative to the model
    :param tune_metric: str defaults to 'f1' if classification or 'mse' if regression, Metric to be used during the
    paranmeter tuning phase.
    :param eval_metrics: list defaults to 'f1' if classification or 'mse' if regression. Should contain list of metrics
    wanted for the final model evaluation
    :param num_cv: int default 5, Number of cross validation iterations to be run in model eval and tuning
    :param pipe: sklearn pipeline object default None, If a pipline object is passed in along with a base model to model
    argument the function will append the model to the pipeline.
    :param scale: string default None, Expects either "standard" or "minmax". When set will create a sklearn pipeline
    object with scale and sklearn model
    :param select_features: dict default None, The expected dict should contain pass through arguments for the tuner
    utils select features function. Note the selection happens pre tuning and not during it.
    :param bins: list default None, For binary classification problems determines the number of granularity of the
    probability bins used in the distribution by percent actual output
    :param num_top_fts: int default None, Tells feature importance function to plot top X features. If none all feature
    importances will be plotted
    :param tuner: str default "random_cv", String argument defining the parameter tuning function to be used.
    :param n_iterations: int default 15, used to determine the number of parameter settings that are selected for
    tuning. This argument is ignored when "grid_cv" is used
    :param get_ft_imp: boolean default True, tells model_eval whether or not to get and plot the feature
    importance of the models
    :param n_jobs: int default 1, Expects integer value of number of parallel process to run during model fitting.
    :param random_seed: int default None, Expects integer value for the random seed to be passed into relative
    functions and classes. If not passed will use np.random to set this value
    :param binary: boolean default True, Flag to tell the functions whether or not is a binary classification problem.
    If it is a regression problem this argument is ignored.
    :param log: string or list default None, Expects either a string ("log", "data", "mod") or a list containing these
    keywords to tell the logger what to log. Note when a list is passed in the function will create a directory to store
    the logged out components.
    :param log_name: str default None, prefix name of logged out data. Ignored if log is None
    :param log_path: str default None, path to save log data to. Ignored if no log is None
    :param log_note: str default None, Note to be used in the log that is saved out. Ignored if no log
    :return:
    """

    if random_seed is None:
        random_seed = np.random.randint(1000, size=1)[0]
        print("Random Seed Value: " + str(random_seed))

    # init the model and define the problem type (linear and svr don't take random_state args)
    if model not in ["linear", "svr"] and not params:
        params = {"random_state": random_seed}
    mod_scv = mi.init_model(model=model, params=params)

    problem_type = mi.define_problem_type(mod_scv)
    if problem_type is None:
        logger.warning("Could not detect problem type exiting")
        return

    if pipe and scale:
        logger.warning("ERROR CAN'T PASS IN PIPE OBJECT AND ALSO SCALE ARG")
        return

    if pipe:
        mod_scv, params = mi.build_pipes(mod=mod_scv, params=params, pipe=pipe)
    elif scale:
        mod_scv, params = mi.build_pipes(mod=mod_scv, params=params, scale=scale)

    if select_features:
        print("Selecting features")

        sub_fts, drop_fts = tu.select_features(
            X=X,
            y=y,
            methods=select_features["methods"],
            problem_type=problem_type,
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

    # ensure that a tune metric is defined
    if tune_metric is None:
        if problem_type == "clf":
            tune_metric = "f1"
        else:
            tune_metric = "neg_mean_squared_error"

    # ensure that eval metrics have been defined
    if len(eval_metrics) == 0 and problem_type == "clf":
        eval_metrics = ["f1"]
    elif len(eval_metrics) == 0 and problem_type == "regress":
        eval_metrics = ["mse"]

    # set up the parameter search object
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

    test_params = params.copy()

    mod = scv.best_estimator_
    params = mod.get_params()
    print("Parameters of the best model: \n")
    if type(mod).__name__ == "Pipeline":
        print(type(mod.named_steps["clf"]).__name__ + " Parameters")
        print(str(mod.named_steps["clf"].get_params()) + "\n")

    elif "Voting" in type(mod).__name__:
        print(
            type(mod).__name__ + " weights: " + str(mod.get_params()["weights"]) + "\n"
        )
        for c in mod.estimators_:
            if type(c).__name__ == "Pipeline":
                print(type(c.named_steps["clf"]).__name__ + " Parameters")
                print(str(c.named_steps["clf"].get_params()) + "\n")
            else:
                print(type(c).__name__ + " Parameters")
                print(str(c.get_params()) + "\n")
    else:
        print(type(mod).__name__ + " Parameters")
        print(str(mod.get_params()) + "\n")

    print("Performing model eval on best estimator")

    log_data = model_eval(
        X=X,
        y=y,
        model=mod,
        params={},
        metrics=eval_metrics,
        bins=bins,
        pipe=None,
        scale=None,
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

        log_data["test_params"] = test_params
        log_data["tune_metric"] = tune_metric
        if log_note:
            log_data["note"] = log_note

        if isinstance(log, list):
            log_path, log_name, timestr = lu.construct_save_path(
                fl_path=log_path,
                fl_name=log_name,
                model_name=log_data["model"],
                save_dir=True,
            )
        else:
            log_path, log_name, timestr = lu.construct_save_path(
                fl_path=log_path,
                fl_name=log_name,
                model_name=log_data["model"],
                save_dir=False,
            )

        if isinstance(log, list):
            for x in log:
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
                    lu.pickle_data(
                        data=mod, fl_path=log_path, fl_name=log_name, data_type=x
                    )
                else:
                    logger.warning("LOG TYPE NOT SUPPORTED: " + x)
        elif log == "log":
            lu.log_results(
                fl_name=log_name, fl_path=log_path, log_data=log_data, tune_test=True
            )
        else:
            logger.warning("LOG TYPE NOT SUPPORTED: " + str(log))

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
    :param model: Model object containing fit, predict, predict_proba attributes, sklearn pipeline object or
    string indicator of model to eval
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
    :param tune_test: boolean default False, Used as a pass through argument from the tune_test_model function
    :return: model fitted on last cross validation set and last set of cv data for predictions
    """

    if random_seed is None:
        random_seed = np.random.randint(1000, size=1)[0]
    print("Random Seed Value: " + str(random_seed))

    mod = mi.init_model(model=model, params=params)
    problem_type = mi.define_problem_type(mod=mod)
    if problem_type is None:
        logger.warning("Could not detect problem type exiting")
        return

    if pipe:
        mod = mi.build_pipes(mod=mod, pipe=pipe)
    elif scale:
        mod = mi.build_pipes(mod=mod, scale=scale)

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

        print("Finished cv run: " + str(cnt) + " time: " + str(time.time() - cv_st))
        cnt += 1

    print("\nCV Run Scores")
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
        display(bt)

    if "roc_auc" in metrics:
        pu.plot_roc_curve(y_true=y_test, pred_probs=pred_probs)
    if "precision_recall_auc" in metrics:
        pu.plot_precision_recall_curve(y_true=y_test, pred_probs=pred_probs)

    if get_ft_imp:
        ft_imp_df = tu.feature_importances(mod=mod, X=X, num_top_fts=num_top_fts)

    # create a copy of the final testing data and append the predictions, pred probs and true values
    fin_test_df = X_test.copy(deep=True)
    fin_test_df["true_labels"] = y_test
    fin_test_df["preds"] = preds
    fin_test_df["pred_probs"] = pred_probs

    if log:
        log_data = {
            "features": list(X.columns),
            "random_seed": random_seed,
            "metrics": metric_dictionary,
            "params": list(),
        }

        if type(mod).__name__ == "Pipeline":
            log_data["params"].append(
                [
                    type(mod.named_steps["clf"]).__name__,
                    str(mod.named_steps["clf"].get_params()),
                ]
            )

        elif "Voting" in type(mod).__name__:
            log_data["params"].append(
                str([type(mod).__name__, str(mod.get_params()["weights"])])
            )
            for c in mod.estimators_:
                if type(c).__name__ == "Pipeline":
                    log_data["params"].append(
                        [
                            type(c.named_steps["clf"]).__name__,
                            str(c.named_steps["clf"].get_params()),
                        ]
                    )
                else:
                    log_data["params"].append(
                        [type(c).__name__, str(c.get_params()),]
                    )
        else:
            log_data["params"].append([type(mod).__name__, str(mod.get_params())])

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
        if tune_test:
            return log_data
        else:
            lu.log_results(
                fl_name=log_name, fl_path=log_path, log_data=log_data, tune_test=False
            )
            return [mod, fin_test_df]

    return [mod, fin_test_df]
