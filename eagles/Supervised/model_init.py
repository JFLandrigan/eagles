from eagles.Supervised import config
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

import numpy as np

import logging

logger = logging.getLogger(__name__)


def define_problem_type(mod=None, y=None):

    problem_type = None

    if type(mod).__name__ in config.clf_models:
        problem_type = "clf"
    elif type(mod).__name__ in config.regress_models:
        problem_type = "regress"
    elif type(mod).__name__ == "Pipeline":
        if type(mod.named_steps["clf"]).__name__ in config.clf_models:
            problem_type = "clf"
        elif type(mod.named_steps["clf"]).__name__ in config.regress_models:
            problem_type = "regress"
        elif "Classifier" in type(mod).__name__:
            problem_type = "clf"
        elif "Regressor" in type(mod).__name__:
            problem_type = "regress"
    elif "Classifier" in type(mod).__name__:
        problem_type = "clf"
    elif "Regressor" in type(mod).__name__:
        problem_type = "regress"

    if problem_type is None:
        if len(np.unique(y)) == 2:
            problem_type = "clf"

    return problem_type


def init_model(model=None, params={}):

    if model is None:
        logger.warning("NO MODEL PASSED IN")
        return

    if model == "rf_clf":
        mod = RandomForestClassifier(**params)
    elif model == "et_clf":
        mod = ExtraTreesClassifier(**params)
    elif model == "gb_clf":
        mod = GradientBoostingClassifier(**params)
    elif model == "dt_clf":
        mod = DecisionTreeClassifier(**params)
    elif model == "logistic":
        mod = LogisticRegression(**params)
    elif model == "svc":
        mod = SVC(**params)
    elif model == "knn_clf":
        mod = KNeighborsClassifier(**params)
    elif model == "nn":
        mod = MLPClassifier(**params)
    elif model == "ada_clf":
        mod = AdaBoostClassifier(**params)
    elif model == "rf_regress":
        mod = RandomForestRegressor(**params)
    elif model == "et_regress":
        mod = ExtraTreesRegressor(**params)
    elif model == "gb_regress":
        mod = GradientBoostingRegressor(**params)
    elif model == "dt_regress":
        mod = DecisionTreeRegressor(**params)
    elif model == "linear":
        mod = LinearRegression(**params)
    elif model == "lasso":
        mod = Lasso(**params)
    elif model == "elastic":
        mod = ElasticNet(**params)
    elif model == "svr":
        mod = SVR(**params)
    elif model == "knn_regress":
        mod = KNeighborsRegressor(**params)
    elif model == "ada_regress":
        mod = AdaBoostRegressor(**params)
    else:
        mod = model

    return mod


def build_pipes(mod=None, params=None, scale=None, pipe=None):

    if pipe:
        pipe.steps.append(["clf", mod])
        mod = pipe

    elif scale:
        if scale == "standard":
            mod = Pipeline([("scale", StandardScaler()), ("clf", mod)])
        elif scale == "minmax":
            mod = Pipeline([("scale", MinMaxScaler()), ("clf", mod)])

    if params:
        params = {k if "clf__" in k else "clf__" + k: v for k, v in params.items()}
        return [mod, params]
    else:
        return mod
