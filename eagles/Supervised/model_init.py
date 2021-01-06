from eagles.Supervised import config
from eagles.Supervised.utils.feature_selection import EaglesFeatureSelection
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Lasso,
    ElasticNet,
    Ridge,
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

import numpy as np

import logging
import warnings

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


def init_model(model=None, params={}, random_seed=None, tune_test=False):

    if model is None:
        logger.warning("NO MODEL PASSED IN")
        return

    random_state_flag = [True if "random_state" in pr else False for pr in params]
    random_state_flag = any(random_state_flag)

    if model not in [
        "linear",
        "svr",
        "vc_clf",
        "vc_regress",
        "knn_clf",
        "knn_regress",
    ] and ("random_state" not in params.keys() and random_state_flag is False):
        if tune_test:
            params["random_state"] = [random_seed]
        else:
            params["random_state"] = random_seed

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
    elif model == "vc_clf":
        if "estimators" not in params.keys():
            params["estimators"] = [
                ("rf", RandomForestClassifier(random_state=params["random_state"])),
                ("lr", LogisticRegression(random_state=params["random_state"])),
            ]
        mod = VotingClassifier(**params)
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
    elif model == "ridge":
        mod = Ridge(**params)
    elif model == "elastic":
        mod = ElasticNet(**params)
    elif model == "svr":
        mod = SVR(**params)
    elif model == "knn_regress":
        mod = KNeighborsRegressor(**params)
    elif model == "ada_regress":
        mod = AdaBoostRegressor(**params)
    elif model == "vc_regress":
        if "estimators" not in params.keys():
            params["estimators"] = [
                ("rf", RandomForestRegressor(random_state=params["random_state"])),
                ("linear", LinearRegression()),
            ]
        mod = VotingRegressor(**params)
    else:
        mod = model

    return mod


def build_pipes(
    mod=None,
    params: dict = None,
    scale: str = None,
    pipe=None,
    select_features: str = None,
    problem_type: str = "clf",
):
    if problem_type == "clf":
        mod_type = "clf"
    else:
        mod_type = "rgr"
    # If pipeline passed in then add on the classifier
    # else init the pipeline with the model
    if pipe:
        pipe.steps.append((mod_type, mod))
        mod = pipe
    else:
        pipe = Pipeline(steps=[(mod_type, mod)])
        mod = pipe

    # If scaling wanted adds the scaling
    if scale:
        if scale == "standard":
            mod.steps.insert(0, ("scale", StandardScaler()))
        elif scale == "minmax":
            mod.steps.insert(0, ("scale", MinMaxScaler()))
        else:
            warnings.warn(
                "scaler not supported expects standard or minmax got: "
                + scale
                + " no scaler added to model"
            )

    # Appends the feature selection wanted
    # if wanted scaling then feature selection is second step (i.e. position 1) else first step (i.e. position 0)
    if scale:
        insert_position = 1
    else:
        insert_position = 0

    if select_features:
        if select_features not in ["eagles", "select_from_model"]:
            warnings.warn(
                "select_features not supported expects eagles or select_from_model got: "
                + str(select_features)
            )

        if select_features == "eagles":
            mod.steps.insert(
                insert_position,
                (
                    "feature_selection",
                    EaglesFeatureSelection(
                        methods=["correlation", "regress"], problem_type=problem_type
                    ),
                ),
            )
        elif select_features == "select_from_model":
            if problem_type == "clf":
                mod.steps.insert(
                    insert_position,
                    (
                        "feature_selection",
                        SelectFromModel(
                            estimator=LogisticRegression(
                                solver="liblinear", penalty="l1"
                            )
                        ),
                    ),
                )
            elif problem_type == "regress":
                mod.steps.insert(
                    insert_position,
                    ("feature_selection", SelectFromModel(estimator=Lasso())),
                )

    # Adjust the params for the model to make sure have appropriate prefix
    if params:
        if problem_type == "clf":
            param_prefix = "clf__"
        else:
            param_prefix = "rgr__"
        params = {
            k if param_prefix in k else param_prefix + k: v for k, v in params.items()
        }
        return [mod, params]
    else:
        return mod
