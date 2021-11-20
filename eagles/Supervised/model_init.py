from eagles.Supervised import config
from eagles.Supervised.utils.feature_selection import EaglesFeatureSelection
from sklearn.feature_selection import SelectFromModel, SelectKBest
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
    PoissonRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.impute import (
    SimpleImputer,
    MissingIndicator,
    KNNImputer,
)  # , IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold

from xgboost import XGBRegressor, XGBClassifier

import numpy as np

import logging
import warnings

logger = logging.getLogger(__name__)


def init_cv_splitter(cv_method=None, num_cv: int = 5, random_seed: int = None):
    if type(cv_method) == str:
        if cv_method == "kfold":
            splitter = KFold(n_splits=num_cv, shuffle=True, random_state=random_seed)
        elif cv_method == "time":
            splitter = TimeSeriesSplit(n_splits=num_cv)
        elif cv_method == "stratified_k":
            splitter = StratifiedKFold(n_splits=num_cv, random_state=random_seed)
    else:
        splitter = cv_method
    return splitter


def init_model(model=None, params={}, random_seed=None, tune_test=False):

    if model is None:
        logger.warning("NO MODEL PASSED IN")
        return

    random_state_flag = [True if "random_state" in pr else False for pr in params]
    random_state_flag = any(random_state_flag)

    if (
        model
        not in [
            "linear",
            "svr",
            "vc_clf",
            "vc_regress",
            "knn_clf",
            "knn_regress",
            "poisson",
        ]
        and ("random_state" not in params.keys() and random_state_flag is False)
    ):
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
    elif model == "ada_clf":
        mod = AdaBoostClassifier(**params)
    elif model == "vc_clf":
        if "estimators" not in params.keys():
            params["estimators"] = [
                ("rf", RandomForestClassifier()),
                ("lr", LogisticRegression()),
            ]
        mod = VotingClassifier(**params)
    elif model == "xgb_clf":
        mod = XGBClassifier(**params)
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
    elif model == "poisson":
        mod = PoissonRegressor(**params)
    elif model == "svr":
        mod = SVR(**params)
    elif model == "knn_regress":
        mod = KNeighborsRegressor(**params)
    elif model == "ada_regress":
        mod = AdaBoostRegressor(**params)
    elif model == "vc_regress":
        if "estimators" not in params.keys():
            params["estimators"] = [
                ("rf", RandomForestRegressor()),
                ("linear", LinearRegression()),
            ]
        mod = VotingRegressor(**params)
    elif model == "xgb_regress":
        mod = XGBRegressor(**params)
    else:
        mod = model

    return mod


def build_pipes(
    mod=None,
    pipe=None,
    params: dict = None,
    imputer: str or object = None,
    scale: str or object = None,
    select_features: str or object = None,
    mod_type: str = "clf",
    num_features: int = None,
):

    # If pipeline passed in then add on the classifier
    # else init the pipeline with the model
    if pipe:
        pipe.steps.append((mod_type, mod))
        mod = pipe
    else:
        pipe = Pipeline(steps=[(mod_type, mod)])
        mod = pipe

    # inserts imputation method into the first position in the pipe
    if imputer:
        if type(imputer) == str:
            if "simple" in imputer:
                mod.steps.insert(
                    0, ("impute", SimpleImputer(strategy=imputer.split("_")[1]))
                )
            elif imputer == "missing_indicator":
                mod.steps.insert(0, ("impute", MissingIndicator()))
            elif imputer == "knn":
                mod.steps.insert(0, ("impute", KNNImputer()))
        else:
            mod.steps.insert(0, ("impute", imputer))

    # If scaling wanted adds the scaling
    if scale:
        if imputer:
            insert_pos = 1
        else:
            insert_pos = 0
        if type(scale) == str:
            if scale == "standard":
                mod.steps.insert(insert_pos, ("scale", StandardScaler()))
            elif scale == "minmax":
                mod.steps.insert(insert_pos, ("scale", MinMaxScaler()))
            elif scale == "robust":
                mod.steps.insert(insert_pos, ("scale", RobustScaler()))
        else:
            mod.steps.insert(insert_pos, ("scale", scale))

    # Appends the feature selection wanted
    # if wanted scaling then feature selection is second step (i.e. position 1) else first step (i.e. position 0)

    if select_features:

        if scale and imputer:
            insert_position = 2
        elif scale or imputer:
            insert_position = 1
        else:
            insert_position = 0

        if type(select_features) == str:
            if select_features not in ["eagles", "select_from_model", "selectkbest"]:
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
                            methods=["correlation", "regress"], problem_type=mod_type
                        ),
                    ),
                )
            elif select_features == "select_from_model":
                if mod_type == "clf":
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
                elif mod_type == "rgr":
                    mod.steps.insert(
                        insert_position,
                        (
                            "feature_selection",
                            SelectFromModel(estimator=Lasso()),
                        ),
                    )
            elif select_features == "selectkbest":
                if num_features < 10:
                    k = np.ceil(num_features / 2).astype(int)
                else:
                    k = 10

                mod.steps.insert(
                    insert_position,
                    (
                        "feature_selection",
                        SelectKBest(k=k),
                    ),
                )
        else:
            mod.steps.insert(
                insert_position,
                select_features,
            )

    pipe_steps = ""
    for k in mod.named_steps.keys():
        pipe_steps = pipe_steps + type(mod.named_steps[k]).__name__ + ", "
    print("Final pipeline: " + pipe_steps)

    # Adjust the params for the model to make sure have appropriate prefix
    if len(params) > 0:
        param_prefix = mod_type + "__"

        params = {
            k if param_prefix in k else param_prefix + k: v for k, v in params.items()
        }
        return [mod, params]
    else:
        return [mod, params]
