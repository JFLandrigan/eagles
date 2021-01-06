from eagles.Supervised.utils import plot_utils as pu
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.base import clone
import numpy as np
import pandas as pd
import math
from IPython.display import display
import logging

logger = logging.getLogger(__name__)

# TODO add in low variance feature selection methods
# TODO add in feature selection based on significance from coef p vals
# TODO adjust so that if need scaling prefit of the regress models it will do it


class EaglesFeatureSelection:
    def __init__(
        self,
        methods=[],
        problem_type="clf",
        model_pipe=None,  # TODO change this to scale arg
        rf_imp_thresh=0.005,
        corr_thresh=0.9,
        percent_rank_drop=0.1,  # proportion to drop from bottom of avg rank between models
        class_weight=None,
        bin_fts=[],
        dont_drop=[],
        random_seed=None,
        n_jobs=None,
        plot_ft_importance=False,
        plot_ft_corr=False,
        print_out=False,
    ):
        self.name = "EaglesFeatureSelection"
        self.features = []
        self.sub_features = []
        self.imp_drop = []
        self.lin_drop = []
        self.rank_drop = []
        self.corr_drop = []
        self.drop_fts = []
        self.methods = methods
        self.problem_type = problem_type
        self.model_pipe = model_pipe  # TODO change this to scale arg
        self.rf_imp_thresh = rf_imp_thresh
        self.corr_thresh = corr_thresh
        self.percent_rank_drop = percent_rank_drop
        self.class_weight = class_weight
        self.bin_fts = bin_fts
        self.dont_drop = dont_drop
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.plot_ft_importance = plot_ft_importance
        self.plot_ft_corr = plot_ft_corr
        self.print_out = print_out

    def get_support(self, indeces: bool = False) -> list:
        """
        Function to return boolean mask or indeces of selected features
        :param indeces: bool
        :return: list of int indeces or bool mask
        """
        if not indeces:
            mask = [True if ft in self.sub_features else False for ft in self.features]
            return mask
        else:
            inds = [
                i
                for i in range(len(self.features))
                if self.features[i] in self.sub_features
            ]
            return inds

    def _drop_correlated_features(self, X):

        X_tmp = X.copy(deep=True)

        # correlation drop
        corr_fts = [x for x in X_tmp.columns if x not in self.bin_fts]
        correlations = X_tmp[corr_fts].corr()

        if self.plot_ft_corr:
            pu.plot_feature_correlations(
                df=X_tmp[corr_fts].copy(deep=True),
                plot_title="Feature Correlation Pre-Drop",
            )

        upper = correlations.where(
            np.triu(np.ones(correlations.shape), k=1).astype(np.bool)
        )
        self.corr_drop = [
            column
            for column in upper.columns
            if any(upper[column].abs() > self.corr_thresh)
        ]

        # drop the correlation features first then fit the models
        if self.print_out:
            print(
                "Features dropping due to high correlation: "
                + str(self.corr_drop)
                + " \n"
            )
        self.sub_features = [
            x
            for x in self.features
            if (x not in self.corr_drop) or (x in self.dont_drop)
        ]

        return

    def _drop_rf_low_importance(self, X, y):
        if self.problem_type == "clf":
            forest = RandomForestClassifier(
                n_estimators=200,
                class_weight=self.class_weight,
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
                verbose=0,
            )
        else:
            forest = RandomForestRegressor(
                n_estimators=200,
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
                verbose=0,
            )

        # remove this step from the rf wouldn't scale features unless can think or another
        # preproc step that may pass in like pca that would then run feature select on
        if self.model_pipe:
            tmp_pipe = clone(self.model_pipe)
            tmp_pipe.steps.append(["mod", forest])
            forest = clone(tmp_pipe)

        forest.fit(X[self.sub_features], y)

        if self.model_pipe:
            forest = forest.named_steps["mod"]

        ftImp = {
            "Feature": self.sub_features,
            "Importance": forest.feature_importances_,
        }
        ftImp_df = pd.DataFrame(ftImp)
        ftImp_df.sort_values(["Importance"], ascending=False, inplace=True)
        self.imp_drop = list(
            ftImp_df[ftImp_df["Importance"] < self.rf_imp_thresh]["Feature"]
        )

        if self.print_out:
            print(
                "Features dropping from low importance: " + str(self.imp_drop) + " \n"
            )

        if self.plot_ft_importance:
            pu.plot_feature_importance(
                ft_df=ftImp_df,
                mod_type=type(forest).__name__,
                plot_title="RF Feature Selection Importance",
            )
        return

    def _regress_drop(self, X, y):

        X_tmp = X.copy(deep=True)

        if self.problem_type == "clf":
            if len(set(y)) > 2:
                solver = "saga"
                multi_class = "multinomial"
            else:
                solver = "saga"
                multi_class = "auto"

            lin_mod = LogisticRegression(
                penalty="l1",
                solver=solver,
                multi_class=multi_class,
                class_weight=self.class_weight,
                random_state=self.random_seed,
            )
        else:
            lin_mod = Lasso(random_state=self.random_seed)

        # remove this step from the rf wouldn't scale features unless can think
        # or another preproc step that may pass in like pca that would then run feature select on
        # or just make this a scale logic
        if self.model_pipe:
            tmp_pipe = clone(self.model_pipe)
            tmp_pipe.steps.append(["mod", lin_mod])
            lin_mod = clone(tmp_pipe)

        if self.print_out:
            print(X[self.sub_features])
        lin_mod.fit(X_tmp[self.sub_features], y)

        if self.model_pipe:
            lin_mod = lin_mod.named_steps["mod"]

        if self.problem_type == "clf":
            tmp = pd.DataFrame({"Feature": self.sub_features, "Coef": lin_mod.coef_[0]})
        else:
            tmp = pd.DataFrame({"Feature": self.sub_features, "Coef": lin_mod.coef_})

        self.lin_drop = list(tmp["Feature"][tmp["Coef"] == 0])
        if self.print_out:
            print("Features dropping from l1 regression: " + str(self.lin_drop) + " \n")

        if self.plot_ft_importance:
            pu.plot_feature_importance(
                ft_df=tmp,
                mod_type=type(lin_mod).__name__,
                plot_title="Logistic l1 Feature Selection Coefs",
            )
        return

    def _avg_rank_drop(self, X, y):
        if self.problem_type == "clf":
            forest = RandomForestClassifier(
                n_estimators=200,
                class_weight=self.class_weight,
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
                verbose=0,
            )
        else:
            forest = RandomForestRegressor(
                n_estimators=200,
                random_state=self.random_seed,
                n_jobs=self.n_jobs,
                verbose=0,
            )

        # fit rf and get ranks
        forest.fit(X[self.sub_features], y)
        tmp_rf = {
            "Feature": self.sub_features,
            "RF Importance": forest.feature_importances_,
        }
        tmp_rf = pd.DataFrame(tmp_rf)
        tmp_rf.sort_values(["RF Importance"], ascending=False, inplace=True)
        tmp_rf["rf_rank"] = [i for i in range(tmp_rf.shape[0])]

        if self.problem_type == "clf":
            if len(set(y)) > 2:
                solver = "saga"
                multi_class = "multinomial"
            else:
                solver = "sag"
                multi_class = "auto"
            lin_mod = LogisticRegression(
                penalty="l2",
                class_weight=self.class_weight,
                solver=solver,
                multi_class=multi_class,
                random_state=self.random_seed,
            )
        else:
            lin_mod = Ridge(random_state=self.random_seed)

        # fit lr and get ranks
        lin_mod.fit(X[self.sub_features], y)
        if self.problem_type == "clf":
            tmp_lin = pd.DataFrame(
                {"Feature": self.sub_features, "Coef": abs(lin_mod.coef_[0])}
            )
        else:
            tmp_lin = pd.DataFrame(
                {"Feature": self.sub_features, "Coef": lin_mod.coef_}
            )
        tmp_lin.sort_values(["Coef"], ascending=True, inplace=True)
        tmp_lin["lr_rank"] = [i for i in range(tmp_lin.shape[0])]

        # create the rank df
        rank_df = tmp_rf.merge(
            tmp_lin, how="left", left_on="Feature", right_on="Feature"
        )
        rank_df["Average Rank"] = rank_df[["rf_rank", "lr_rank"]].mean(axis=1)
        rank_df.sort_values(["Average Rank"], ascending=True, inplace=True)
        num_drop = math.ceil(self.percent_rank_drop * rank_df.shape[0])
        num_keep = rank_df.shape[0] - num_drop
        rank_df["drop"] = np.append(np.repeat(1, num_drop), np.repeat(0, num_keep))
        self.rank_drop = list(rank_df[rank_df["drop"] == 1]["Feature"])

        if self.plot_ft_importance:
            rank_df.sort_values(["Average Rank"], ascending=False, inplace=True)
            display(rank_df[["Feature", "Average Rank", "RF Importance", "Abs Coef"]])

        return

    def transform(self, X):
        """
        Function to perform transform on given data (i.e. select features)
        :param X:
        :return: subsetted dataframe
        """

        return X[self.sub_features]

    def fit(self, X, y=None):
        """
        Function that takes X and y and performs the desired selection methods to reduce feature space
        :param X:
        :param y:
        :return:
        """
        if len(self.methods) == 0:
            logger.warning("NO SELECT FEATURES METHODS PASSED")
            return X

        # create copy to avoid changing the original data
        tmp_x = X.copy(deep=True)

        # get the initial features
        self.features = list(X.columns[:])
        self.sub_features = list(X.columns[:])

        if self.print_out:
            print("Init number of features: " + str(len(self.features)) + " \n")

        if "correlation" in self.methods:
            self._drop_correlated_features(tmp_x)

        if "rf_importance" in self.methods:
            self._drop_rf_low_importance(tmp_x, y)

        if "regress" in self.methods:
            self._regress_drop(tmp_x, y)

        if "avg_model_rank" in self.methods:
            self._avg_rank_drop(tmp_x, y)

        self.drop_fts = list(
            set(self.imp_drop + self.lin_drop + self.corr_drop + self.rank_drop)
        )

        self.sub_features = [
            col
            for col in self.sub_features
            if (col not in self.drop_fts) or (col in self.dont_drop)
        ]

        if self.print_out:
            print("Final number of fts : " + str(len(self.sub_features)) + "\n \n")
            print("Final features: " + str(self.sub_features) + "\n \n")
            print("Dropped features: " + str(self.drop_fts) + "\n \n")

        return self
