from eagles.Supervised import model_init as mi
from eagles.Supervised.utils import (
    tuner_utils as tu,
    _print_utils,
    plot_utils as pu,
    logger_utils as lu,
    metric_utils as mu,
)

import time
import pandas as pd
import numpy as np
import scipy
from IPython.display import display

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

import warnings
import logging

logger = logging.getLogger(__name__)


# need to determine which paramters would be passed in when call eval method versus what would be passed in with class init


class SupervisedTuner:
    # should set defaults like tuner = 'random_cv'
    def __init__(
        self,
        tune_metric=None,
        eval_metrics=[],
        num_cv=5,
        bins=None,
        num_top_fts=None,
        tuner=None,
        n_iterations=15,
        get_ft_imp=True,
        n_jobs=1,
        random_seed=None,
        binary=True,
        disp=True,
        log=None,
        log_name=None,
        log_path=None,
        log_note=None,
    ) -> None:
        self.tune_metric = tune_metric
        self.eval_metrics = eval_metrics
        self.num_cv = num_cv
        self.bins = bins
        self.num_top_fts = num_top_fts
        self.tuner = tuner
        self.n_iterations = n_iterations
        self.get_ft_imp = get_ft_imp
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.binary = binary
        self.disp = disp
        self.log = log
        self.log_name = log_name
        self.log_path = log_path
        self.log_note = log_note
        self.mod = None
        self.params = None

        # init random seed
        if self.random_seed is None:
            self.random_seed = np.random.randint(1000, size=1)[0]
            print("Random Seed Value: " + str(random_seed))

        # TODO check if tuner is supported or not and then do this ********************
        if self.tuner is None:
            self.tune_test = False
        else:
            self.tune_test = True

        return

    def _check_data(self) -> None:
        # Data format
        # Check to see if pandas dataframe if not then convert to one
        if not isinstance(self.X, pd.DataFrame):
            if isinstance(self.X, scipy.sparse.csr.csr_matrix):
                self.X = self.X.todense()
            self.X = pd.DataFrame(self.X)
        if not isinstance(self.y, pd.Series):
            self.y = pd.Series(self.y)

        return

    # TODO look into implementing this: https://github.com/ray-project/tune-sklearn
    def _tune_model_params(self):
        # set up the parameter search object
        if self.tuner == "random_cv":
            scv = RandomizedSearchCV(
                self.mod,
                param_distributions=self.params,
                n_iter=self.n_iterations,
                scoring=self.tune_metric,
                cv=self.num_cv,
                refit=True,
                n_jobs=self.n_jobs,
                verbose=2,
                random_state=self.random_seed,
            )

        elif self.tuner == "bayes_cv":
            scv = BayesSearchCV(
                estimator=self.mod,
                search_spaces=self.params,
                scoring=self.tune_metric,
                n_iter=self.n_iterations,
                cv=self.num_cv,
                verbose=2,
                refit=True,
                n_jobs=self.n_jobs,
            )

        elif self.tuner == "grid_cv":
            scv = GridSearchCV(
                self.mod,
                param_grid=self.params,
                scoring=self.tune_metric,
                cv=self.num_cv,
                refit=True,
                n_jobs=self.n_jobs,
                verbose=1,
            )

        scv.fit(self.X, self.y)

        print(
            self.tune_metric
            + " score of best estimator during parameter tuning: "
            + str(scv.best_score_)
            + "\n"
        )

        return scv

    def model_eval(self):
        print("Performing CV Runs: " + str(self.num_cv))
        kf = KFold(n_splits=self.num_cv, shuffle=True, random_state=self.random_seed)

        if self.binary:
            avg = "binary"
        else:
            avg = "macro"

        metric_dictionary = mu.init_model_metrics(metrics=self.eval_metrics)

        # TODO need to implement for forward chain as well
        cnt = 1
        for train_index, test_index in kf.split(self.X):
            cv_st = time.time()

            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            self.mod.fit(X_train, y_train)
            preds = self.mod.predict(X_test)

            if self.problem_type == "clf":
                pred_probs = self.mod.predict_proba(X_test)[:, 1]
            else:
                pred_probs = []

            metric_dictionary = mu.calc_metrics(
                metrics=self.eval_metrics,
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
                + str(round(time.time() - cv_st, 4))
            )
            cnt += 1

        if self.disp:
            tmp_metric_dict = {
                k: metric_dictionary[k]
                for k in metric_dictionary.keys()
                if "_func" not in k
            }
            tmp_metric_df = pd.DataFrame(tmp_metric_dict)
            tmp_metric_df.loc["mean"] = tmp_metric_df.mean()
            tmp_metric_df.loc["std"] = tmp_metric_df.std()
            cv_cols = [i for i in range(1, self.num_cv + 1)] + ["mean", "std"]
            tmp_metric_df.insert(loc=0, column="cv run", value=cv_cols)
            tmp_metric_df.reset_index(drop=True, inplace=True)
            display(tmp_metric_df)

        print("Final cv train test split")
        for metric in self.eval_metrics:
            print(
                metric
                + " score: "
                + str(round(metric_dictionary[metric + "_scores"][-1], 4))
            )

        if self.problem_type == "clf":
            print(" \n")
            cf = confusion_matrix(y_test, preds)
            cr = classification_report(
                y_test, preds, target_names=[str(x) for x in self.mod.classes_]
            )

            if self.disp:
                pu.plot_confusion_matrix(cf=cf, labels=self.mod.classes_)
                print(cr)

        if self.binary and self.problem_type == "clf":
            prob_df = pd.DataFrame({"probab": pred_probs, "actual": y_test})
            bt, corr = tu.create_bin_table(
                df=prob_df, bins=self.bins, bin_col="probab", actual_col="actual"
            )
            if self.disp:
                display(bt)
                if pd.notnull(corr):
                    print(
                        "Correlation between probability bin order and percent actual: "
                        + str(round(corr, 3))
                    )

        if self.disp:
            if "roc_auc" in self.eval_metrics:
                pu.plot_roc_curve(y_true=y_test, pred_probs=pred_probs)
            if "precision_recall_auc" in self.eval_metrics:
                pu.plot_precision_recall_curve(y_true=y_test, pred_probs=pred_probs)

        if self.get_ft_imp:
            ft_imp_df = tu.feature_importances(
                mod=self.mod, X=self.X, num_top_fts=self.num_top_fts, disp=self.disp
            )
        return

    def eval(
        self,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        model=None,
        params: dict = None,
        pipe=None,
        scale=None,
        select_features=None,
    ):

        # check data format
        self.X = X
        self.y = y
        self._check_data()

        # init the model and define the problem type (linear and svr don't take random_state args)
        self.mod = mi.init_model(
            model=model,
            params=params,
            random_seed=self.random_seed,
            tune_test=self.tune_test,
        )

        self.problem_type = mi.define_problem_type(self.mod)
        if self.problem_type is None:
            logger.warning("Could not detect problem type exiting")
            return

        if pipe and (scale or select_features):
            warnings.warn(
                "ERROR CAN'T PASS IN PIPE OBJECT WITH SCALE AND/OR SELECT FEATURES"
            )
            return

        if pipe:
            self.mod, params = mi.build_pipes(mod=self.mod, params=params, pipe=pipe)
        elif scale or select_features:
            self.mod, params = mi.build_pipes(
                mod=self.mod,
                params=params,
                scale=scale,
                select_features=select_features,
                problem_type=self.problem_type,
            )

        # now that init the class can prob have user define these and then check in the init
        if self.tune_metric is None:
            if self.problem_type == "clf":
                self.tune_metric = "f1"
            else:
                self.tune_metric = "neg_mean_squared_error"

        # ensure that eval metrics have been defined
        if len(self.eval_metrics) == 0 and self.problem_type == "clf":
            self.eval_metrics = ["f1"]
        elif len(self.eval_metrics) == 0 and self.problem_type == "regress":
            self.eval_metrics = ["mse"]

        # if param tuning wanted then implement tune params and grab best estimator
        if self.tuner:
            scv = self._tune_model_params()
            self.mod = scv.best_estimator_
            self.params = self.mod.get_params()
            _ = _print_utils(mod=self.mod, X=self.X)

        # perform the model eval
        _ = self.model_eval()

        # generate logs
        if type(self.mod).__name__ == "Pipeline":
            if "feature_selection" in self.mod.named_steps:
                inds = [self.mod.named_steps["feature_selection"].get_support()][0]
                features = list(X.columns[inds])
            else:
                features = list(X.columns[:])
        else:
            features = list(X.columns[:])

        return
