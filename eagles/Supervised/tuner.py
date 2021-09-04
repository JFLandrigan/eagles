from eagles.Supervised import model_init as mi
from eagles.Supervised.utils import tuner_utils as tu
from eagles.Supervised.utils import plot_utils as pu
from eagles.Supervised.utils import logger_utils as lu
from eagles.Supervised.utils import metric_utils as mu

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
                param_distributions=params,
                n_iter=n_iterations,
                scoring=tune_metric,
                cv=num_cv,
                refit=True,
                n_jobs=n_jobs,
                verbose=2,
                random_state=random_seed,
            )

        elif self.tuner == "bayes_cv":
            scv = BayesSearchCV(
                estimator=self.mod,
                search_spaces=params,
                scoring=tune_metric,
                n_iter=n_iterations,
                cv=num_cv,
                verbose=2,
                refit=True,
                n_jobs=n_jobs,
            )

        elif self.tuner == "grid_cv":
            scv = GridSearchCV(
                self.mod,
                param_grid=params,
                scoring=tune_metric,
                cv=num_cv,
                refit=True,
                n_jobs=n_jobs,
                verbose=1,
            )

        scv.fit(X, y)

        print(
            "Mean cross val "
            + self.tune_metric
            + " score of best estimator during parameter tuning: "
            + str(scv.best_score_)
            + "\n"
        )

        return scv

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

        problem_type = mi.define_problem_type(self.mod)
        if problem_type is None:
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
                problem_type=problem_type,
            )

        # now that init the class can prob have user define these and then check in the init
        if self.tune_metric is None:
            if problem_type == "clf":
                self.tune_metric = "f1"
            else:
                self.tune_metric = "neg_mean_squared_error"

        # ensure that eval metrics have been defined
        if len(self.eval_metrics) == 0 and problem_type == "clf":
            self.eval_metrics = ["f1"]
        elif len(self.eval_metrics) == 0 and problem_type == "regress":
            self.eval_metrics = ["mse"]

        # if param tuning wanted then implement tune params and grab best estimator
        if self.tuner:
            scv = self._tune_model_params()
            self.mod = scv.best_estimator_
            self.params = self.mod.get_params()

        return
