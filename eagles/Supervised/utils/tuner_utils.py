from eagles.Supervised.utils import plot_utils as pu
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import train_test_split


def feature_comps(data=None, outcome=None, problem_type="clf"):
    return


def select_features(
    X=None,
    y=None,
    methods=[],
    problem_type="clf",
    model_pipe=None,
    imp_thresh=0.005,
    corr_thresh=0.7,
    bin_fts=None,
    dont_drop=None,
    random_seed=None,
    n_jobs=None,
    plot_ft_importance=False,
    plot_ft_corr=False,
):
    """
    Function to reduce feature set size
    Expects:
        X - pandas df containing the feature columns
        y - pandas series containg the outcomes
        imp_thresh - min importance threshold for the rf importance (below thresh cut)
        corr_thresh - correlation threshold where fts above thresh are cut
        nonbin_fts - list of col names with non binarized features
        display_imp - coolean if true then displays the feature importances of top 20
        dont_drop - list of col names don't want to drop regradless of corr or importance
    Returns: list of included features, list of dropped features
    """

    if len(methods) == 0:
        print("NO SELECT FEATURES METHODS PASSED")
        return

    # get the initial features
    ft_cols = X.columns[:]
    print("Init number of features: " + str(len(ft_cols)) + " \n")

    imp_drop = []
    lin_drop = []
    corr_drop = []
    if dont_drop is None:
        dont_drop = []

    if "correlation" in methods:
        # correlation drop
        corr_fts = [x for x in X.columns if x not in bin_fts]
        correlations = X[corr_fts].corr()
        upper = correlations.where(
            np.triu(np.ones(correlations.shape), k=1).astype(np.bool)
        )
        corr_drop = [
            column for column in upper.columns if any(upper[column].abs() > corr_thresh)
        ]

        if plot_ft_corr:
            pu.plot_feature_correlations(
                df=X[corr_fts].copy(deep=True),
                plot_title="Feature Correlation Pre-Drop",
            )

        # drop the correlation features first then fit the models
        print("Features dropping due to high correlation: " + str(corr_drop) + " \n")
        ft_cols = [x for x in ft_cols if (x not in corr_drop) or (x in dont_drop)]
        X = X[ft_cols].copy(deep=True)

    # Model importance
    X_train, X_test, y_train, y_test = train_test_split(
        X[ft_cols], y, test_size=0.2, random_state=random_seed
    )

    if "rf_importance" in methods:

        if problem_type == "clf":
            forest = RandomForestClassifier(
                n_estimators=200, random_state=random_seed, n_jobs=n_jobs
            )
        else:
            forest = RandomForestRegressor(
                n_estimators=200, random_state=random_seed, n_jobs=n_jobs
            )

        if model_pipe:
            tmp_pipe = clone(model_pipe)
            tmp_pipe.steps.append(["mod", forest])
            forest = clone(tmp_pipe)

        forest.fit(X_train, y_train)

        if model_pipe:
            forest = forest.named_steps["mod"]

        rf_importances = forest.feature_importances_

        ftImp = {"Feature": ft_cols, "Importance": rf_importances}
        ftImp_df = pd.DataFrame(ftImp)
        ftImp_df.sort_values(["Importance"], ascending=False, inplace=True)
        imp_drop = list(ftImp_df[ftImp_df["Importance"] < imp_thresh]["Feature"])
        print("Features dropping from low importance: " + str(imp_drop) + " \n")

        if plot_ft_importance:
            pu.plot_feature_importance(
                ft_df=ftImp_df,
                mod_type=type(forest).__name__,
                plot_title="RF Feature Selection Importance",
            )

    if "regress" in methods:

        if problem_type == "clf":
            lin_mod = LogisticRegression(
                penalty="l1", solver="liblinear", random_state=random_seed
            )
        else:
            lin_mod = Lasso(random_state=random_seed)

        if model_pipe:
            tmp_pipe = clone(model_pipe)
            tmp_pipe.steps.append(["mod", lin_mod])
            lin_mod = clone(tmp_pipe)

        lin_mod.fit(X_train, y_train)

        if model_pipe:
            lin_mod = lin_mod.named_steps["mod"]

        if problem_type == "clf":
            tmp = pd.DataFrame({"Feature": ft_cols, "Coef": lin_mod.coef_[0]})
        else:
            tmp = pd.DataFrame({"Feature": ft_cols, "Coef": lin_mod.coef_})

        lin_drop = list(tmp["Feature"][tmp["Coef"] == 0])
        print("Features dropping from l1 regression: " + str(lin_drop) + " \n")

        if plot_ft_importance:
            pu.plot_feature_importance(
                ft_df=tmp,
                mod_type=type(lin_mod).__name__,
                plot_title="Logistic l1 Feature Selection Coefs",
            )

    # get the final drop and feature sets
    drop_fts = list(set(imp_drop + lin_drop + corr_drop))

    sub_fts = [col for col in ft_cols if (col not in drop_fts) or (col in dont_drop)]

    print("Final number of fts : " + str(len(sub_fts)) + "\n \n")
    print("Final features: " + str(sub_fts) + "\n \n")
    print("Dropped features: " + str(drop_fts) + "\n \n")

    return sub_fts, drop_fts


def create_bin_table(df=None, bins=None, bin_col=None, actual_col=None):
    """
    Function to generate the bin tables with percents
    Expects: df - pandas df from reco weighting containing rectaken_01 and col to be binned
            bins - default to prob taken bins unless passed list of bin steps i.e. [x/100 for x in range(-5,105,5)]
            bin_col - name of the col to be binned
            save_dir - directory to save the pandas dataframe out to
    Returns: Saves the generated dataframe out to csv
    """
    # Generate the bin col name
    bin_col_name = bin_col + "_bin"

    # Generate the list of bins (go by 5%)
    # default to prob taken include -5 so that anything at 0 will have bin and go above 100 so
    # that include values in bins from 95 to 100
    if bins is None:
        bin_list = [x / 100 for x in range(-5, 105, 5)]
    else:
        bin_list = bins

    # create the bins
    df[bin_col_name] = pd.cut(df[bin_col], bin_list)

    # get the counts for the number of obs in each bin and the percent taken in each bin
    cnts = df[bin_col_name].value_counts().reset_index()
    cnts.columns = [bin_col_name, "count"]

    # Get the percent ivr per bin
    percs = df.groupby(by=bin_col_name)[actual_col].mean().reset_index()
    percs.columns = [bin_col_name, "percent_actual"]

    # combine the counts and the percents, sort the table by bin and write the table out
    wrt_table = cnts.merge(
        percs, left_on=bin_col_name, right_on=bin_col_name, how="inner"
    )
    wrt_table.sort_values(by=bin_col_name, inplace=True)

    return wrt_table


def get_feature_importances(mod_type=None, mod=None, features=None):

    if (
        ("RandomForest" in mod_type)
        or ("GradientBoosting" in mod_type)
        or ("DecisionTree" in mod_type)
        or ("ExtraTrees" in mod_type)
    ):
        importance_values = mod.feature_importances_

        ftImp = {"Feature": features, "Importance": importance_values}
        ftImp_df = pd.DataFrame(ftImp)

        # display_imp is true then plot the importance values of the features
        ftImp_df = ftImp_df.sort_values(["Importance"], ascending=False).reset_index(
            drop=True
        )

        return ftImp_df

    elif (
        ("Regression" in mod_type)
        or (mod_type == "Lasso")
        or (mod_type == "ElasticNet")
    ):
        if mod_type == "LogisticRegression":
            tmp = pd.DataFrame({"Feature": features, "Coef": mod.coef_[0]})
        else:
            tmp = pd.DataFrame({"Feature": features, "Coef": mod.coef_})

        tmp["Abs_Coef"] = tmp["Coef"].abs()
        tmp = tmp.sort_values(["Abs_Coef"], ascending=False).reset_index(drop=True)
        tmp = tmp[["Feature", "Coef"]].copy(deep=True)

        return tmp

    return


def feature_importances(mod=None, X=None, num_top_fts=None):

    if type(mod).__name__ == "Pipeline":

        if "feature_selection" in mod.named_steps:
            inds = [mod.named_steps["feature_selection"].get_support()][0]
            tmp_fts = X.columns[inds]
        else:
            tmp_fts = list(X.columns)

        tmp_mod = mod.named_steps["clf"]
        ft_imp_df = get_feature_importances(
            mod_type=type(mod.named_steps["clf"]).__name__,
            mod=tmp_mod,
            features=tmp_fts,
        )
        pu.plot_feature_importance(
            ft_df=ft_imp_df,
            mod_type=type(mod.named_steps["clf"]).__name__,
            num_top_fts=num_top_fts,
            plot_title=type(mod.named_steps["clf"]).__name__ + " Model Importance",
        )

    elif type(mod).__name__ == "VotingClassifier":

        ft_imp_df = pd.DataFrame()

        for c in mod.estimators_:
            if type(c).__name__ == "Pipeline":

                if "feature_selection" in c.named_steps:
                    inds = [mod.named_steps["feature_selection"].get_support()][0]
                    tmp_fts = X.columns[inds]
                else:
                    tmp_fts = list(X.columns)

                tmp_mod = c.named_steps["clf"]
                tmp_ft_imp_df = get_feature_importances(
                    mod_type=type(c.named_steps["clf"]).__name__,
                    mod=tmp_mod,
                    features=tmp_fts,
                )
                pu.plot_feature_importance(
                    ft_df=tmp_ft_imp_df,
                    mod_type=type(c.named_steps["clf"]).__name__,
                    num_top_fts=num_top_fts,
                    plot_title=type(c.named_steps["clf"]).__name__
                    + " Model Importance",
                )
            else:
                tmp_ft_imp_df = get_feature_importances(
                    mod_type=type(c).__name__, mod=c, features=list(X.columns)
                )
                pu.plot_feature_importance(
                    ft_df=tmp_ft_imp_df,
                    mod_type=type(c).__name__,
                    num_top_fts=num_top_fts,
                    plot_title=type(c).__name__ + " Model Importance",
                )

            tmp_ft_imp_df.columns = ["features", "value"]
            tmp_ft_imp_df["features"] = (
                type(c.named_steps["clf"]).__name__ + "_" + tmp_ft_imp_df["features"]
            )

            ft_imp_df = pd.concat([ft_imp_df, tmp_ft_imp_df])

    else:
        ft_imp_df = get_feature_importances(
            mod_type=type(mod).__name__, mod=mod, features=list(X.columns)
        )
        pu.plot_feature_importance(
            ft_df=ft_imp_df,
            mod_type=type(mod).__name__,
            num_top_fts=num_top_fts,
            plot_title=type(mod).__name__ + " Model Importance",
        )

    return ft_imp_df
