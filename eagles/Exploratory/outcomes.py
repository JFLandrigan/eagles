from eagles.Exploratory.utils import plot_utils as pu

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
import logging
from IPython.display import display

logger = logging.getLogger(__name__)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def _perform_regress(
    data: pd.DataFrame = None,
    outcome_type: str = None,
    outcome: str = None,
    fts: list = None,
    disp: bool = True,
) -> pd.DataFrame:

    # todo Should also add in regression plots to show the fits
    # set model based on binary or not
    tmp_data = data.copy(deep=True)
    tmp_data.dropna(subset=[outcome] + fts, axis=0, inplace=True)

    if outcome_type == "categorical":
        tmp_data[outcome] = tmp_data[outcome].astype(int)

        est = sm.Logit(tmp_data[outcome], tmp_data[fts])
        est2 = est.fit(method="bfgs")
    else:
        est = sm.OLS(tmp_data[outcome], tmp_data[fts])
        est2 = est.fit()

    results_summary = est2.summary()
    res = results_summary.tables[1]
    res = pd.read_html(res.as_html(), header=0, index_col=0)[0].reset_index()
    res.rename(columns={"index": "feature"}, inplace=True)

    if disp:
        display(res)

    return res


def _get_proportions_by_outcomes(
    data: pd.DataFrame = None,
    outcome: str = None,
    categorical_fts: list = [],
    disp: bool = True,
    plot: bool = False,
) -> pd.DataFrame:

    cat_df = pd.DataFrame()
    for ft in categorical_fts:

        grp_df = data.groupby(ft, as_index=False)[outcome].agg("count")
        grp_df["proportion_samples"] = round((grp_df[outcome] / len(data)) * 100, 2)
        grp_df[ft] = list(map(lambda x: ft + "_" + str(x), grp_df[ft]))
        grp_df.columns = ["feature", "count", "proportion_samples"]
        cat_df = pd.DataFrame([cat_df, grp_df])

    if disp:
        display(cat_df)

    if plot:
        pu.plot_proportions_by_outcome(data=data, outcome=outcome, fts=categorical_fts)

    return cat_df


def _get_corr_to_outcome(
    data: pd.DataFrame = None,
    outcome: str = None,
    continuous_fts: list = [],
    disp: bool = True,
    plot: bool = False,
) -> pd.DataFrame:

    corrs = np.array([])

    for ft in continuous_fts:
        corrs = np.append(corrs, data[[outcome, ft]].corr().iloc[0, 1])

    corr_df = pd.DataFrame(
        {
            "outcome": np.repeat(outcome, len(continuous_fts)),
            "feature": continuous_fts,
            "correlation": corrs,
        }
    )

    if disp:
        display(corr_df)

    return corr_df


def _get_descriptives_by_outcome(
    data: pd.DataFrame = None,
    outcome: list = [],
    continuous_fts: list = [],
    descriptive_stats: list = [],
    disp: bool = True,
    plot: bool = False,
) -> pd.DataFrame:

    if len(descriptive_stats) == 0:
        descriptive_stats = ["mean", "median", "std", "min", "max"]

    grp_df = data.groupby(outcome, as_index=False)[continuous_fts].agg(
        descriptive_stats
    )

    col_names = [stat + "_" + ft for ft in continuous_fts for stat in descriptive_stats]
    col_names = outcome + col_names
    grp_df.reset_index(inplace=True)
    grp_df.columns = col_names

    if disp:
        display(grp_df)

    if plot:
        pu.plot_outcome_boxes(data=data, outcome=outcome, fts=continuous_fts)

    return grp_df


def stats_by_outcome(
    data: pd.DataFrame = None,
    outcome_type: str = "categorical",
    outcome: str = None,
    categorical_fts: list = [],
    continuous_fts: list = [],
    analyses: list = [],
    descriptive_stats: list = [],
    scale: str = None,
    disp: bool = True,
    # remove_outliers: bool = False,
    plot: bool = False,
) -> dict:
    """

    :param data: expects pandas dataframe
    :param outcome_type: str type of outcome (i.e. categorical or continuous)
    :param outcome: str name of column containing the outcome
    :param continuous_fts: list of strings corresponding to column names of continuous features
    :param categorical_fts: list of strings corresponding to column names of categorical features
    :param analyses: list of desired analyses, options include descriptives (i.e. returns mean, median, max, min, std),
    proportions (i.e. proportion comparisons for categorical features), regress (i.e. returns significant
    predictors of the outcomes by performing regression using the features as predictors)
    :param scale: string indicating whether or not to scale the data. Expects either "standard" or "minmax"
    :param disp: default True, boolean indicator to display result dataframes
    :param plot: Boolean, default False. IF true plots are displayed
    :return: dictionary containing analyse keyed and result dataframe value pairs
    """
    if outcome is None:
        logger.warning("No outcome passed in")
        return None

    analyses_dict = {}
    fts = categorical_fts + continuous_fts

    # if scale then scale the continuous features
    if scale:
        if scale == "standard":
            scaler = StandardScaler()
        elif scale == "minmax":
            scaler = MinMaxScaler()
        else:
            logger.warning("Scaler not supported")
            return None

        data[continuous_fts] = scaler.fit_transform(data[continuous_fts])

    if len(analyses) == 0:
        analyses = ["descriptives", "proportions", "regress"]

    for a in analyses:
        # get the base descriptives for the continuous features by outcome
        if a == "descriptives":
            if outcome_type == "categorical":
                desc_df = _get_descriptives_by_outcome(
                    data=data,
                    outcome=[outcome],
                    continuous_fts=continuous_fts,
                    descriptive_stats=descriptive_stats,
                    disp=disp,
                    plot=plot,
                )
                analyses_dict["descriptives"] = desc_df
            else:
                corr_df = _get_corr_to_outcome(
                    data=data,
                    outcome=outcome,
                    continuous_fts=continuous_fts,
                    disp=disp,
                    plot=plot,
                )
                analyses_dict["correlations"] = corr_df
                desc_df = _get_descriptives_by_outcome(
                    data=data,
                    outcome=categorical_fts,
                    continuous_fts=[outcome],
                    descriptive_stats=descriptive_stats,
                    disp=disp,
                    plot=plot,
                )
                analyses_dict["desc_df"] = desc_df

        # get the proportion labels for categorical features by outcome
        elif a == "proportions" and len(categorical_fts) > 0:
            prop_df = _get_proportions_by_outcomes(
                data=data,
                outcome=outcome,
                categorical_fts=categorical_fts,
                disp=disp,
                plot=plot,
            )
            analyses_dict["proportions"] = prop_df

        elif a == "regress":
            sig_df = _perform_regress(
                data=data,
                outcome=outcome,
                fts=fts,
                outcome_type=outcome_type,
                disp=disp,
            )
            analyses_dict["regress"] = sig_df
        else:
            logger.warning(a + " not supported")

    return analyses_dict
