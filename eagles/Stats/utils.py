from scipy.stats import norm, t
from scipy import stats
import pandas as pd
import numpy as np
from IPython.display import display
import warnings


def get_z_scores(data: np.array = None) -> np.array:
    """
    Function to calculate z scores for an array of data
    :param data: numpy array default None. Function to z score array of data
    :return: numpy array
    """

    z_scores = (data - data.mean()) / data.std()

    return z_scores


def calc_guassian_critical_value(p: float = 0.95) -> float:
    crit_value = norm.ppf(p)
    return crit_value


def get_p_value_zstat(z_stat: float = None, tails: "str" = "two") -> float:
    if tails == "two":
        return stats.norm.sf(np.abs(z_stat)) * 2
    elif tails == "one_less":
        return stats.norm.cdf(z_stat)
    elif tails == "one_greater":
        return stats.norm.sf(z_stat)
    else:
        warnings.warn(
            "Tails not supported expected two, one_less or one_greater got: " + tails
        )
        return np.nan


def calc_students_critical_value(
    alpha: float = 0.05, dof: int = None, tails: str = "two"
) -> float:
    div = 2 if tails == "two" else 1

    if tails in ["two", "one_greater"]:
        return t.ppf(1 - alpha / div, dof)
    else:
        return t.ppf(alpha / div, dof)


def get_p_value_tstat(t_stat: float = None, dof: float = None) -> float:
    """
    Function to calculate p value for ttest based on t statistic and degrees of freedom
    :param t_stat: float default None, t statistic from ttest
    :param dof: float default None, degrees of freedom for given ttest
    :return: float p value
    """
    return (1.0 - t.cdf(abs(t_stat), dof)) * 2.0


def get_agg_stats_means(
    data: pd.DataFrame = None, iv: str or list = None, dv: str = None, disp: bool = True
) -> pd.DataFrame:
    """
    Function to get aggregated stats when comparing continous DV to categorical DV (i.e. t test and ANOVA)
    :param data: pandas dataframe default None, expects pandas dataframe with columns related to ivs and dv
    :param iv: string or list default None, expects single string name of columns or
    list of string column names containing ivs
    :param dv: string default None, name of column containing dependent variable for analysis
    :param disp: boolean default True, boolean indicator to display results
    :return: pandas dataframe with aggregated stats
    """
    agg_stats = data.groupby(iv)[dv].agg(
        ["count", "mean", "median", "std", "var", "min", "max"]
    )
    agg_stats = agg_stats.round(4)
    if disp:
        display(
            agg_stats.style.set_caption("Group Level Stats").background_gradient(
                subset=["mean"], cmap="vlag"
            )
        )

    return agg_stats


def _highlight_max(s):
    """
    highlight the maximum in a Series yellow.
    """
    is_max = s == s.max()
    return ["background-color: yellow" if v else "" for v in is_max]


def get_agg_stats_proportions(
    data: pd.DataFrame = None, group: str = None, outcome: str = None, disp: bool = True
) -> pd.DataFrame:
    agg_stats = data.groupby(group)[outcome].agg(["count", "mean"])
    agg_stats.columns = ["Number of Observations", "Percent Positive"]

    agg_stats.loc[len(agg_stats.index)] = [data.shape[0], data[outcome].mean()]
    agg_stats.index = list(data[group].unique()) + ["total"]
    agg_stats["Percent Positive"] = agg_stats["Percent Positive"] * 100
    agg_stats = agg_stats.round(2)
    if disp:
        display(
            agg_stats.style.set_caption("Group Level Stats").apply(
                _highlight_max, subset=["Percent Positive"]
            )
        )
    return agg_stats
