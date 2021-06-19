import pandas as pd
import numpy as np
from math import sqrt
import warnings


def calc_cohens_d_one_sample(
    sample_mn: float = None, pop_mn: float = None, sample_std: float = None
) -> float:

    if not all([sample_mn, pop_mn, sample_std]):
        warnings.warn("Missing required values for cohens d one sample")
        return np.nan

    d = (sample_mn - pop_mn) / sample_std

    return d


def calc_cohens_d(
    obs_1: np.array = None,
    obs_2: np.array = None,
    paired: bool = False,
    hedges_g: bool = False,
) -> float:
    """
    Function to calculate cohens d for independt sample and paired t-test
    :param obs_1: numpy array default None, expects numpy array containing observations from group 1
    :param obs_2: numpy array default None, expects numpy array containing observations from group 2
    :param paired: bool default False, boolean indicator for whether paired or independent ttest
    :param hedges_g: bool default False, boolean indicator for whether or not to calculate hedges g
    :return: returns float d or g
    """
    mn1 = obs_1.mean()
    mn2 = obs_2.mean()
    s1 = obs_1.var(ddof=1)
    s2 = obs_2.var(ddof=1)
    n1 = len(obs_1)
    n2 = len(obs_2)

    if paired:
        if n1 != n2:
            warnings.warn("Different number of observations between group 1 and 2")
            return np.nan
        d = (mn1 - mn2) / sqrt((s1 + s2) / 2)
    else:
        dof = len(obs_1) + len(obs_2) - 2
        pooled_std = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / dof)
        d = (mn1 - mn2) / pooled_std

    if hedges_g:
        g = d * (1 - (3 / 4 * (n1 + n2) - 9))
        return g
    else:
        return d


def calc_cohens_d_from_t(
    t_stat: float, n1: int = None, n2: int = None, hedges_g: bool = False,
) -> float:
    """
    Function to calculate cohens d or hedges g from t stat
    :param t_stat: float default None: t statistic
    :param n1: int default None: number of obs in group 1
    :param n2: int default None: number of obs in group 2
    :param hedges_g: bool default False, boolean indicator for whether or not to calculate hedges g
    :return: returns float d or g
    """
    d = t_stat * sqrt((1 / n1) + (1 / n2))
    if hedges_g:
        g = d * (1 - (3 / 4 * (n1 + n2) - 9))
        return g
    else:
        return d


def calc_eta_squared(
    f_stat: float = None,
    dof_groups: int = None,
    dof_residual: int = None,
    ss_between: float = None,
    ss_total=None,
) -> float:
    """
    Function to calculate eta2 from F stat and dof
    :param f_stat: float default None, f statistic from ANOVA
    :param dof_groups: int default None, degrees of freedom from groups of IV
    :param dof_residual: int default None, degrees of freedom from error residuals
    :param ss_between: float default None, sum of square between groups
    :param ss_total: float default None, sum of squares total
    :return: float eta squared of anova
    """
    if f_stat and dof_groups and dof_residual:
        eta2 = (dof_groups * f_stat) / (dof_groups * f_stat + dof_residual)
    elif ss_between and ss_total:
        eta2 = ss_between / ss_total
    else:
        warnings.warn("Missing parameters for eta2 calculation")
        return np.nan

    return eta2


def calc_kruskal_eta_squared(
    h_stat: float = None, n: int = None, n_groups: int = None
) -> float:
    """
    Function to calculate eta2 based on kruskal wallis h statistic
    :param h_stat: float default None, h statistic from kruskal wallis test
    :param n: int default None, total number of observations
    :param n_groups: int default None, total number of groups
    :return: float effect size eta2
    """
    return (h_stat - n_groups + 1) / (n - n_groups)


def calc_wilcoxon_signed_rank_r(x: np.array = None, y: np.array = None) -> float:
    """
    Function to calculate effect size r for the wilcoxon signed rank test. z calculated using
     https://www.statisticshowto.com/wilcoxon-signed-rank-test/ r = z/sqrt(n)
    :param x: np.array default None, Array containing 1st measurements
    :param y: np.array default None, Array containing 2nd measurements
    :return: float effect size r
    """

    if len(x) != len(y):
        warnings.warn("Length of arrays should be equal. Returning None")
        return np.nan

    def count_ties(df):
        t = df["abs_diffs"].value_counts()
        return sum(t[t > 1])

    n = len(x)
    df = pd.DataFrame({"x": x, "y": y})
    df["diffs"] = df["y"] - df["x"]
    df["abs_diffs"] = abs(df["diffs"])
    num_ties = count_ties(df)
    df.sort_values(by="abs_diffs", ascending=True, inplace=True)
    df["ranks"] = [i for i in range(1, len(df) + 1)]
    neg_sum_ranks = sum(df[df["diffs"] < 0]["ranks"])
    pos_sum_ranks = sum(df[df["diffs"] > 0]["ranks"])
    stat = neg_sum_ranks if neg_sum_ranks < pos_sum_ranks else pos_sum_ranks
    reducer = (num_ties ** 3 - num_ties) / 48
    numerator = stat - ((n * (n + 1)) / 4)
    denominator = np.sqrt(((n * (n + 1)) * (2 * n + 1) / 24) - reducer)

    z = abs(numerator / denominator)
    r = z / np.sqrt(len(x))

    return r


def calc_man_whitney_u_r(u_stat: float = None, n1: int = None, n2: int = None) -> float:
    """
    Function to calculate effect size r for the man whitney u test
    :param u_stat: float default None, expects u test statistic
    :param n1: int default None, expects sample size of group 1
    :param n2: int default None, expects sample size of group 2
    :return: float r effect size
    """
    return 1 - ((2 * u_stat) / (n1 * n2))


def calc_chi2_goodness_fit_w(obs: np.array = None) -> float:
    """
    Function to calculate the cohens W effect size fir ch12 goodness of fit test
    :param obs: numpy array default None, expects numpy array containing observed counts
    :return: float cohens W
    """

    if obs is None:
        warnings.warn("Missing required obs arg for cohens W effect size calculation")
        return np.nan

    obs = obs / sum(obs)
    num_cats = len(obs)
    exp = np.repeat(round(1 / num_cats, 2), num_cats)
    w = np.sqrt(sum((obs - exp) ** 2 / exp))

    return w


def calc_chi2_independence_es(
    effect_type: str = "phi",
    chi2_stat: float = None,
    n_obs: int = None,
    dof: float = None,
    contingency: pd.DataFrame = None,
) -> float:
    """
    Function to calculate effect size of chi2 test (source: https://www.statology.org/effect-size-chi-square/)
    :param effect_type: string default 'phi', string indicating which effect size to compute.
    Options include phi, cramers, odds_ratio
    :param chi2_stat: float default None, chi2 test statistic
    :param n_obs: int default None, total number of observations
    :param dof: float default None, degrees of freedom for given test
    :param contingency: pd.DataFrame default None, if odds_ratio then expects contingency table
    :return: float effect size
    """

    if effect_type in ["phi", "cramers"] and (chi2_stat is None or n_obs is None):
        warnings.warn(
            "Missing required args for phi or cramers effect size calculation (i.e. chi2_stat or n_obs)"
        )
        return np.nan

    if effect_type == "phi":
        return np.sqrt((chi2_stat / n_obs))

    elif effect_type == "cramers":
        if dof is None:
            warnings.warn("Missing dof for cramers effect size")
            return np.nan
        else:
            return np.sqrt(chi2_stat / (n_obs * dof))

    elif effect_type == "odds_ratio":
        if contingency is None:
            warnings.warn(
                "odds_ratio effect size calculation expects contingency table"
            )
            return np.nan

        if contingency.shape[0] == 2 and contingency.shape[1] == 2:
            return (contingency.iloc[0, 0] / contingency.iloc[1, 1]) / (
                contingency.iloc[0, 1] / contingency.iloc[1, 0]
            )
        else:
            warnings.warn(
                "odds_ration expects 2x2 contingency table got shape: "
                + str(contingency.shape)
            )
            return np.nan
    else:
        warnings.warn(
            "Effect size type not supported expected phi, cramers or odds_ratio got: "
            + effect_type
        )
        return np.nan


def z_test(z_stat: float = None, n1: int = None, n2: int = None) -> float:
    """
    Function to calculate the effect size for z test of two proportions
    :param z_stat: float default None, z statistic obtained by test
    :param n1: int default None, number of observations in group 1
    :param n2: int default None, number of observations in group 2
    :return: float effect size
    """
    return z_stat / np.sqrt(n1 + n2)
