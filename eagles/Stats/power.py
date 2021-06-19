from eagles.Stats import utils as u
import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
import math
import warnings

# TODO add calc of sample size for compare means when have effect size wanted but don't know the est means of sds
#  can uses statsmodels power tt_ind_solve_power()


def sample_size_comparing_two_proportions(
    p1: float = None,
    p2: float = None,
    alpha: float = 0.05,
    power: float = 0.8,
    tails: str = "two",
) -> int:
    """
    Function to calculate sample size when comparing proportions between groups
    :param p1: float default None, proportion should be in range 0 - 1
    :param p2: float default None, proportion should be in range 0 - 1
    :param alpha: float default 0.05, desired alpha or confidence level
    :param power: float default .8, desired power for statistical test
    :param tails: string deafult 'two', Type of test i.e one or two tailed (expects 'one' or 'two')
    :return: int, number of samples needed per group
    """
    if p1 is None or p2 is None:
        warnings.warn("Missing p1 or p2")
        return None

    if p1 > 1 or p2 > 1 or p1 < 0 or p2 < 0:
        warnings.warn("p1 or p2 is outside range accepted. Should be between 0 - 1")
        return None

    if tails == "two":
        div = 2
    elif tails == "one":
        div = 1
    else:
        warnings.warn("Got unexpected tails got: " + tails + "expects one or two")
        return None

    Za = u.calc_guassian_critical_value(p=(1 - (alpha / div)))
    Zb = abs(u.calc_guassian_critical_value(p=(1 - power)))

    n = (((Za + Zb) ** 2) * (p1 * (1 - p1) + p2 * (1 - p2))) / ((p1 - p2) ** 2)

    return math.ceil(n)


def sample_size_comparing_two_means(
    group1_mean: float = None,
    group2_mean: float = None,
    sigma: float = None,
    alpha: float = 0.05,
    power: float = 0.8,
    tails: str = "two",
) -> int:
    """
    Function to calculate sample size when comparing two means
    :param group1_mean: float default None, expected mean of group 1
    :param group2_mean: float default None, expected mean of group 2
    :param sigma: float default None, expected population variance
    :param alpha: float default 0.05, desired alpha or confidence level
    :param power: float default .8, desired power for statistical test
    :param tails: string default "two", Type of test i.e one or two tailed (expects 'one' or 'two')
    :return: int, number of samples needed per group
    """

    if group1_mean is None or group2_mean is None or sigma is None:
        warnings.warn("Missing required group1_mean, group2_mean or sigma")
        return np.nan

    if tails == "two":
        div = 2
    elif tails == "one":
        div = 1
    else:
        warnings.warn("Got unexpected tails got: " + tails + "expects one or two")
        return None

    Za = u.calc_guassian_critical_value(p=(1 - (alpha / div)))
    Zb = abs(u.calc_guassian_critical_value(p=(1 - power)))

    n = (((Za + Zb) ** 2) * 2 * sigma) / ((group2_mean - group1_mean) ** 2)

    return math.ceil(n)


def power_achieved_two_proportions(
    effect_size: float = None,
    n1: int = None,
    n2: int = None,
    alpha: float = 0.05,
    tails: str = "two",
) -> float:
    """
    Function to calculate the power of z-test for two independent proportions
    :param effect_size: float default None, effect size of the test
    :param n1: int default None, number of observations in group 1
    :param n2: int default None, number of observations in group 2
    :param alpha: float default 0.05, significance level for the given test
    :param tails: string default "two", Indicator for one sided or two sided test,
    Options include 'two', 'one_less' or 'one_greater'
    :return: float power achieved
    """

    ratio = n2 / n1

    if tails == "two":
        nip = NormalIndPower()
        return nip.power(
            effect_size=effect_size,
            nobs1=n1,
            alpha=alpha,
            ratio=ratio,
            alternative="two-sided",
        )
    elif tails == "one_less":
        nip = NormalIndPower()
        return nip.power(
            effect_size=effect_size,
            nobs1=n1,
            alpha=alpha,
            ratio=ratio,
            alternative="smaller",
        )
    elif tails == "one_greater":
        nip = NormalIndPower()
        return nip.power(
            effect_size=effect_size,
            nobs1=n1,
            alpha=alpha,
            ratio=ratio,
            alternative="larger",
        )
    else:
        warnings.warn(
            "Tails indicator not supported expects two, one_less or one_greater got: "
            + tails
        )
        return np.nan


def power_achieved_chi2(
    alpha: float = 0.05, dof: float = None, effect_size: float = None, n_obs: int = None
) -> float:
    """
    Function to calculate the power achieved in chi2 test of independence
    :param alpha: float default 0.05, alpha value for test
    :param dof: float default None, degrees of freedom for the test
    :param effect_size: float default None, effect size achieved for given test
    :param n_obs: int default None, total number of observations for the test
    :return: float, power achieved
    """

    k = stats.chi2.ppf(1 - alpha, dof)
    nc = n_obs * (effect_size ** 2)
    return stats.ncx2.sf(k, dof, nc)


def power_achieved_ttest(
    critical_value: float = None,
    d: float = None,
    n1: int = None,
    n2: int = None,
    tails: "str" = "two",
    paired: bool = False,
) -> float:
    """
    Function to calculate the power achieved for a test
    :param critical_value: float default None, critical value needed to achieve sig result
    :param d: float default None, effect size of the test
    :param n1: int default None, sample size of group 1
    :param n2: in default None, sample size of group 2
    :param tails: str default "two", two tailed or one tailed test. If one expects 'one_less' or 'one_greater'
    :param paired: bool default False, boolean indicator for paired of independent samples ttest
    :return: float (power achieved)
    """

    # get total sample size
    n = n1 + n2

    # set sample based on paired or not
    if paired:
        sample = 1
        dof = (n - 1) * sample
        nc = d * np.sqrt(n / sample)
    else:
        # if paired and equal samples
        if n1 == n1:
            sample = 2
            dof = (n - 2) * sample
            nc = d * np.sqrt(n / sample)
        # calc denom factors for independent when unequal groups
        else:
            dof = n1 + n2 - 2
            nc = d * (1 / np.sqrt((1 / n1) + (1 / n2)))

    if tails == "two":
        power = stats.nct.sf(critical_value, dof, nc) + stats.nct.cdf(
            -critical_value, dof, nc
        )
        return power
    elif tails == "one_less":
        power = stats.nct.cdf(critical_value, dof, nc)
        return power
    elif tails == "one_greater":
        power = stats.nct.sf(critical_value, dof, nc)
        return power
    else:
        warnings.warn(
            "Number of tails is not supported. Expects two, one_less or one_greater. Got: "
            + tails
        )
        return np.nan
