from eagles.Stats import (
    utils as u,
    effect_size as es,
    assumptions as assump,
    plot_utils as pu,
)
from eagles.Stats import config
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM, anova_lm
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp, describe
from statsmodels.stats.power import tt_solve_power
from statsmodels.stats import api as sms
from IPython.display import display
from io import StringIO
import warnings


# TODO add in optional label args to use in the res stat labels and the plot lables

# TODO double check how calculating the pvalues for onsided versus two sided test


def one_sample_t_test(
    data: np.array = None,
    pop_mn: float = None,
    tails: str = "two-sided",
    normality_test: str = "shapiro",
    disp: bool = True,
) -> pd.DataFrame:
    """
    Function to run a one sample t test
    :param data: numpy array default None, expects 1d numpy array
    :param pop_mn: float default None, The given population mean comparing to
    :param tails: str default 'two-sided', takes two or one indicator for if two sided or one sided.
    If one tailed test expects 'less' or 'greater'
    :param normality_test: string default 'shapiro'. Name for test of normality expects either
    shapiro, dagostino, ks or None. If None then runs all tests
    :param disp: boolean default True. Boolean indicator to display results or not
    :return: pandas dataframe
    """

    if data is None or pop_mn is None:
        warnings.warn("Missing required arguments for one sample t test calculation")
        return None

    # Test for normality
    _ = assump.test_normality(test=normality_test, x=data, disp=disp)

    descriptives = pd.DataFrame(describe(data)).T
    stats = ["n", "min max", "mean", "variance", "skewness", "kurtosis"]
    descriptives.columns = stats

    dof = len(data) - 1
    t_stat, p_val = ttest_1samp(a=data, popmean=pop_mn, nan_policy="omit")
    ef_sz = es.calc_cohens_d_one_sample(
        sample_mn=data.mean(), pop_mn=pop_mn, sample_std=data.std(ddof=1)
    )

    if tails == "less":
        if t_stat < 0:
            p_val = p_val / 2
        else:
            p_val = 1 - p_val / 2
    elif tails == "greater":
        if t_stat > 0:
            p_val = p_val / 2
        else:
            p_val = 1 - p_val / 2

    res = pd.DataFrame(
        {
            "Statistic": ["t stat", "dof", "p value", "cohens d"],
            "Value": [t_stat, dof, p_val, ef_sz],
        }
    )

    if disp:
        display(descriptives.style.set_caption("Descriptives"))
        display(res.style.set_caption("Test Results"))

    return res


def t_test(
    data: pd.DataFrame = None,
    iv: str = None,
    dv: str = None,
    paired: bool = False,
    alpha: float = 0.05,
    tails: str = "two",
    welch: bool = False,
    hedges_g: bool = False,
    variance_test: str = "levene",
    normality_test: str = "shapiro",
    disp: bool = True,
) -> pd.DataFrame:
    """
    Function to perform either a paired or independent samples t test
    :param data: pandas dataframe default None, expects pandas dataframe with columns related to ivs and dv
    :param iv: string default None, expects single string name of column relating to measurements
    :param dv: string default None, name of column containing dependent variable for analysis
    :param paired: bool default False, boolean indicator to run paired or independent samples t test
    :param alpha: float default .05, alpha level for test
    :param tails: str default 'two-sided', takes two or one indicator for if two sided or one sided.
    If one tailed test expects 'less' or 'greater'
    :param welch: bool default False, indicator to perform welches t test for unequal varince
    :param hedges_g: bool default False, indicator to calculate hedges g instead of cohens d
    :param variance_test: string default levene. Which test to perform for equal variances.
    Expects either levene or bartlett
    :param normality_test: string default 'shapiro'. Name for test of normality expects either
    shapiro, dagostino, ks or None. If None then runs all tests
    :param disp: bool default True, indicator to display results or not
    :return:
    """

    # get the group names being compared
    groups = list(data[iv].unique())

    # base level stats
    s1 = data[data[iv] == groups[0]].var(ddof=1)[0]
    n1 = len(data[data[iv] == groups[0]])
    n2 = len(data[data[iv] == groups[1]])

    # Run test of normality and plot distributions
    for x in groups:
        if disp:
            print("Nomrality test for: " + str(x))
        _ = assump.test_normality(
            test=normality_test, x=data[data[iv] == x][dv], disp=disp
        )

    if disp:
        pu.plot_distributions_by_group(data=data, obs_var=dv, group_var=iv)

    # Run test of equal variance and if sig and not welch then throw warning
    var_stat, var_pval = assump.test_equal_variances(
        *[list(data[data[iv] == x][dv]) for x in groups], test=variance_test, disp=disp
    )
    if var_pval < 0.05 and not welch:
        warnings.warn("test of unequal variances was significant and not using welch")

    # Do quick check for sample sizes when running a paired t-test
    if paired:
        if n1 != n2:
            warnings.warn("Number of observations does not match between obs")
            return None
        else:
            dof = n1 - 1
            # standard dev of differences  div by sqrt of sample size
            t_stat, p_val = ttest_rel(*[list(data[data[iv] == x][dv]) for x in groups])
    else:
        if not welch:
            dof = n1 + n2 - 2
            t_stat, p_val = ttest_ind(
                *[list(data[data[iv] == x][dv]) for x in groups], equal_var=True
            )
        else:
            dof = ((s1 / n1) + (s1 / n2)) ** 2 / (
                (s1 / n1) ** 2 / (n1 - 1) + (s1 / n2) ** 2 / (n2 - 1)
            )
            t_stat, p_val = ttest_ind(
                *[list(data[data[iv] == x][dv]) for x in groups], equal_var=False
            )

    if tails == "less":
        if t_stat < 0:
            p_val = p_val / 2
        else:
            p_val = 1 - p_val / 2
    elif tails == "greater":
        if t_stat > 0:
            p_val = p_val / 2
        else:
            p_val = 1 - p_val / 2

    ef_sz = es.calc_cohens_d_from_t(t_stat=t_stat, n1=n1, n2=n2, hedges_g=hedges_g)
    if tails == "two-sided":
        alternative = "two-sided"
    elif tails == "less":
        alternative = "smaller"
    else:
        alternative = "larger"
    power = tt_solve_power(
        effect_size=ef_sz, nobs=n1 + n2, alpha=alpha, alternative=alternative
    )

    # calc the 95% CI bounds
    cm = sms.CompareMeans(
        sms.DescrStatsW(data[data[iv] == groups[0]][dv]),
        sms.DescrStatsW(data[data[iv] == groups[1]][dv]),
    )
    if welch:
        lb, ub = cm.tconfint_diff(
            alpha=alpha, alternative=alternative, usevar="unequal"
        )
    else:
        lb, ub = cm.tconfint_diff(alpha=alpha, alternative=alternative, usevar="pooled")
    # Put results into a df (put cols in a config file to reduce the code lines
    stat_values = [
        round(t_stat, 4),
        dof,
        p_val,
        [round(lb, 4), round(ub, 4)],
        round(ef_sz, 4),
        power,
    ]
    res = pd.DataFrame({"Statistic": config.t_test_cols, "Value": stat_values,})
    agg_stats = u.get_agg_stats_means(data=data, iv=iv, dv=dv, disp=disp)
    if disp:
        display(res.style.set_caption("Test Results").hide_index())
        pu.group_box_plot(data=data, iv=iv, dv=dv)

    return res, agg_stats


def _run_aov_multi_comps(
    data: pd.DataFrame = None,
    iv: str or list = None,
    dv: str = None,
    alpha: float = 0.05,
    disp: bool = True,
):
    """
    Function to run tukey hsd pairwise comps for anova testing
    :param data: pandas dataframe default None, expects pandas dataframe containing dv and iv cols
    :param iv: string or list of strings default none, expects a single string or list of strings
    containing the grouping ivs
    :param dv: string default None, expects string name of dv col in data
    :param alpha: float default .05, expects float for p val significance level
    :param disp: boolean default true, boolean indicator to display result table or not
    :return: pandas df containing the results of pairwise comps
    """

    if type(iv) == list and len(iv) == 2:
        data = data.copy(deep=True)
        data["paired_groups"] = data[iv[0]] + " " + data[iv[1]]
        iv = "paired_groups"

    multicomp = sms.multicomp.MultiComparison(data=data[dv], groups=data[iv])
    res = multicomp.tukeyhsd(alpha=alpha)
    res = pd.read_csv(StringIO(res.summary().as_csv()), skiprows=1)
    res.columns = [col.strip() for col in res.columns]

    if disp:
        display(
            res.style.set_caption("Pairwise Comps")
            .apply(
                [lambda x: "color: green" if x < 0.05 else "color: red"],
                subset=["p-adj"],
            )
            .background_gradient(subset=["meandiff"], cmap="vlag")
        )
    return res


def _fit_anova(model):
    # Get intermittent bug of numpy LinAlgError: SVD did not converge
    # so wrapped in try while true to get around issue
    # see this discussion
    # https://stackoverflow.com/questions/63761366/numpy-linalg-linalgerror-svd-did-not-converge-in-linear-least-squares-on-first
    try_cnt = 1
    while True:
        try:
            model = model.fit()
            break
        except:
            try_cnt += 1
            if try_cnt <= 20:
                continue
            else:
                warnings.warn("Could not fit ANOVA RM due to convergence issues")
                return
    return model


def anova(
    data: pd.DataFrame = None,
    iv: str or list = None,
    dv: str = None,
    obs_id: str = None,
    anova_type: str = "one_way",
    normality_test: str = "shapiro",
    variance_test: str = "levene",
    disp: bool = True,
) -> list:
    """
    Function to perform one-way, two-way or repeated measure ANOVAs
    :param data: pandas dataframe default None, expects pandas dataframe with columns related to ivs and dv
    :param iv: string or list default None, expects single string name of columns or
    list of string column names containing ivs
    :param dv: string default None, name of column containing dependent variable for analysis
    :param obs_id: list default None. IDs of the observations for repeated measures anova ignored otherwise
    :param anova_type: Expects either 'one_way', 'two_way', or 'rm' (repeated measures)
    :param normality_test: string default 'shapiro'. Name for test of normality expects either
    shapiro, dagostino, ks or None. If None then runs all tests
    :param disp: boolean default True, boolean indicator to display results
    :param plot_interactions: boolean default True, if two-way indicator of whether or not to plot interactions
    :return: list with anova table type pandas dataframe, aggregated stats type pandas dataframe
    """

    # Run test of normality and plot distributions
    if type(iv) == str:
        groups = [iv]
    else:
        groups = iv.copy()
    for x in groups:
        if disp:
            print("Nomrality test for group: " + str(x))
        _ = assump.test_normality(
            test=normality_test, x=data[data[iv] == x][dv], disp=disp
        )

    if disp:
        if type(iv) == str:
            pu.plot_distributions_by_group(data=data, obs_var=dv, group_var=iv)
        elif type(iv) == list:
            if len(iv) == 1:
                pu.plot_distributions_by_group(data=data, obs_var=dv, group_var=iv[0])
            else:
                pu.plot_distributions_by_group(data=data, obs_var=dv, group_var=iv[0])
                pu.plot_distributions_by_group(data=data, obs_var=dv, group_var=iv[1])

        # Run test of equal variance and if sig and not welch then throw warning
    var_stat, var_pval = assump.test_equal_variances(
        *[list(data[data[iv] == x][dv]) for x in groups], test=variance_test, disp=disp
    )
    if var_pval < 0.05:
        warnings.warn("test of equal variances was significant")

    if anova_type == "one_way":
        formula = dv + " ~ " + "C(" + iv + ")"
        model = ols(formula, data)
        model = _fit_anova(model)
        aov_table = anova_lm(model)
        eta2 = es.calc_eta_squared(
            f_stat=aov_table["F"].iloc[0],
            dof_groups=aov_table["df"].iloc[0],
            dof_residual=aov_table["df"].iloc[1],
        )
        aov_table["eta squared"] = [eta2, np.nan]
        aov_table.index = [iv, "Redidual"]

        agg_stats = u.get_agg_stats_means(data=data, iv=iv, dv=dv, disp=disp)

        if disp:
            display(aov_table.style.set_caption("ANOVA Table"))
            pu.group_box_plot(data=data, iv=iv, dv=dv)

        multi_comps = _run_aov_multi_comps(data=data, iv=iv, dv=dv, disp=disp)

        return [aov_table, agg_stats, multi_comps]

    elif anova_type == "rm":
        aov_rm = AnovaRM(
            data=data, depvar=dv, subject=obs_id, within=iv, aggregate_func="mean"
        )
        res = _fit_anova(aov_rm)
        aov_table = pd.DataFrame(res.summary().tables[0])

        agg_stats = u.get_agg_stats_means(data=data, iv=iv, dv=dv, disp=disp)

        if disp:
            display(aov_table.style.set_caption("ANOVA Table"))
            pu.group_box_plot(data=data, iv=iv, dv=dv)

        multi_comps = _run_aov_multi_comps(data=data, iv=iv[0], dv=dv, disp=disp)

        return [res, agg_stats, multi_comps]

    elif anova_type == "two_way":
        if type(iv) == str:
            warnings.warn("For two_way iv should be list of iv vars got string")
            return
        else:
            formula = dv + " ~ "
            tmp = []
            for i in iv:
                tmp.append("C(" + i + ")")
                formula = formula + "C(" + i + ") + "
            formula = formula + tmp[0] + ":" + tmp[1]

        model = ols(formula, data)
        model = _fit_anova(model)
        aov_table = anova_lm(model)
        aov_table["eta squared"] = aov_table[:-1]["sum_sq"] / sum(aov_table["sum_sq"])
        aov_table.index = [iv[0], iv[1], iv[0] + " : " + iv[1], "Residual"]

        agg_stats = u.get_agg_stats_means(data=data, iv=iv, dv=dv, disp=disp)

        if disp:
            display(aov_table.style.set_caption("ANOVA Table"))
            pu.group_box_plot(data=data, iv=iv, dv=dv)

        multi_comps = _run_aov_multi_comps(data=data, iv=iv, dv=dv, disp=disp)

        return [aov_table, agg_stats, multi_comps]

    else:
        warnings.warn(
            "anova type not supported expected 'one_way', 'two_way', or 'rm' got: "
            + anova_type
        )
        return None
