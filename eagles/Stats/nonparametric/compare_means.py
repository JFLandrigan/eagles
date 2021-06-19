from eagles.Stats import utils as u, plot_utils as pu, effect_size as es
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, kruskal
from IPython.display import display
import warnings


def wilcoxon_signed_rank_test(
    data: pd.DataFrame = None,
    iv: str = None,
    dv: str = None,
    tails: str = "two-sided",
    disp: bool = True,
) -> list:
    """
    Function to perform wilcoxon signed rank test
    :param data: pandas dataframe default None, dataframe containg column specified by iv and dv params
    :param iv: string default None, column name containing grouping and/or measurement indicator (e.g. t1, t2...)
    :param dv: string default None, column name containg the outcome of interest
    :param tails: string default 'two-sided'. String indicating to perform one tail or two tail test.
    Expects 'two-sided', 'greater', 'less'
    :param disp: boolean deafult True, boolean indicator of whether or not to display results
    :return:
    """

    if not all([data, iv, dv]):
        warnings.warn("Missing required arguments. Returning None")
        return None

    x = data[iv].unique()[0]
    y = data[iv].unique()[1]

    w_stat, p_val = wilcoxon(
        x=data[data[iv] == x][dv], y=data[data[iv] == y][dv], alternative=tails
    )

    ef_sz = es.calc_wilcoxon_signed_rank_r(x=x, y=y)

    res = pd.DataFrame(
        {
            "Statistic": ["test statistic", "p Value", "effect size r"],
            "Value": [w_stat, p_val, ef_sz],
        }
    )

    agg_stats = u.get_agg_stats_means(data=data, iv=iv, dv=dv, disp=disp)

    if disp:
        display(res.style.set_caption("Test Results").hide_index())
        pu.group_box_plot(data=data, iv=iv, dv=dv)

    return [res, agg_stats]


def man_whitney_u_test(
    data: pd.DataFrame = None,
    iv: str = None,
    dv: str = None,
    tails: str = "two-sided",
    disp: bool = True,
) -> list:
    """
    Function to perform man whitney u test.
    :param data: pandas dataframe default None, dataframe containg column specified by iv and dv params
    :param iv: string default None, column name containing grouping and/or measurement indicator (e.g. t1, t2...)
    :param dv: string default None, column name containg the outcome of interest
    :param tails: str default 'two-sided', takes two or one indicator for if two sided or one sided.
    If one tailed test expects 'less' or 'greater'
    :param disp:
    :return:
    """

    if not all([data, iv, dv]):
        warnings.warn("Missing required arguments. Returning None")
        return None

    x = data[iv].unique()[0]
    y = data[iv].unique()[1]

    u_stat, p_val = mannwhitneyu(
        x=data[data[iv] == x][dv], y=data[data[iv] == y][dv], alternative=tails
    )
    ef_sz = es.calc_man_whitney_u_r(u_stat=u_stat, n1=len(x), n2=len(y))

    res = pd.DataFrame(
        {
            "Statistic": ["w Statistic", "p Value", "effect size r"],
            "Value": [u_stat, p_val, ef_sz],
        }
    )

    agg_stats = u.get_agg_stats_means(data=data, iv=iv, dv=dv, disp=disp)

    if disp:
        display(res.style.set_caption("Test Results").hide_index())
        pu.group_box_plot(data=data, iv=iv, dv=dv)

    return [res, agg_stats]


def kruskal_wallis_test(
    data: pd.DataFrame = None, iv: str or list = None, dv: str = None, disp: bool = True
) -> list:
    """
    Function to perform a kruskal wallis test.
     :param data: pandas dataframe default None, expects pandas dataframe with columns related to ivs and dv
    :param iv: string default None, expects single string name of column
    :param dv: string default None, name of column containing dependent variable for analysis
    :param disp:
    :return:
    """
    if not all([data, iv, dv]):
        warnings.warn("Missing required arguments. Returning None")
        return None

    groups = list(data[iv].unique())
    h_stat, p_val = kruskal(
        *[list(data[data[iv] == x][dv]) for x in groups], nan_policy="omit"
    )
    ef_sz = es.calc_kruskal_eta_squared(
        h_stat=h_stat, n=data.shape[0], n_groups=len(groups)
    )
    res = pd.DataFrame(
        {
            "Statistic": ["h Statistic", "p Value", "eta squared"],
            "Value": [h_stat, p_val, ef_sz],
        }
    )

    agg_stats = u.get_agg_stats_means(data=data, iv=iv, dv=dv, disp=disp)

    if disp:
        display(agg_stats)
        display(res)

    return [res, agg_stats]
