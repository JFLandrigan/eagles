from eagles.Stats import utils as u, config, power as p, effect_size as es
import pandas as pd
import numpy as np
from IPython.display import display


def z_proportions_test(
    data: pd.DataFrame = None,
    group: str = None,
    outcome: str = None,
    alpha: float = 0.05,
    tails: str = "two",
    disp: bool = True,
) -> list:
    """
    Function to perform a z test to compare two population proportions
    :param data: pandas dataframe default None, Expects pandas dataframe with columns
    relative to group and outcome params
    :param group: string default None, Title of the pandas dataframe col containing the group for observations
    :param outcome: string default None, Title of the pandas dataframe column containing
    the outcome (i.e. 1 or 0) of observation
    :param alpha: float default 0.05, significance level desired
    :param tails: string default "two", expects string indicating one or two tailed test.
    If one tailed expects "one_less" or "one_greater"
    :param disp: bool default True, boolean indicator to display results or not
    :return: pandas dataframe containing relevant stats
    """

    groups = list(data[group].unique())
    p1 = data[data[group] == groups[0]][outcome].mean()
    p2 = data[data[group] == groups[1]][outcome].mean()
    p_total = data[outcome].mean()
    n1 = len(data[data[group] == groups[0]])
    n2 = len(data[data[group] == groups[1]])

    z_stat = (p1 - p2) / np.sqrt((p_total * (1 - p_total)) * ((1 / n1) + (1 / n2)))
    p_val = u.get_p_value_zstat(z_stat=z_stat, tails=tails)
    ef_sz = es.z_test(z_stat=z_stat, n1=n1, n2=n2)
    power = p.power_achieved_two_proportions(
        effect_size=ef_sz, n1=n1, n2=n2, alpha=alpha, tails=tails
    )
    stats = [z_stat, p_val, ef_sz, power]

    res = pd.DataFrame({"Statistic": config.z_test_cols, "Value": stats})
    agg_stats = u.get_agg_stats_proportions(
        data=data, group=group, outcome=outcome, disp=disp
    )
    if disp:
        display(res.style.set_caption("Z Test Results"))

    return [res, agg_stats]
