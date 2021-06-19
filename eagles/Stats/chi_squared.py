from eagles.Stats import config, utils as u, power as p, effect_size as es
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chisquare, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from IPython.display import display
import warnings


def chi_2_goodness_fit(
    data: pd.DataFrame = None,
    category: str = None,
    expected_frequencies: np.array = None,
    alpha: float = 0.05,
    disp: bool = True,
):
    """
    Function to perform chi2 goodness of fit.
    :param data: pandas dataframe default None. Expects pandas dataframe containing category col
    :param category: string default none. Expects string label of col containing data to perform test on
    :param expected_frequencies: numpy array default None, Expected frequencies when null hypothesis is true.
    Can be left none if expected frequencies are equal to 1 / number categories
    :param alpha: float default 0.05. Significance level
    :param disp: bollean default True. Boolean indicator to didplay results or not
    :return: list with test stat results and observed frequencies
    """

    if data is None or category is None:
        warnings.warn("Missing data or category col label")
        return

    observed_frequencies = np.array(data[category].value_counts())
    dof = len(observed_frequencies) - 1
    chi2_stat, p_val = chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)
    ef_sz = es.calc_chi2_goodness_fit_w(obs=observed_frequencies)
    power = p.power_achieved_chi2(
        alpha=alpha, dof=dof, effect_size=ef_sz, n_obs=sum(observed_frequencies)
    )
    stats = [chi2_stat, dof, p_val, ef_sz, power]
    res = pd.DataFrame({"Statistic": config.chi_2_cols, "Value": stats})
    res.rename(columns={"effect size": "cohens w"}, inplace=True)

    if disp:
        display(
            pd.DataFrame(data[category].value_counts()).style.set_caption("Observed")
        )
        display(res.style.set_caption("Test Results"))

    return [res, observed_frequencies]


def chi_2_test_independence(
    data: pd.DataFrame = None,
    category_cols: list = None,
    alpha: float = 0.05,
    effect_size_type: str = "phi",
    yates: bool = False,
    disp: bool = True,
) -> list:
    """
    Function to perform a chi2 test of independence
    :param data: pandas dataframe default None, Dataframe containing the category columns to run the chi2 on
    :param category_cols: list default None, List containing column names to run chi2 test of independence on
    :param alpha: float default 0.05, alpha value for given test
    :param effect_size_type string default 'phi', string indicating which effect size to compute.
    Options include phi, cramers, odds_ratio
    :param yates: bool default False, boolean indicator to perform yates correction method or not
    :param disp: bool default True, boolean indicator for whether or not to display results
    :return: list, dataframe containing results and contingency table
    """

    if len(category_cols) != 2:
        warnings.warn("Expected 2 column names for test but got: " + str(category_cols))

    n = data.shape[0]
    contingency = pd.crosstab(data[category_cols[0]], data[category_cols[1]])

    # check for category sizes
    size_check = contingency < 5
    if np.array(size_check).any():
        warnings.warn(
            "Cell in observed frequencies contains less then 5 observations but it\
             is recommended to have greater then 5 in each category. Try using stats.chi_squared.fishers_exact"
        )

    chi2_stat, p_val, dof, _ = chi2_contingency(observed=contingency, correction=yates)
    ef_sz = es.calc_chi2_independence_es(
        effect_type=effect_size_type,
        chi2_stat=chi2_stat,
        n_obs=n,
        dof=dof,
        contingency=contingency,
    )

    if dof == 1 and yates == False:
        warnings.warn("Degrees of freedom was 1 and yates correction was not used")

    power = p.power_achieved_chi2(alpha=alpha, dof=dof, effect_size=ef_sz, n_obs=n)

    stats = [chi2_stat, dof, p_val, ef_sz, power]
    res = pd.DataFrame({"Statistic": config.chi_2_cols, "Value": stats})

    if disp:
        display(pd.DataFrame(contingency).style.set_caption("Observed"))
        display(res.style.set_caption("Test Results"))

    return [res, contingency]


# TODO add degrees of freedom


def fishers_exact(
    data: pd.DataFrame = None, category_cols: list = None, disp: bool = True,
) -> list:
    """
    Function to run fishers exact test.
    :param data:
    :param category_cols:
    :param alpha:
    :param disp: boolean default True. Boolean indicator to display test results or not
    :return:
    """

    if len(category_cols) > 2 or data is None:
        warnings.warn(
            "Incorrect data provided check that data is pandas dataframe and category_cols contains 2 string labels"
        )
        return None

    observed_frequencies = pd.crosstab(data[category_cols[0]], data[category_cols[1]])
    odds_ratio, p_val = fisher_exact(table=observed_frequencies)

    stats = [odds_ratio, p_val]
    res = pd.DataFrame({"Statistic": ["odds ratio", "p value"], "Value": stats})

    if disp:
        display(observed_frequencies.style.set_caption("Contingency Table"))
        display(res.style.set_caption("Test Results"))

    return [res, observed_frequencies]


def mcnemar_test(
    data: pd.DataFrame = None, category_cols: list = None, disp: bool = True
) -> list:
    """
    Function to run McNemar Test. Should be used when have repeated measures design with categorical data.
    :param data: pandas dataframe default None. Expects pandas dataframe containing columns specified in category_cols
    :param category_cols: list default None. List containing string labels of columns containing data to run test on
    :param disp: boolean default True. Boolean indicator to display test results or not
    :return: list containg pandas dataframe with test results and contingency table of observed frequencies.
    """

    if data is None:
        warnings.warn("Missing required argument data")
        return None

    if len(category_cols) > 2:
        warnings.warn(
            "More then 2 categories provided, you may want to use stats.chi_squared.cochrans_q_test"
        )
        return None

    observed_frequencies = pd.crosstab(data[category_cols[0]], data[category_cols[1]])
    test_stat, p_val = mcnemar(table=observed_frequencies)
    # calculate the degrees of freedom
    dof = (observed_frequencies.shape[0] - 1) * (observed_frequencies.shape[1] - 1)
    stats = [test_stat, dof, p_val]
    res = pd.DataFrame(
        {"Statistic": ["McNemar x2 stat", "dof", "p value"], "Value": stats}
    )

    if disp:
        display(observed_frequencies.style.set_caption("Contingency Table"))
        display(res.style.set_caption("Test Results"))

    return [res, observed_frequencies]


def cochrans_q_test(
    data: pd.DataFrame = None, category_cols: list = None, disp: bool = True
) -> list:
    """
    Function to run cochrans q test. Expects data to be coded as 1 and 0
    :param data: pandas dataframe default None. Dataframe containing columns listed in category cols
    :param category_cols: list default None. List containing string names of columns containing data for test
    :param disp: boolean default True. Boolean indicator to display results or not
    :return: pandas dataframe containing test results and array of summed category cols
    """
    observed_frequencies = data[category_cols].sum()
    q_stat, p_val = cochrans_q(x=data[category_cols])
    dof = len(category_cols) - 1
    stats = [q_stat, dof, p_val]
    res = pd.DataFrame({"Statistic": ["q Stat", "dof", "p value"], "Value": stats})

    if disp:
        display(observed_frequencies.style.set_caption("Contingency Table"))
        display(res.style.set_caption("Test Results"))
    return [res, observed_frequencies]
