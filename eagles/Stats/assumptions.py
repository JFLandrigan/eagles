import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest, ks_1samp, levene, bartlett, norm
from eagles.Stats import config
from IPython.display import display
import warnings


def test_normality(
    test: str = None, x: np.array = None, disp: bool = True
) -> pd.DataFrame:
    """
    Function to run tests of normality, including shapriro wilkes, D’Agostino’s K^2 test, Anderson-Darling Test.
    Tests implemented using scipy stats module
    :param test: string default None, expects either shapiro, dagostino, ks or None. If None then runs all tests
    :param x: numpy array default None, expects numpy array of data points
    :param disp: boolean default True, boolean indicator for whether or not to display results of test
    :return: list containing shapiro wilkes test stat and p val
    """
    if test is not None:
        tests = [test]
    else:
        tests = ["shapiro", "dagostino", "ks"]

    res = pd.DataFrame()

    for test in tests:
        if test == "shapiro":
            stat, p_val = shapiro(x)
            test_name = config.normality_tests[0]
        elif test == "dagostino":
            stat, p_val = normaltest(x)
            test_name = config.normality_tests[1]
        elif test == "ks":
            stat, p_val = ks_1samp(x, norm.cdf)
            test_name = config.normality_tests[2]
        else:
            warnings.warn(
                "Normality test not supported. Expects shapiro, dagostino, ks or None got: "
                + str(test)
            )

        res = pd.concat(
            [
                res,
                pd.DataFrame(
                    {
                        "Normality Test": [test_name],
                        "Test Statistic": [stat],
                        "p Value": [p_val],
                    }
                ),
            ],
        )

    if any(res["p Value"] < 0.05):
        warnings.warn(
            "Performing test of normality resulted in significant value at p threshold of < 0.05"
        )

    if disp:
        display(round(res, 4))

    return res


def test_equal_variances(*args, test: str = "levene", disp: bool = True) -> list:
    """
    Function to test for equal variances between samples
    :param args: Takes variable number of arrays
    :param test: string default 'levene'. expects string indicating which test to run either levene or bartlett
    :param disp: bool default True, boolean indicator to display test results or not
    :return: list, test statistic and p value
    """

    if test == "levene":
        res = levene(*args)
        stat = res[0]
        pval = res[1]
    elif test == "bartlett":
        res = bartlett(*args)
        stat = res[0]
        pval = res[1]
    else:
        warnings.warn(
            "equal variance test not supported expected levene or bartlett got: " + test
        )
        return None

    if disp:
        print(
            test
            + " test statistic was: "
            + str(round(stat, 4))
            + " p value of: "
            + str(round(pval, 4))
        )
    return [stat, pval]
