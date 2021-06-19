import pandas as pd
from statsmodels.graphics.regressionplots import plot_partregress_grid, plot_fit
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

sns.set_style("whitegrid")


def qq_plot_for_normality(data: pd.DataFrame = None, cols: list = None) -> None:
    """
    Wrapper function to use statsmodels qqplot functionality over columns of dataframe
    :param data: pandas dataframe default None.
    :return:
    """
    if len(cols) == 0:
        warnings.warn("No columns provided for plotting")
        return

    for col in cols:
        plot = qqplot(data=data[col], fit=True, line="45")
        plot.suptitle("QQ Plot of: " + col, fontsize=16)

    return None


def plot_distributions_by_group(
    data: pd.DataFrame = None, obs_var: str = None, group_var: str or list = None
) -> None:
    """
    Function to plot the distributions of the data by group
    :param data: pandas dataframe default None. Expects pandas dataframe containing obs_var col and group_var col
    :param obs_var: string default None. Expects column name containing values of observations
    :param group_var: string default None. Expects column names of groups comparing
    :return: None
    """

    if obs_var is None or obs_var not in list(data.columns):
        warnings.warn("obs_var is None or not in data")
        return
    if group_var is None or group_var not in list(data.columns):
        warnings.warn("group_var is None or not in data")
        return

    _ = plt.figure(figsize=(6, 6))
    ax = sns.kdeplot(
        data=data[[obs_var, group_var]],
        x=obs_var,
        hue=group_var,
        shade=True,
        legend=True,
    )
    ax.set_xlabel(obs_var, fontsize=20)
    ax.set_ylabel("", fontsize=20)
    ax.tick_params(labelsize=12)
    plt.show()

    return None


def group_box_plot(
    data: pd.DataFrame = None, iv: str or list = None, dv: str = None
) -> None:
    """
    Wrapper function to plot box plots
    :param data: pandas dataframe default None, expects pandas dataframe with columns related to ivs and dv
    :param iv: string or list default None, expects single string name of columns or
    list of string column names containing ivs
    :param dv: string default None, name of column containing dependent variable for analysis
    :return: None
    """

    if type(iv) == str:
        iv_1 = iv
        iv_2 = None
    elif type(iv) == list:
        if len(iv) == 2:
            iv_1 = iv[0]
            iv_2 = iv[1]
        else:
            iv_1 = iv[0]
            iv_2 = None
    else:
        warnings.warn(
            "iv type not supported expects string or list got: " + str(type(iv))
        )
        return

    _ = plt.figure(figsize=(6, 6))
    ax = sns.boxplot(x=iv_1, y=dv, hue=iv_2, data=data)
    ax.set_xlabel(iv_1, fontsize=20)
    ax.set_ylabel(dv, fontsize=20)
    ax.tick_params(labelsize=12)
    return


def plot_linear_regress_fit() -> None:
    return None


def plot_ols_regress_components(model=None) -> None:
    """
    Wrapper function to plot the regress lines by component of model
    :param model: statsmodel ols
    :return: None
    """

    plot_partregress_grid(results=model, fig=plt.figure(figsize=(8, 8)))

    return None


def plot_residual_diagnostics(model=None) -> None:
    """
    Function to plot residual diagnostics
    :param model: statsmodel ols default None. Expects fitted statsmodel ols model
    :return: None
    """

    df = pd.DataFrame({"Residuals": model.resid, "Fitted Values": model.fittedvalues})

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle("Residual Diagnostics")
    rf = sns.scatterplot(ax=ax[0], x=df["Fitted Values"], y=df["Residuals"])
    rf.axhline(y=0, linestyle="--", color="r")
    ax[0].set_title("Residuals vs Fitted Values")
    qqplot(ax=ax[1], data=df["Residuals"], fit=True, line="45")
    ax[1].set_title("Residual QQ Plot")

    return None
