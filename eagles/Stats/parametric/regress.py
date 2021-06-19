from eagles.Stats import plot_utils as pu
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson, NegativeBinomial
from scipy.stats import skew, kurtosis
from IPython.display import display
from io import StringIO
import warnings

# TODO convert from using the model summary and then to pandas to just pulling the params from the model directly
# TODO add a "fit" plot
# TODO check for a model comparison module like in R where do aov(mod1, mod2 ....)

# TODO add in mixed effect models. make it so that take in string formula and parse into the dictionary

# TODO for the poisson model figure out how to add in the offset term https://stackoverflow.com/questions/42194356/offset-argument-in-statsmodels-api-glm
#  https://github.com/JaehyunAhn/statsmodels/blob/master/statsmodels/genmod/generalized_linear_model.py#L282

# TODO Figure out how to implement quasipoisson, negative binomial model, zero inflated model (part of the poisson model chapter beyond mlr book)


def _build_poisson_tables(mod=None) -> list:
    coef_table = pd.DataFrame(mod.params)
    coef_table.columns = ["Estimate"]
    coef_table["Std. Error"] = mod.bse  # .050 .001
    coef_table["z"] = mod.tvalues
    coef_table["P>|z|"] = mod.pvalues
    coef_table["[0.025"] = mod.conf_int()[0]
    coef_table["0.975]"] = mod.conf_int()[1]

    qs = mod.resid_deviance.quantile([0.25, 0.75])

    residual_table = pd.DataFrame(
        {
            "Min": [mod.resid_deviance.min()],
            "1Q": [qs.iloc[0]],
            "Median": [mod.resid_deviance.median()],
            "3Q": [qs.iloc[1]],
            "Max": [mod.resid_deviance.max()],
            "Skew": [skew(mod.resid_deviance)],
            "Kurtosis": [kurtosis(mod.resid_deviance)],
        }
    )

    fit_table = pd.DataFrame(
        {
            "Residual Deviance": [sum(mod.resid_deviance ** 2)],
            "Residual dof": [mod.df_resid],
            "AIC": [mod.aic],
            "BIC": [mod.bic],
        }
    )
    return [coef_table, residual_table, fit_table]


def _fit_regress(model, logit: bool = False):
    # Get intermittent bug of numpy LinAlgError: SVD did not converge
    # so wrapped in try while true to get around issue
    # see this discussion
    # https://stackoverflow.com/questions/63761366/numpy-linalg-linalgerror-svd-did-not-converge-in-linear-least-squares-on-first
    try_cnt = 1
    while True:
        try:
            if logit:
                model = model.fit(method="bfgs")
            else:
                model = model.fit()
            break
        except:
            try_cnt += 1
            if try_cnt <= 20:
                continue
            else:
                warnings.warn("Fit failure")
                return
    return model


def regression(
    data: pd.DataFrame = None,
    formula: str = None,
    regress_type: str = "linear",
    plot_diagnostics: bool = True,
    plot_regress_components: bool = True,
    disp: bool = True,
) -> dict:
    """
    Function to perform regression on outcome
    :param formula: string default None, expects forumla for regression (i.e. "DV ~ IV1 + IV2")
    :param data: pandas DataFrame default None, expects pandas dataframe with column names relating to formula
    :param regress_type: string default 'linear', string indicating regression type. Expects 'linear','logistic', 'poisson'.
    Logistic and Poisson models are git using glm methods
    :param plot_diagnostics: bool default True, boolean indicator to plot residual diagnostic plots
    (if disp False then ignored)
    :param plot_regress_components: bool default True, boolean indicator to plot partial fits of regression components
    (if disp False then ignored)
    :param disp: bool default True, boolean indicator to print out / display the model fit results
    :return: pandas dataframe with results of regression
    """

    res_dict = {}
    fit_table = pd.DataFrame()
    residual_table = pd.DataFrame()

    if regress_type == "linear":
        mod = smf.ols(formula=formula, data=data)
        mod = _fit_regress(model=mod)
        coef_table = pd.read_csv(StringIO(mod.summary().tables[1].as_csv()))
        coef_table.columns = [col.strip() for col in coef_table.columns]
    elif regress_type == "logistic":
        mod = smf.logit(formula=formula, data=data)
        mod = _fit_regress(mod, logit=True)
    elif regress_type == "poisson":
        mod = smf.glm(formula=formula, data=data, family=Poisson())
        mod = _fit_regress(model=mod)
        coef_table, residual_table, fit_table = _build_poisson_tables(mod=mod)
    else:
        warnings.warn(
            "regress_type not supported. Expected 'linear', 'logistic' got: "
            + regress_type
        )
        return None

    # Get the model summary and put it in a DF
    if regress_type in ["linear", "logistic"]:
        qs = mod.resid.quantile([0.25, 0.75])
        residual_table = pd.DataFrame(
            {
                "Min": [mod.resid.min()],
                "1Q": [qs.iloc[0]],
                "Median": [mod.resid.median()],
                "3Q": [qs.iloc[1]],
                "Max": [mod.resid.max()],
                "Skew": [skew(mod.resid)],
                "Kurtosis": [kurtosis(mod.resid)],
            }
        )

        fit_table = pd.DataFrame(
            {
                "R2": [mod.rsquared],
                "Adj R2": [mod.rsquared_adj],
                "AIC": [mod.aic],
                "BIC": [mod.bic],
            }
        )

    if disp:
        if residual_table.shape[0] > 0:
            display(residual_table)
        print(" ")
        display(coef_table)
        print(" ")
        if fit_table.shape[0] > 0:
            display(fit_table)

    if plot_diagnostics and regress_type == "linear":
        pu.plot_residual_diagnostics(model=mod)

    if plot_regress_components and regress_type in ["linear"]:
        pu.plot_ols_regress_components(model=mod)

    res_dict["model"] = mod
    res_dict["residual_table"] = residual_table
    res_dict["coef_table"] = coef_table
    res_dict["fit_table"] = fit_table

    return res_dict
