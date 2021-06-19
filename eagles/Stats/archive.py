# one way anova scipy
# groups = list(data[iv].unique())
# f_stat, pval = f_oneway(*[list(data[data[iv] == x][dv]) for x in groups])
# dof_groups = len(groups) - 1
# dof_residuals = data.shape[0] - len(groups)


# def t_test(
#     obs_1: np.array = None,
#     obs_2: np.array = None,
#     paired: bool = False,
#     alpha: float = 0.05,
#     tails: str = "two",
#     welch: bool = False,
#     hedges_g: bool = False,
#     variance_test: str = "levene",
#     disp: bool = True,
#     labels: list = None,
# ) -> pd.DataFrame:
#     """
#     Function to perform either a paired or independent samples t test
#     :param obs_1: numpy array containing observations relative to group 1
#     :param obs_2: numpy array containing observations relative to group 2
#     :param paired: bool default False, boolean indicator to run paired or independent samples t test
#     :param alpha: float default .05, alpha level for test
#     :param tails: str default two, takes two or one indicator for if two sided or one sided.
#     If one tailed test expects 'one_less' or 'one_greater'
#     :param welch: bool default False, indicator to perform welches t test for unequal varince
#     :param hedges_g: bool default False, indicator to calculate hedges g instead of cohens d
#     :param variance_test: string default levene. Which test to perform for equal variances.
#     Expects either levene or bartlett
#     :param disp: bool default True, indicator to display results or not
#     :param labels: list default None, expects list of labels with index 0 = obs1 and index 1 = obs2
#     :return:
#     """
#
#     mn1 = obs_1.mean()
#     mn2 = obs_2.mean()
#     s1 = obs_1.var(ddof=1)
#     s2 = obs_2.var(ddof=1)
#     n1 = len(obs_1)
#     n2 = len(obs_2)
#
#     # Run test of normality and plot distributions
#     _ = assump.test_normality(x=np.append(obs_1, obs_2), disp=disp)
#     if disp:
#         if labels is not None:
#             groups = np.append(
#                 np.repeat(labels[0], len(obs_1)), np.repeat(labels[1], len(obs_2))
#             )
#         else:
#             groups = np.append(
#                 np.repeat("obs_1", len(obs_1)), np.repeat("obs_2", len(obs_2))
#             )
#         data = pd.DataFrame({"groups": groups, "obs": np.append(obs_1, obs_2)})
#         pu.plot_distributions_by_group(data=data, obs_var="obs", group_var="groups")
#
#     # Run test of equal variance and if sig and not welch then throw warning
#     var_stat, var_pval = assump.test_equal_variances(
#         obs_1, obs_2, test=variance_test, disp=disp
#     )
#     if var_pval < 0.05 and not welch:
#         warnings.warn("test of equal variances was significant and not using welch")
#
#     # Do quick check for sample sizes when running a paired t-test
#     if paired:
#         if n1 != n2:
#             warnings.warn("Number of observations does not match between obs")
#             return None
#         else:
#             dof = n1 - 1
#             # standard dev of differences  div by sqrt of sample size
#             denom = (obs_1 - obs_2).std() / np.sqrt(n1)
#     else:
#         if not welch:
#             dof = n1 + n2 - 2
#             denom = np.sqrt(
#                 (((n1 - 1) * s1 + (n2 - 1) * s2) / dof) * ((1 / n1) + (1 / n2))
#             )
#         else:
#             dof = ((s1 / n1) + (s1 / n2)) ** 2 / (
#                 (s1 / n1) ** 2 / (n1 - 1) + (s1 / n2) ** 2 / (n2 - 1)
#             )
#             denom = np.sqrt((s1 / n1) + (s1 / n2))
#
#     t_stat = (mn1 - mn2) / denom
#
#     critical_value = u.calc_students_critical_value(alpha=alpha, dof=dof, tails=tails)
#     p_val = u.get_p_value_tstat(t_stat, dof)
#
#     if tails == "one_less":
#         if t_stat < 0:
#             p_val = p_val / 2
#         else:
#             p_val = 1 - p_val / 2
#     elif tails == "one_greater":
#         if t_stat > 0:
#             p_val = p_val / 2
#         else:
#             p_val = 1 - p_val / 2
#
#     ef_sz = u.calc_cohens_d(obs_1=obs_1, obs_2=obs_2, paired=paired, hedges_g=hedges_g)
#     power = p.power_achieved_ttest(
#         critical_value=critical_value, d=ef_sz, n1=n1, n2=n2, tails=tails, paired=paired
#     )
#
#     # calc the 95% CI bounds
#     if tails == "two":
#         div = 2
#     else:
#         div = 1
#     lb = round((mn1 - mn2) - t.ppf((1 - (alpha / div)), dof) * denom, 4)
#     ub = round((mn1 - mn2) + t.ppf((1 - (alpha / div)), dof) * denom, 4)
#
#     # Put results into a df (put cols in a config file to reduce the code lines
#     stat_values = [
#         [round(mn1, 4), round(obs_1.std(), 4)],
#         [round(mn2, 4), round(obs_2.std(), 4)],
#         t_stat,
#         dof,
#         critical_value,
#         p_val,
#         [lb, ub],
#         round(ef_sz, 4),
#         power,
#     ]
#     res = pd.DataFrame({"Statistic": config.t_test_cols, "Value": stat_values,})
#
#     if disp:
#         display(res)
#         pu.t_test_group_plot(obs_1=obs_1, obs_2=obs_2, labels=labels)
#
#     return res


# contains code with mixed effect models
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from IPython.display import display
from io import StringIO
import warnings

#
# # TODO add in mixed effect models
#
#
# def _fit_regress(model, logit: bool = False):
#     # Get intermittent bug of numpy LinAlgError: SVD did not converge
#     # so wrapped in try while true to get around issue
#     # see this discussion
#     # https://stackoverflow.com/questions/63761366/numpy-linalg-linalgerror-svd-did-not-converge-in-linear-least-squares-on-first
#     try_cnt = 1
#     while True:
#         try:
#             if logit:
#                 model = model.fit(method="bfgs")
#             else:
#                 model = model.fit()
#             break
#         except:
#             try_cnt += 1
#             if try_cnt <= 20:
#                 continue
#             else:
#                 warnings.warn("Fit failure")
#                 return
#     return model
#
#
# def regression(
#     data: pd.DataFrame = None,
#     formula: str = None,
#     regress_type: str = "ols",
#     random_effects: dict = None,
#     family=sm.families.Binomial(),
#     groups: str = None,
#     plot_fit: bool = True,
#     disp: bool = True,
# ):
#     """
#     Function to perform regression on outcome
#     :param formula: string default None, expects forumla for regression (i.e. "DV ~ IV1 + IV2")
#     :param data: pandas DataFrame default None, expects pandas dataframe with column names relating to formula
#     :param regress_type: string default 'ols', string indicating regression type. Expects 'ols', 'logistic', 'mixed_lm' or 'mixed_glm'
#     :param random_effects: dictionary default None, expects dictionary containing specified
#     random effects see https://www.statsmodels.org/dev/generated/statsmodels.formula.api.mixedlm.html. Ignored when
#     regress_type is ols, or logistic
#     :param family: statsmodels family instance default Binomial(). see https://tedboy.github.io/statsmodels_doc/doc/glm.html#families
#     :param groups: string default none. expects string defining the groups
#     :param plot_fit: bool default True, boolean indicator to plot regression fit
#     :param disp: bool default True, boolean indicator to print out / display the model fit results
#     :return: pandas dataframe with results of regression
#     """
#
#     if regress_type == "ols":
#         mod = smf.ols(formula=formula, data=data)
#         mod = _fit_regress(model=mod)
#     elif regress_type == "logistic":
#         mod = smf.logit(formula=formula, data=data)
#         mod = _fit_regress(mod, logit=True)
#     elif regress_type == "mixed_lm":
#         mod = smf.mixedlm(
#             formula=formula, data=data, vc_formula=random_effects, groups=groups
#         )
#         mod = _fit_regress(model=mod)
#     elif regress_type == "mixed_glm":
#         mod = smf.glm(formula=formula, data=data, family=family)
#         mod = _fit_regress(model=mod)
#     else:
#         warnings.warn(
#             "regress_type not supported. Expected 'ols', 'logistic' or 'mixed' got: "
#             + regress_type
#         )
#         return None
#
#     # Get the model summary and put it in a DF
#     if regress_type in ["ols", "logistic"]:
#         fit_table = mod.summary().tables[0]
#         coef_table = pd.read_csv(StringIO(mod.summary().tables[1].as_csv()))
#     elif regress_type in ["mixed_glm"]:
#         fit_table = None
#         coef_table = pd.read_csv(StringIO(mod.summary().tables[1].as_csv()))
#     elif regress_type in ["mixed_lm"]:
#         fit_table = None
#         coef_table = mod.summary().tables[1]
#
#     coef_table.columns = [col.strip() for col in coef_table.columns]
#     # display(coef_table)
#
#     if regress_type not in ["logistic", "mixed_lm", "mixed_glm"]:
#         assumption_table = pd.read_csv(
#             StringIO(mod.summary().tables[2].as_csv()), header=None
#         )
#         assumption_table.columns = ["" for i in range(len(assumption_table.columns))]
#
#     if disp:
#         if fit_table:
#             display(fit_table)
#         if regress_type not in ["logistic", "mixed_lm", "mixed_glm"]:
#             display(assumption_table)
#
#         coef_table_t = coef_table.copy(deep=True)
#         coef_table_t.reset_index(drop=True, inplace=True)
#         coef_table_t.rename(columns={"index": ""}, inplace=True)
#         end_row = (
#             len(coef_table_t) - 2
#             if regress_type in ["mixed_lm", "mixed_glm"]
#             else len(coef_table_t) - 1
#         )
#         display(
#             coef_table_t.style.set_caption("Coefficients")
#             .applymap(
#                 lambda x: "color: green" if float(x) < 0.05 else "color: red",
#                 subset=pd.IndexSlice[
#                     0:end_row, ["P>|t|" if regress_type == "ols" else "P>|z|"]
#                 ],
#             )
#             .hide_index()
#         )
#
#     return coef_table
#
#
#
# regress disp code
# if fit_table:
#     display(fit_table)
# if regress_type not in ["logistic"]:
#     display(assumption_table)
#
# coef_table_t = coef_table.copy(deep=True)
# coef_table_t.reset_index(drop=True, inplace=True)
# coef_table_t.rename(columns={"index": ""}, inplace=True)
# end_row = len(coef_table_t) - 1
# display(
#     coef_table_t.style.set_caption("Coefficients")
#         .applymap(
#         lambda x: "color: green" if float(x) < 0.05 else "color: red",
#         subset=pd.IndexSlice[
#                0:end_row, ["P>|t|" if regress_type == "linear" else "P>|z|"]
#                ],
#     )
#         .hide_index()
# )
