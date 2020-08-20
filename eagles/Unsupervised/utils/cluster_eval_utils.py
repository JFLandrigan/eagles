import pandas as pd
from scipy import stats
from statsmodels.stats import multicomp as mc


def create_summary_table(data, plot_dims=[], summary_stats=[]):

    if len(summary_stats) == 0:
        summary_stats = ["mean", "std"]

    tmp = data.copy(deep=True)
    tmp = tmp.groupby(["Cluster"])[plot_dims].agg(summary_stats)

    return tmp


def run_cluster_comps(data=None, ft_cols=None):
    """
    Function to determine where statistically sig differences lie between the clusters
    :param data:
    :return:
    """
    # Get the binary columns
    bin_fts = [col for col in ft_cols if list(set(data[col])) == [0, 1]]
    # Get the continuous columns
    cont_fts = [col for col in ft_cols if col not in bin_fts]
    # init the sig df and post hoc tests df
    sig_results = {"Feature": list(), "p Val": list(), "Stat Test": list()}
    post_hocs = pd.DataFrame()

    # perform chi squared on the binary
    for ft in bin_fts:
        crosstab = pd.crosstab(data["Cluster"], data[ft])
        res = stats.chi2_contingency(crosstab)

        if res[1] < 0.05:
            sig_results["Feature"].append(ft)
            sig_results["p Val"].append(res[1])
            sig_results["Stat Test"].append("chi2")

    # perform one way anova on the continuous
    sig_cont = list()
    for ft in cont_fts:

        result = data.groupby("Cluster")[ft].apply(list)
        fval, p = stats.f_oneway(*result)

        if p < 0.05:
            sig_results["Feature"].append(ft)
            sig_cont.append(ft)
            sig_results["p Val"].append(p)
            sig_results["Stat Test"].append("ANOVA")

    # store the sig results in df
    sig_df = pd.DataFrame(sig_results)

    if sig_df.shape[0] == 0 or len(sig_cont) == 0:
        return sig_df, post_hocs

    elif len(sig_cont) > 0:
        for ft in sig_cont:
            res = mc.pairwise_tukeyhsd(
                endog=data[ft], groups=data["Cluster"], alpha=0.05
            )

            results_summary = res.summary()
            res_df = pd.read_html(results_summary.as_html(), header=0, index_col=0)[
                0
            ].reset_index()
            res_df = res_df[res_df["reject"] == True]
            res_df["feature"] = ft

            post_hocs = pd.concat([post_hocs, res_df])

        post_hocs = post_hocs[["feature", "group1", "group2", "meandiff", "p-adj"]]
        post_hocs["stat test"] = "Tukey HSD"

        return sig_df, post_hocs
