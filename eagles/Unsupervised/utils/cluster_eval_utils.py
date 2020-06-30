import pandas as pd
import pingouin as pg


def create_summary_table(data, plot_dims=[], summary_stats=[]):

    if len(summary_stats) == 0:
        summary_stats = ["mean", "std"]

    tmp = data.copy(deep=True)
    tmp = tmp.groupby(["Cluster"])[plot_dims].agg(summary_stats)

    print("Base Cluster Stats \n")
    print(round(tmp.T, 2))
    print("\n\n")

    return


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
    sig_results = {"Feature": list(), "p Val": list(), "Effect Size": list()}
    post_hocs = pd.DataFrame()

    # perform chi squared on the binary
    for ft in bin_fts:
        expected, observed, stats = pg.chi2_independence(data, x="Cluster", y=ft)
        if stats[stats["test"] == "log-likelihood"]["p"].iloc[0] < 0.05:
            sig_results["Feature"].append(ft)
            sig_results["p Val"].append(
                stats[stats["test"] == "log-likelihood"]["p"].iloc[0]
            )
            sig_results["Effect Size"].append(
                stats[stats["test"] == "log-likelihood"]["cramer"].iloc[0]
            )

    # perform one way anova on the continuous
    sig_cont = list()
    for ft in cont_fts:
        aov = pg.anova(data=data, dv=ft, between="Cluster", detailed=True)
        if aov[aov["Source"] == "Cluster"]["p-unc"].iloc[0] < 0.05:
            sig_results["Feature"].append(ft)
            sig_cont.append(ft)
            sig_results["p Val"].append(
                aov[aov["Source"] == "Cluster"]["p-unc"].iloc[0]
            )
            sig_results["Effect Size"].append(
                aov[aov["Source"] == "Cluster"]["np2"].iloc[0]
            )

    # store the sig results in df
    sig_df = pd.DataFrame(sig_results)

    if sig_df.shape[0] == 0 or len(sig_cont) == 0:
        return sig_df, post_hocs

    elif len(sig_cont) > 0:
        for ft in sig_cont:
            pt = pg.pairwise_tukey(data=data, dv=ft, between="Cluster")
            pt = pt[pt["p-tukey"] < 0.005]
            pt = pt[["A", "B", "diff", "p-tukey", "hedges"]]
            pt["Feature"] = ft

            post_hocs = pd.concat([post_hocs, pt])

        post_hocs = post_hocs[["Feature", "A", "B", "diff", "p-tukey", "hedges"]]
        post_hocs.rename(
            columns={
                "diff": "Difference Between Means",
                "p-tukey": "p Val",
                "hedges": "Effect Size",
            },
            inplace=True,
        )

        return sig_df, post_hocs
