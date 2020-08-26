import pandas as pd
from eagles.Exploratory import missing, distributions, categories
from eagles.Exploratory.utils import plot_utils as pu
from IPython.display import display

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def get_correlations(
    data: pd.DataFrame = None, cols: list = [], plot: bool = False
) -> pd.DataFrame:
    if len(cols) == 0:
        cols = data.columns

    corr_df = data[cols].corr()

    if plot:
        pu.plot_correlations(data=data, cols=cols)

    return corr_df


def get_base_descriptives(
    data: pd.DataFrame = None, cols: list = [], stats: list = []
) -> pd.DataFrame:

    if len(cols) == 0:
        cols = data.columns

    # filter out object cols
    cols = [col for col in cols if data[col].dtype != "O"]

    if len(stats) == 0:
        stats = ["mean", "median", "std", "min", "max", "skew"]

    stat_df = pd.DataFrame()

    for stat in stats:
        tmp = None
        if stat == "mean":
            tmp = pd.DataFrame(data[cols].mean())
        elif stat == "median":
            tmp = pd.DataFrame(data[cols].median())
        elif stat == "std":
            tmp = pd.DataFrame(data[cols].std())
        elif stat == "min":
            tmp = pd.DataFrame(data[cols].min())
        elif stat == "max":
            tmp = pd.DataFrame(data[cols].max())
        elif stat == "skew":
            tmp = pd.DataFrame(data[cols].skew())
        else:
            print(stat + " not supported")

        stat_df = pd.concat([stat_df, tmp], axis=1)

    stat_df.reset_index(inplace=True)
    stat_df.columns = ["feature"] + stats

    display(stat_df)

    return stat_df


def run_battery(
    data: pd.DataFrame = None,
    categorical_cols: list = [],
    continuous_cols: list = [],
    tests: list = [],
    gen_stats: list = [],
    cap_stats: list = [],
    plot=True,
) -> dict:

    if len(categorical_cols) == 0:
        categorical_cols = [col for col in data.columns if data[col].dtype == "O"]

    if len(continuous_cols) == 0:
        continuous_cols = [col for col in data.columns if data[col].dtype != "O"]

    cols = categorical_cols + continuous_cols

    if len(tests) == 0:
        tests = [
            "info",
            "missing",
            "descriptive",
            "distributions",
            "correlations",
            "category_stats",
        ]

    return_dict = {}

    for test in tests:
        if test == "info":
            n_rows = data[cols].shape[0]
            n_cols = data[cols].shape[1]
            memory_stat = data[cols].memory_usage(index=True).sum()
            total_percent_missing = round(
                (
                    data[cols].isna().sum().sum()
                    / (data[cols].shape[0] * data[cols].shape[1])
                )
                * 100,
                2,
            )
            info_df = pd.DataFrame(
                {
                    "stat": [
                        "n_rows",
                        "n_cols",
                        "total_memory",
                        "total_percent_missing",
                    ],
                    "value": [n_rows, n_cols, memory_stat, total_percent_missing],
                }
            )
            display(info_df)
            return_dict["info"] = info_df
        elif test == "missing":
            msg_df = missing.get_proportion_missing(data=data, cols=cols, plot=plot)
            return_dict["missing"] = msg_df
        elif test == "descriptive":
            stat_df = get_base_descriptives(
                data=data, cols=continuous_cols, stats=gen_stats
            )
            return_dict["descriptives"] = stat_df
        elif test == "distributions":
            dist_df = distributions.find_caps(
                data=data, cols=continuous_cols, stats=cap_stats, plot=plot
            )
            return_dict["distributions"] = dist_df
        elif test == "correlations":
            corr_df = get_correlations(data=data, cols=continuous_cols, plot=plot)
            return_dict["correlations"] = corr_df
        elif test == "category_stats":
            cat_df = categories.get_sample_stats(
                data=data, cols=categorical_cols, plot=plot
            )
            return_dict["categorical_stats"] = cat_df
        else:
            print(test + " not supported")

    return return_dict
