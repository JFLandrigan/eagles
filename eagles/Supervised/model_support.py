from eagles.Supervised import config
import pandas as pd
from IPython.display import display


def print_classifiers() -> None:

    print("Supoorted Classification Models")
    mods = pd.DataFrame(
        {"Abbreviation": config.clf_model_abbreviations, "Model": config.clf_models}
    )
    display(mods)

    print("\nClassification Metrics:")
    print(config.clf_metrics)

    return


def print_regressors() -> None:
    print("Supoorted Regression Models")
    mods = pd.DataFrame(
        {
            "Abbreviation": config.regress_model_abbreviations,
            "Model": config.regress_models,
        }
    )
    display(mods)

    print("\nRegressor Metrics:")
    print(config.regress_metrics)

    return
