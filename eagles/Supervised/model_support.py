from eagles.Supervised import config


def print_classifiers() -> None:
    for i in range(len(config.clf_model_abbreviations)):
        print(
            "Abbreviattion: "
            + config.clf_model_abbreviations[i]
            + ", Model: "
            + config.clf_models[i]
        )
    return


def print_regressors() -> None:
    for i in range(len(config.regress_model_abbreviations)):
        print(
            "Abbreviattion: "
            + config.regress_model_abbreviations[i]
            + ", Model: "
            + config.regress_models[i]
        )
    return
