def _print_param_tuning_results(mod=None, X=None) -> None:
    """
    Utility function for printing results of paramter tuning

    Args:
        mod ([type], optional): Model object. Defaults to None.
        X ([type], optional): Pandas data with feature columns for training. Defaults to None.
    """

    print("Parameters of the best model: \n")
    if type(mod).__name__ == "Pipeline":
        print(type(mod.named_steps["clf"]).__name__ + " Parameters")
        print(str(mod.named_steps["clf"].get_params()) + "\n")

    elif "Voting" in type(mod).__name__:
        print(
            type(mod).__name__ + " weights: " + str(mod.get_params()["weights"]) + "\n"
        )
        for c in mod.estimators_:
            if type(c).__name__ == "Pipeline":
                print(type(c.named_steps["clf"]).__name__ + " Parameters")
                print(str(c.named_steps["clf"].get_params()) + "\n")
            else:
                print(type(c).__name__ + " Parameters")
                print(str(c.get_params()) + "\n")
    else:
        print(type(mod).__name__ + " Parameters")
        print(str(mod.get_params()) + "\n")

    return
