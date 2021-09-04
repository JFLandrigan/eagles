from eagles import config
import os
import pickle
import time
import pandas as pd


def construct_save_path(fl_path=None, fl_name=None, model_name=None, save_dir=False):
    timestr = time.strftime("%Y%m%d-%H%M")

    # if no file path and not creating savedir then just create a data directory in the path of utils file
    if fl_path is None and not save_dir:

        fl_path = (
            os.path.abspath(os.path.dirname(__file__))
            + config.ext_char
            + "data"
            + config.ext_char
        )

    # if fl path is none and creating save dir for mult objects then create data_modelname directory in utils
    elif fl_path is None and save_dir:
        fl_path = (
            os.path.abspath(os.path.dirname(__file__))
            + config.ext_char
            + "data_"
            + model_name
            + "_"
            + timestr
            + config.ext_char
        )

    # if fl path passed in and creating a save dir for mult items then tack on model name and time to end of file path
    elif fl_path and save_dir:
        tmp_fl_path = (
            fl_path + config.ext_char + model_name + "_" + timestr + config.ext_char
        )
        fl_path = tmp_fl_path

    # if file name is none then create general log name using model name and time stamp
    if fl_name is None:
        fl_name = model_name + "_" + timestr + ".txt"

    if not os.path.exists(fl_path):
        os.mkdir(fl_path)

    return [fl_path, fl_name, timestr]


def build_log_data(
    mod,
    features,
    metric_dictionary,
    random_seed,
    cf=None,
    cr=None,
    bt=None,
    ft_imp_df=None,
    test_params,
    tune_metric,
    note=None,
):
    log_data = {
            "features": features,
            "random_seed": random_seed,
            "metrics": metric_dictionary,
            "params": list(),
        }

    if type(mod).__name__ == "Pipeline":
        log_data["params"].append(
            [
                type(mod.named_steps["clf"]).__name__,
                str(mod.named_steps["clf"].get_params()),
            ]
        )

    elif "Voting" in type(mod).__name__:
        log_data["params"].append(
            str([type(mod).__name__, str(mod.get_params()["weights"])])
        )
        for c in mod.estimators_:
            if type(c).__name__ == "Pipeline":
                log_data["params"].append(
                    [
                        type(c.named_steps["clf"]).__name__,
                        str(c.named_steps["clf"].get_params()),
                    ]
                )
            else:
                log_data["params"].append(
                    [type(c).__name__, str(c.get_params()),]
                )
    else:
        log_data["params"].append([type(mod).__name__, str(mod.get_params())])

    if cf:
        log_data["cf"] = cf
    if cr:
        log_data["cr"] = cr

    if type(mod).__name__ == "Pipeline":
        log_data["model"] = type(mod).__name__
        pipe_steps = "Pipe steps: "
        for k in mod.named_steps.keys():
            pipe_steps = pipe_steps + type(mod.named_steps[k]).__name__ + " "
        log_data["pipe_steps"] = pipe_steps
    else:
        log_data["model"] = type(mod).__name__

    if bt:
        log_data["bin_table"] = bt

    if ft_imp_df:
        log_data["ft_imp_df"] = ft_imp_df

    if test_params:
        log_data["test_params"] = test_params
    if tune_metric:
        log_data["tune_metric"] = tune_metric

    if note:
        log_data["note"] = note

    return

def log_results(fl_name=None, fl_path=None, log_data=None, tune_test=True):

    fl_path, fl_name, timestr = construct_save_path(
        fl_name=fl_name, fl_path=fl_path, model_name=log_data["model"]
    )
    save_path = fl_path + fl_name

    f = open(save_path, "w")

    if "note" in log_data.keys():
        f.write(str(log_data["note"]) + " \n \n")

    if log_data["model"] == "Pipeline":
        f.write("Pipepline" + "\n")
        f.write(log_data["pipe_steps"] + "\n \n")
    else:
        f.write("Model testing: " + log_data["model"] + "\n \n")

    if tune_test:
        f.write("params tested: " + str(log_data["test_params"]) + "\n \n")
        f.write("tune metric: " + log_data["tune_metric"] + "\n \n")

    f.write("Features included: " + "\n" + str(log_data["features"]) + "\n \n")
    f.write("Random Seed Value: " + str(log_data["random_seed"]) + " \n \n")

    f.write("Params of model: " + "\n")
    for pr in log_data["params"]:
        f.write(str(pr) + "\n\n")
    f.write("\n\n")

    tmp_metric_dict = log_data["metrics"]
    for metric in tmp_metric_dict.keys():
        if "_scores" in metric:
            f.write(metric + " scores: " + str(tmp_metric_dict[metric]) + " \n")
            f.write(metric + " mean: " + str(tmp_metric_dict[metric].mean()) + " \n")
            f.write(
                metric
                + " standard deviation: "
                + str(tmp_metric_dict[metric].std())
                + " \n"
            )

    f.write(" \n")

    f.write("Final cv train test split results \n")
    for metric in tmp_metric_dict.keys():
        if "_scores" in metric:
            f.write(metric + " score: " + str(tmp_metric_dict[metric][-1]) + "\n")

    f.write(" \n \n")

    if "cf" in log_data.keys():
        f.write("Raw Confusion Matrix \n")
        f.write(str(log_data["cf"]) + " \n \n")
    if "cr" in log_data.keys():
        f.write("Classification Report \n")
        f.write(log_data["cr"] + " \n \n")

    if "bin_table" in log_data.keys():
        f.write(str(log_data["bin_table"]) + " \n \n")

    if "ft_imp_df" in log_data.keys():
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", 250)
        f.write(log_data["ft_imp_df"].to_string())

    f.close()

    return


def pickle_data(data=None, fl_path=None, fl_name=None, data_type=None):

    fname = data_type + "_" + fl_name
    fname = fname.replace("txt", "pkl")

    # save out the data
    with open(fl_path + fname, "wb") as handle:
        pickle.dump(data, handle)

    return



