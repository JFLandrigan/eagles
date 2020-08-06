from eagles import config
import os
import pickle
import time
import pandas as pd


def construct_save_dir(fl_path=None, fl_name=None, model_name=None):

    # construct the save path to get the general fl path and fl name
    fl_path, fl_name, timestr = construct_save_path(
        fl_path=fl_path, fl_name=fl_name, model_name=model_name
    )

    # if no timestamp in the model name then add a time stamp for creating the data dir
    if timestr not in fl_name:
        timestr = time.strftime("%Y%m%d-%H%M")
        # strip .txt from file name so that can create a dir to save log, data and model to
        tmp_fl_name = fl_name.replace(".txt", "")
        dir_name = fl_path + tmp_fl_name + "_" + timestr
    else:
        tmp_fl_name = fl_name.replace(".txt", "")
        dir_name = fl_path + tmp_fl_name

    os.mkdir(dir_name)

    return [dir_name, fl_name]


def construct_save_path(fl_path=None, fl_name=None, model_name=None):
    # check the top level save path, if none then create a data dir where the tune files exist
    if fl_path is None:

        data_dir = (
            os.path.abspath(os.path.dirname(__file__))
            + config.ext_char
            + "data"
            + config.ext_char
        )
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        fl_path = data_dir

    # if file name is none then create general log name using model name and time stamp
    timestr = time.strftime("%Y%m%d-%H%M")
    if fl_name is None:
        fl_name = model_name + "_" + timestr + ".txt"

    return [fl_path, fl_name, timestr]


def log_results(fl_name=None, fl_path=None, log_data=None, tune_test=True):

    fl_path, fl_name, timestr = construct_save_path(
        fl_name=fl_name, fl_path=fl_path, model_name=log_data["model"]
    )
    save_path = fl_path + fl_name

    print("File path for data log: " + save_path)

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
        f.write(pr + " : " + str(log_data["params"][pr]) + "\n")
    f.write("\n\n")

    # f.write("Params of model: " + str(log_data["params"]) + " \n \n")

    tmp_metric_dict = log_data["metrics"]
    for metric in tmp_metric_dict.keys():
        if "_scores" in metric:
            f.write(metric + " scores: " + str(tmp_metric_dict[metric]) + " \n")
            f.write(metric + "mean: " + str(tmp_metric_dict[metric].mean()) + " \n")
            f.write(
                metric
                + "standard deviation: "
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
        f.write(str(log_data["cf"]) + " \n \n")
    if "cr" in log_data.keys():
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

    # save out the data
    with open(fl_path + data_type + '_' + fl_name, "wb") as handle:
        pickle.dump(data, handle)

    return
