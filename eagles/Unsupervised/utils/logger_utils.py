import os
import time
import pandas as pd
from eagles import config


def construct_save_dir(fl_path=None, fl_name=None, model_name=None):

    # construct the save path to get the general fl path and fl name
    fl_path, fl_name, timestr = construct_save_path(
        fl_path=fl_path, fl_name=fl_name, model_name=model_name
    )

    # if no timestamp in the model name then add a time stamp for creating the data dir
    if ~timestr in fl_name:
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


def log_results(fl_name=None, fl_path=None, log_data=None):

    fl_path, fl_name, timestr = construct_save_path(
        fl_name=fl_name, fl_path=fl_path, model_name=log_data["method"]
    )
    save_path = fl_path + fl_name

    f = open(save_path, "w")

    if "note" in log_data.keys():
        f.write(str(log_data["note"]) + " \n \n")

    if log_data["method"] == "Pipeline":
        f.write("Pipepline" + "\n")
        f.write(log_data["pipe_steps"] + "\n \n")
    else:
        f.write("Model testing: " + log_data["method"] + "\n \n")

    f.write("Features included: " + "\n" + str(log_data["features"]) + "\n \n")
    f.write("Params of model: " + str(log_data["params"]) + " \n \n")

    f.write("Silhouette Score: " + str(log_data["Silhouette Score"]) + "\n")
    if "WSS" in log_data.keys():
        f.write("WSS Total: " + str(log_data["WSS"]) + "\n")

    f.write("Number of Observations per Cluster \n")
    f.write(str(log_data["data"]["Cluster"].value_counts()) + "\n\n")

    f.write("Base Cluster Stats \n")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 250)
    f.write(log_data["base_cluster_stats"].T.to_string())
    f.write("\n\n")

    f.close()

    return
