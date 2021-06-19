import os
from sys import platform
import pandas as pd
from eagles import config


def construct_path():
    file_path = os.path.abspath(os.path.dirname(__file__)) + config.ext_char
    return file_path


def list_datasets() -> None:
    """
    List out file names and quick dataset stats
    :return: None
    """

    print("Datasets Include\n")
    path = construct_path()
    files = [fl for fl in os.listdir(path) if "csv" in fl]
    for fl in files:
        print(f"***** {fl} ******\n")
        tmp = load_data(data_set=fl.replace(".csv", ""))
        print(tmp.info())
        print("-----------------------------------------\n")

    return None


def load_data(data_set: str = None):
    """
    Function to load in datasets.
    :param data_set: Strings default None. Expects iris, wines, stack-overflow-data or titanic
    :return: pandas dataframe with relevant data
    """
    fl_path = construct_path()
    return pd.read_csv(fl_path + data_set + ".csv")
