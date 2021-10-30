import os
from sys import platform
import pandas as pd
from eagles import config


def construct_path():
    file_path = os.path.abspath(os.path.dirname(__file__)) + config.ext_char
    return file_path


def load_iris():
    file_path = construct_path()
    return pd.read_csv(file_path + "iris.csv")


def load_wines():
    file_path = construct_path()
    return pd.read_csv(file_path + "wines.csv")


def load_stack_overflow_dat():
    file_path = construct_path()
    return pd.read_csv(file_path + "stack-overflow-data.csv")
