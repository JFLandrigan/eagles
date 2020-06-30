import os
from sys import platform
import pandas as pd


def construct_path():
    if platform == "linux" or platform == "linux2":
        # linux
        ext_char = "/"
    elif platform == "darwin":
        # OS X
        ext_char = "/"
    elif platform == "win32":
        # Windows...
        ext_char = "\\"

    file_path = os.path.abspath(os.path.dirname(__file__)) + ext_char

    return file_path


def load_iris():
    file_path = construct_path()
    return pd.read_csv(file_path + "iris.csv")


def load_stack_overflow_dat():
    file_path = construct_path()
    return pd.read_csv(file_path + "stack-overflow-data.csv")
