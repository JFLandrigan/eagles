import os
from sys import platform

if platform == "linux" or platform == "linux2":
    # linux
    ext_char = "/"
elif platform == "darwin":
    # OS X
    ext_char = "/"
elif platform == "win32":
    # Windows...
    ext_char = "\\"
