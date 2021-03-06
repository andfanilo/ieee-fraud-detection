import os
import random
import sys
from gc import get_referents
from types import FunctionType, ModuleType

import numpy as np
import pandas as pd
from colorama import init
from IPython.display import Markdown, display
from path import Path
from termcolor import colored

init()


def print_colored_green(line):
    print(colored(line, "green"))


def get_root_dir():
    # Return Project Root
    # Trick from https://stackoverflow.com/a/25389715 to handle root project
    return Path(__file__).parent.parent


def getsize(obj):
    """sum size of object & members."""
    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    BLACKLIST = type, ModuleType, FunctionType

    if isinstance(obj, BLACKLIST):
        raise TypeError("getsize() does not take argument of type: " + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


def printmd(string):
    """
    Print Markdown
    From https://stackoverflow.com/questions/23271575/printing-bold-colored-etc-text-in-ipython-qtconsole
    """
    display(Markdown(string))


def seed_everything(seed=0):
    """seed to make all processes deterministic
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
