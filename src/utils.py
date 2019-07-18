import sys
from types import ModuleType, FunctionType
from gc import get_referents

from IPython.display import Markdown, display

import pandas as pd

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


def get_all_dataframe(ds):
    """
    Return X_train and X_test concatenated
    """
    return pd.concat([ds.X_train, ds.X_test], axis = 0, sort = False)


def printmd(string):
    """
    Print Markdown
    From https://stackoverflow.com/questions/23271575/printing-bold-colored-etc-text-in-ipython-qtconsole
    """
    display(Markdown(string))
