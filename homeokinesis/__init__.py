"""Homeokinesis: the SOX controller from The Playful Machine (Der and Martius, 2012).

This folder is the package `homeokinesis`, whether installed with pip (see the
README) or used with the repository root on sys.path:

    from homeokinesis import HomeokinesisNumpy   # numpy version
    from homeokinesis import HomeokinesisTorch   # torch version, loaded on first use

The Python code is in python/, the C++ code in cpp/. The torch version needs
torch, and its module also imports gymnasium and Isaac Lab, so it is only
imported when HomeokinesisTorch is asked for.
"""
from .python.homeokinesis_numpy import Homeokinesis as HomeokinesisNumpy

__all__ = ["HomeokinesisNumpy", "HomeokinesisTorch"]


def __getattr__(name):
    if name == "HomeokinesisTorch":
        from .python.homeokinesis_torch import Homeokinesis
        return Homeokinesis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
