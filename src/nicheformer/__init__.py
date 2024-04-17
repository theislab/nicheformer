from importlib.metadata import version

from . import data, models

__all__ = ["data", "models"]

__version__ = version("nicheformer")
