"""Imaging data, filters, instruments, and PSF handling."""

from .Data import Data, Multiple_Band_Data_Base
from .Filter import Filter, Multiple_Filter
from .Instrument import MIRI, NIRCam
from .PSF import PSF_Base, PSF_Cutout

__all__ = [
    "Data",
    "Filter",
    "MIRI",
    "NIRCam",
    "Multiple_Filter",
    "Multiple_Band_Data_Base",
    "PSF_Base",
    "PSF_Cutout",
    "all_filt_names",
]

def __getattr__(name):
    if name == "all_filt_names":
        from .. import all_filt_names
        return all_filt_names
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
