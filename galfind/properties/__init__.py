"""Property extraction and calculation utilities."""

from .Property_calculator import Redshift_Extractor, Property_Extractor, SED_Property_Calculator
from .Rest_frame_properties import (
    Rest_Frame_Property_Calculator,
    UV_Beta_Calculator,
    UV_Dust_Attenuation_Calculator,
    Fesc_From_Beta_Calculator,
    mUV_Calculator,
    MUV_Calculator,
    LUV_Calculator,
    SFR_UV_Calculator,
    Optical_Continuum_Calculator,
    Optical_Line_EW_Calculator,
    Dust_Attenuation_From_UV_Calculator,
    Optical_Line_Flux_Calculator,
    Optical_Line_Luminosity_Calculator,
    Ndot_Ion_Calculator,
    Xi_Ion_Calculator,
    SFR_Halpha_Calculator,
    Lya_Break_Strength_Calculator,
)
from ..utils import useful_funcs_austind

# Import all modules containing calculator subclasses to ensure they're loaded
from . import Property_calculator, Rest_frame_properties

# Dynamically import all Property_Extractor and Rest_Frame_Property_Calculator subclasses
_extractors = list(useful_funcs_austind.all_subclasses(Property_Extractor))
_rest_frame_calcs = list(useful_funcs_austind.all_subclasses(Rest_Frame_Property_Calculator))

for cls in _extractors + _rest_frame_calcs:
    globals()[cls.__name__] = cls

__all__ = (
    ["Redshift_Extractor", "Property_Extractor", "SED_Property_Calculator", "Rest_Frame_Property_Calculator"]
    + [cls.__name__ for cls in _extractors]
    + [cls.__name__ for cls in _rest_frame_calcs]
)
