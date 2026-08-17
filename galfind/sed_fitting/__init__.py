"""SED fitting wrappers for external photometric redshift and SED-fitting codes.

Provides unified GALFIND interfaces to multiple external SED fitting codes:
- EAZY: Photo-z and SED fitting
- LePhare: Photo-z and SED fitting
- Bagpipes: Bayesian SED fitting
- Brown_Dwarf_Fitter: Template fitter for low-mass stars and brown dwarfs

All follow the SED_code base class interface for fitting Galaxy/Catalogue objects
and producing SED_result outputs with optional PDF storage.
"""

from .SED_codes import SED_code
from .SED_result import SED_result, Galaxy_SED_results, Catalogue_SED_results
from .LePhare import LePhare
from .EAZY import EAZY
from .Bagpipes import Bagpipes
from .Brown_Dwarf_Fitter import Template_Fitter, Brown_Dwarf_Fitter

__all__ = [
    "SED_code",
    "SED_result",
    "Galaxy_SED_results",
    "Catalogue_SED_results",
    "LePhare",
    "EAZY",
    "Bagpipes",
    "Template_Fitter",
    "Brown_Dwarf_Fitter",
]
