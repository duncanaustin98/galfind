"""Selection-function utilities for GALFIND.

This subpackage provides tools for characterising how a photometric
selection responds to simulated/scattered source catalogues, namely:

- `Grid_2D` (in `grid.py`): a generic 2D histogram-ratio grid (e.g.
  number selected / number simulated) built from pairs of "simulated"
  and "selected" catalogues, with interpolation and plotting helpers.
- `Completeness`, `Catalogue_Completeness`, `Completeness_2D` (in
  `completeness.py`): completeness estimates (the fraction of simulated
  sources recovered by a selection) as a function of a single derived
  property, or per-galaxy/per-catalogue lookups combining multiple such
  completeness curves.
- `Contamination` (in `contamination.py`): a `Grid_2D` subclass intended
  for characterising contamination fractions of a selection (currently a
  placeholder).

These are typically used for luminosity-function and sample-selection
work, where the true (simulated) and recovered (selected) source
distributions must be compared as a function of redshift, magnitude, or
other derived quantities.
"""

from .grid import Grid_2D
from .completeness import Completeness, Catalogue_Completeness, Completeness_2D
from .contamination import Contamination
from .Selector import (
    Redshift_Bin_Selector,
    EPOCHS_Selector,
    Redwards_Lya_Detect_Selector,
    Stacked_Blue_Lya_Non_Detect_Selector,
    Selector,
    Depth_Region_Selector,
    Brown_Dwarf_Selector,
    Mask_Selector,
    EPOCHS_unmasked_criteria,
)

__all__ = [
    "Grid_2D",
    "Completeness",
    "Catalogue_Completeness",
    "Completeness_2D",
    "Contamination",
    "Redshift_Bin_Selector",
    "EPOCHS_Selector",
    "Redwards_Lya_Detect_Selector",
    "Selector",
    "Depth_Region_Selector",
    "Brown_Dwarf_Selector",
    "Mask_Selector",
    "EPOCHS_unmasked_criteria",
    "Stacked_Blue_Lya_Non_Detect_Selector",
]
