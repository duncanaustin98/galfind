
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Catalogue

from .grid import Grid_2D


class Completeness(Grid_2D):
    
    @staticmethod
    def _make_grid(
        sim_cat: Catalogue,
        select_cat: Catalogue,
    ):
        # make grid from the catalogues
        pass