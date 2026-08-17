from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .grid import Grid_2D


class Contamination(Grid_2D):
    """Grid of contamination fraction as a function of two properties.

    Placeholder subclass of `Grid_2D` intended to characterise the
    fraction of selected sources that are spurious/contaminant (e.g.
    low-redshift interlopers) as a function of two derived properties
    (such as redshift and absolute UV magnitude), by comparing a
    simulated source catalogue to a selected source catalogue. Currently
    unimplemented beyond the inherited `Grid_2D` behaviour.
    """

    pass
    # @staticmethod
    # def _make_grid(
    #     sim_cat: Catalogue,
    #     select_cat: Catalogue,
    # ):
    #     # make grid from the catalogues
    #     pass
