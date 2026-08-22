Morphology Fitting
===================

`galfind` provides parametric morphological fitting of galaxy cutouts
through a common `Morphology_Fitter`/`Morphology_Result` interface.
`Galfit_Fitter` wraps the GALFIT code (box constraints written to a
text input file); `PySersic_Fitter` wraps `pysersic`, a JAX/numpyro
Bayesian Sersic-fitting code, with genuine probabilistic priors instead
of GALFIT-style constraints.

.. note::

   `PySersic_Fitter` requires the optional `pysersic` dependency group
   (``pip install galfind[pysersic]``), which pulls in `jax` and
   `numpyro`.

.. toctree::
    :maxdepth: 1

    PySersic
    Galfit
