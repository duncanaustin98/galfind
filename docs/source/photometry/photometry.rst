Handling Source Photometry
==========================

The ``Photometry`` class is potentially the most fundamental of any in the galfind framework when analyzing photometric imaging surveys. 

This simple class stores:
1. An ``Instrument`` object to keep track of the photometric filters contained in each object, as well as their order.
2. Masked ``flux`` (len(instrument),) array to store the raw photometric fluxes as well as which bands are flagged as unreliable (i.e. masked) for the source.
3. Masked ``flux_errs`` (len(flux), 2) array storing the lower and upper :math:`1~\sigma` flux errors.
4. ``depths`` list or dictionary containing the :math:`5~\sigma` local depths of the source. Normally given in units of AB magnitudes.

It is worth noting that even though they are labelled ``flux`` and ``flux_errs``, these can in principle be input as magnitudes.

The ``Photometry`` class is an abstract base class which is parent to 3 child classes: 
- ``Mock_Photometry``: Stores an additional method to scatter the photometric data, and includes a ``min_pc_err`` attribute to create photometric errors based on given depths.
- ``Photometry_obs``: Contains an ``aper_diam`` attribute storing the aperture size used to generate the photometry as well as a dictionary of ``SED_result`` objects, labelled by their respective SED fitting parameters (see [SED fitting](../sed_fitting/sed_fitting.rst) for more details).
- ``Photometry_rest``: Contains a ``z`` attribute to store the redshift of the source as well as methods to calculate rest-frame photometric properties (e.g. :math:`\beta`, :math:`M_{\mathrm{UV}}`, :math:`\xi_{\mathrm{ion}}`, line EWs, etc)

In many circumstances it is advantageous to instantiate many ``Photometry`` objects at once, for example when reading in a large photometric catalogue. This can be done using the ``Multiple_Photometry`` object. Like ``Photometry``, ``Multiple_Photometry`` is also an abstract base class which is parent to ``Multiple_Mock_Photometry`` and ``Multiple_Photometry_obs``, which contain class methods to load the data in bulk. We do not require a ``Multiple_Photometry_rest`` class since it is not common to store rest frame fluxes in photometric catalogues.

.. toctree::
    :maxdepth: 1

    basic_photometry
    photometry_obs
    photometry_rest
    multiple_photometry
