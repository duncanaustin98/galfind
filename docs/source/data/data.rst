Galfind Data Object
===================

The `Data` object in galfind is probably the most fundamental class galfind offers.
This class stores the paths/extensions to the imaging and includes functionality to PSF homogenize, mask, segment, perform forced photometry, run imaging depths, and produce high quality photometric catalogues.
As of now, the PSF homogneization is not available and is under active development.

.. note::

   PSF implementation and subsequent image homogenization not complete!

.. toctree::
    :maxdepth: 1

    data_intro
    PSF_homogenization
    source_extraction
    masking
    running_depths
    cataloguing_the_data