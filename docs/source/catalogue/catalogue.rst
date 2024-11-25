Catalogue Object
================

This `Catalogue` class forms the basis of survey analysis within galfind. It can be instantiated in two different ways:

1. From a .fits photometric catalogue given a `Catalogue_Creator` object
2. From an input galfind `Data` object

The most major component stored in a `Catalogue` object is a list of galfind `Galaxy` objects, from which galaxy properties can be extracted and analysed.

.. note::

   Plotting and number density function documentation not complete!

.. toctree::
    :maxdepth: 1

    load_catalogue
    plotting
    number_density_function