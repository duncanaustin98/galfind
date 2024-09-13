Usage
=====

.. _installation:

Installation
------------

To use Galfind, first install it using pip:

.. code-block:: bash

    pip install galfind

Creating an Instrument
----------------------

Here is a basic example of how to create an instrument using `galfind` in your Python code:

.. code-block:: python

    from galfind import ACS_WFC, NIRCam

    # Create instruments
    acs_wfc = ACS_WFC()
    nircam = NIRCam()

    # Combine instruments
    instrument = acs_wfc + nircam

    # Print the instrument
    print(instrument)

This example demonstrates how to create `ACS_WFC` and `NIRCam` instruments and combine them into a single instrument.
