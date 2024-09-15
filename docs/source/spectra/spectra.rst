Spectral Fitting Overview
==========================

Introduction
------------
Spectral fitting is a crucial technique in astrophysics used to analyze the light spectra from various celestial objects. This document provides an overview of the methods and tools used in spectral fitting.

Key Concepts
-------------
- **Spectral Lines**: Features in the spectrum that correspond to specific wavelengths of light emitted or absorbed by elements.
- **Continuum**: The underlying spectrum without any spectral lines.
- **Redshift**: The change in the wavelength of spectral lines due to the motion of the source relative to the observer.

Tools and Libraries
--------------------
Several tools and libraries are commonly used for spectral fitting:

- **Astropy**: A comprehensive library for astronomy-related computations.
- **Specutils**: A library specifically designed for spectral analysis.
- **Sherpa**: A modeling and fitting application for Python.

Example Workflow
-----------------
1. **Data Preparation**: Load and preprocess the spectral data.
2. **Model Selection**: Choose an appropriate model for the spectral lines and continuum.
3. **Fitting**: Use a fitting algorithm to match the model to the data.
4. **Analysis**: Interpret the fitted parameters and their uncertainties.

References
----------
- Astropy Documentation: https://docs.astropy.org/
- Specutils Documentation: https://specutils.readthedocs.io/
- Sherpa Documentation: https://sherpa.readthedocs.io/

Conclusion
----------
Spectral fitting is a powerful tool for understanding the physical properties of celestial objects. By using the right tools and techniques, astronomers can extract valuable information from spectral data.

.. toctree::
    :maxdepth: 1

    spectrum
    spectral_catalogue
    phot_spec_matching