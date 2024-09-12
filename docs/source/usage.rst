Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install galfind

Creating an instrument
----------------

To make a blank NIRCam() Instrument object
you can use the ``galfind.Instrument.NIRCam()`` function:

.. autofunction:: galfind.Instrument.NIRCam

The ``excl_bands`` parameter should be ``[]`` for now. Otherwise, :py:func:`galfind.Instrument.NIRCam`
will raise an exception.

#.. autoexception:: lumache.InvalidKindError

For example:

>>> import galfind
>>> galfind.Instrument.NIRCam()
["F070W","F090W","F115W","F140M","F150W","F162M","F164N","F150W2","F182M","F187N","F200W","F210M","F212N","F250M","F277W","F300M","F323N","F322W2","F335M","F356W","F360M","F405N","F410M","F430M","F444W","F460M","F466N","F470N","F480M"]
