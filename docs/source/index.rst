Karzas-Latter-Seiler EMP Model Documentation
============================================

A Python implementation of the Karzas-Latter model for high-altitude
electromagnetic pulse (EMP) simulation.

Quick Start
-----------

Simulate the effects of a high-altitude EMP.
In this case, the detonation occurs over New York City (40.7128° N, 74.0060° W)
and the target is Washington, D.C. (38.9072° N, 77.0369° W).

.. code-block:: python

python scripts/run_line_of_sight.py -lat_burst=40.7128 -lon_burst=-74.0060 -lat_target=38.9072 -lon_target=-77.0369

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Info:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
