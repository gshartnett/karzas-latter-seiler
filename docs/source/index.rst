Karzas-Latter-Seiler EMP Model Documentation
============================================

A Python implementation of the Karzas-Latter model for high-altitude
electromagnetic pulse (EMP) simulation.

.. note::
   This model uses crude approximations and is intended for educational
   and policy analysis purposes. Results are accurate to within an order
   of magnitude.

Quick Start
-----------

.. code-block:: python

   import os
   import pickle
   import numpy as np
   import matplotlib.pyplot as plt

   from emp.constants import (
       DEFAULT_A,
       DEFAULT_HOB,
       DEFAULT_Bnorm,
       DEFAULT_Compton_KE,
       DEFAULT_gamma_yield_fraction,
       DEFAULT_pulse_param_a,
       DEFAULT_pulse_param_b,
       DEFAULT_rtol,
       DEFAULT_theta,
       DEFAULT_total_yield_kt,
   )
   from emp.model import EmpModel

   # Define model parameters (can be customized)
   model = EmpModel(
       HOB=DEFAULT_HOB,
       Compton_KE=DEFAULT_Compton_KE,
       total_yield_kt=DEFAULT_total_yield_kt,
       gamma_yield_fraction=DEFAULT_gamma_yield_fraction,
       Bnorm=DEFAULT_Bnorm,
       A=DEFAULT_A,
       theta=DEFAULT_theta,
       pulse_param_a=DEFAULT_pulse_param_a,
       pulse_param_b=DEFAULT_pulse_param_b,
       rtol=DEFAULT_rtol,
   )

   # Perform the integration
   sol = model.solver(np.linspace(0, 50, 200))

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   user_guide/overview

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Info:

   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
