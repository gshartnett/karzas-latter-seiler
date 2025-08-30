"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

import numpy as np

# constants of nature
SCALE_HEIGHT = 7  # atmospheric scale height in km (the 7km comes from Seiler pg. 24, paragraph 1, I was using 10.4 previously)
AIR_DENSITY_AT_SEA_LEVEL = 1.293  # air density in kg/m^3
ELECTRON_MASS = 0.511  # electron rest mass in MeV
SPEED_OF_LIGHT = 3 * 1e8  # speed of light in m/s
MEAN_FREE_PATH_AT_SEA_LEVEL = 0.3  # atmospheric mean free path of light at sea-level in km (value taken from KL paper)
ELECTRON_CHARGE = 1.602 * 1e-19  # electric charge in C
VACUUM_PERMEABILITY = 1.257 * 1e-6  # vacuum permeability in H/m
MEV_TO_KG = 1.79 * 1e-30  # conversion factor from MeV/c^2 to kg
KT_TO_MEV = 2.611e25  # conversion factor from kt to Mev
EARTH_RADIUS = 6378  # Earth's radius in km
ABSORPTION_LAYER_UPPER = (
    50  # altitude in km of the upper boundary of the absorption layer
)
ABSORPTION_LAYER_LOWER = (
    20  # altitude in km of the upper boundary of the absorption layer
)
PHI_MAGNP = 86.294 * np.pi / 180  # latitude of magnetic North Pole (radians)
LAMBDA_MAGNP = 151.948 * np.pi / 180  # longitude of magnetic North Pole (radians)
B0 = 3.12 * 1e-5  # proportionality constant for the dipole geomagnetic field in Tesla

# default parameters
DEFAULT_total_yield_kt = 5.0  # total yield in kilotons
DEFAULT_gamma_yield_fraction = (
    0.05  # fraction of yield deposited in prompt Gamma radiation
)
DEFAULT_Compton_KE = 1.28  # kinetic energy of Compton electron in MeV
DEFAULT_HOB = 100.0  # heigh of burst in km
DEFAULT_Bnorm = B0  # geomagnetic field strength in Teslas
DEFAULT_theta = np.pi / 2  # angle between line of sight vector and magnetic field
DEFAULT_A = (
    0  # angle between radial ray from burst point to target and normal in radians
)
DEFAULT_pulse_param_a = 1e7 * 1e-9  # pulse parameter in 1/ns
DEFAULT_pulse_param_b = 3.7 * 1e8 * 1e-9  # pulse parameter in 1/ns
DEFAULT_rtol = 1e-4  # relative tolerance for ODE integration
DEFAULT_TIME_MAX = 100.0  # max time to integrate to in ns
DEFAULT_NUM_TIME_POINTS = 300  # number of time points to compute
