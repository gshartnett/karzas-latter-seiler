"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from emp.constants import (
    DEFAULT_A,
    DEFAULT_HOB,
    DEFAULT_NUM_TIME_POINTS,
    DEFAULT_TIME_MAX,
    DEFAULT_Bnorm,
    DEFAULT_Compton_KE,
    DEFAULT_gamma_yield_fraction,
    DEFAULT_pulse_param_a,
    DEFAULT_pulse_param_b,
    DEFAULT_rtol,
    DEFAULT_theta,
    DEFAULT_total_yield_kt,
)
from emp.geometry import Point
from emp.model import EmpModel

# Argument parsing
parser = argparse.ArgumentParser(
    description="Compute the surface EMP intensity using the Karzas-Latter-Seiler model"
)

parser.add_argument(
    "-HOB", default=DEFAULT_HOB, type=float, help="Height of burst [km]"
)

parser.add_argument(
    "-lat_burst", required=True, type=float, help="Latitude of burst point [deg]"
)

parser.add_argument(
    "-lon_burst", required=True, type=float, help="Longitude of burst point [deg]"
)

parser.add_argument(
    "-lat_target", required=True, type=float, help="Latitude of target point [deg]"
)

parser.add_argument(
    "-lon_target", required=True, type=float, help="Longitude of target point [deg]"
)

parser.add_argument(
    "-Compton_KE",
    default=DEFAULT_Compton_KE,
    type=float,
    help="Kinetic energy of Compton electrons [MeV]",
)

parser.add_argument(
    "-total_yield_kt",
    default=DEFAULT_total_yield_kt,
    type=float,
    help="Total weapon yield [kt]",
)

parser.add_argument(
    "-gamma_yield_fraction",
    default=DEFAULT_gamma_yield_fraction,
    type=float,
    help="Fraction of yield corresponding to prompt gamma rays",
)

parser.add_argument(
    "-pulse_param_a",
    default=DEFAULT_pulse_param_a,
    type=float,
    help="Pulse parameter a [ns^(-1)]",
)

parser.add_argument(
    "-pulse_param_b",
    default=DEFAULT_pulse_param_b,
    type=float,
    help="Pulse parameter b [ns^(-1)]",
)

parser.add_argument(
    "-rtol",
    default=DEFAULT_rtol,
    type=float,
    help="Relative tolerance used in the ODE integration",
)

parser.add_argument(
    "-method",
    default="Radau",
    type=str,
    help="Integration method to use (see scipy.integrate.solve_ivp for options)",
)

parser.add_argument(
    "-magnetic_field_model",
    default="dipole",
    type=str,
    help="Magnetic field model to use (dipole or IGRF)",
)

parser.add_argument(
    "-time_max",
    default=DEFAULT_TIME_MAX,
    type=float,
    help="Maximum integration time [ns]",
)

parser.add_argument(
    "-num_time_points",
    default=DEFAULT_NUM_TIME_POINTS,
    type=int,
    help="Number of time points to evaluate the solution at",
)

args = vars(parser.parse_args())

# Construct the burst and target points
burst_point = Point.from_gps_coordinates(
    args["lat_burst"], args["lon_burst"], altitude_m=args["HOB"]
)
target_point = Point.from_gps_coordinates(
    args["lat_target"], args["lon_target"], altitude_m=0.0
)

# Define the model
model = EmpModel(
    burst_point=burst_point,
    target_point=target_point,
    total_yield_kt=args["total_yield_kt"],
    gamma_yield_fraction=args["gamma_yield_fraction"],
    Compton_KE=args["Compton_KE"],
    pulse_param_a=args["pulse_param_a"],
    pulse_param_b=args["pulse_param_b"],
    rtol=args["rtol"],
    method=args["method"],
    magnetic_field_model=args["magnetic_field_model"],
)

# Print out param values
print("\nRunning with parameters\n--------------------")
for key, value in model.__dict__.items():
    print(key, "=", value)
print("\n")

# Perform the integration
time_points = np.linspace(0, args["time_max"], args["num_time_points"])
result = model.run(time_points)

# Create data and figure directories
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("figures"):
    os.makedirs("figures")

# Save the result
result.save(filepath="data/emp_solution.json")

# Plot the result
fig, ax = plt.subplots(1, figsize=(7, 5))
ax.plot(
    result.time_points,
    result.E_norm_at_ground,
    "-",
    color="k",
    linewidth=1.5,
    markersize=2,
)
ax.set_xlabel(r"$\tau$ [ns]")
ax.set_ylabel(r"E [V/m]")
plt.minorticks_on()
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.grid(alpha=0.5)
plt.title("Surface EMP Intensity")
plt.savefig("figures/emp_intensity.png", bbox_inches="tight", dpi=600)
plt.show()
