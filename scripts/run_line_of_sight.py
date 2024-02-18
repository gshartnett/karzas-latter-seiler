"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt

from emp.constants import *
from emp.model import EMPMODEL

# argument parsing
parser = argparse.ArgumentParser(
    description="Compute the surface EMP intensity using the Karzas-Latter-Seiler model"
)

parser.add_argument(
    "-HOB", default=DEFAULT_HOB, type=float, help="Height of burst [km]"
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
    "-Bnorm",
    default=DEFAULT_Bnorm,
    type=float,
    help="Local value of the geomagnetic field strength norm [T]",
)

parser.add_argument(
    "-theta",
    default=DEFAULT_theta,
    type=float,
    help="Angle between the line-of-sight vector and the geomagnetic field",
)

parser.add_argument(
    "-A",
    default=DEFAULT_A,
    type=float,
    help="Angle between the line-of-sight vector and the vector normal to the surface of the Earth",
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

args = vars(parser.parse_args())

# define the model
model = EMPMODEL(
    HOB=args["HOB"],
    Compton_KE=args["Compton_KE"],
    total_yield_kt=args["total_yield_kt"],
    gamma_yield_fraction=args["gamma_yield_fraction"],
    Bnorm=args["Bnorm"],
    A=args["A"],
    theta=args["theta"],
    pulse_param_a=args["pulse_param_a"],
    pulse_param_b=args["pulse_param_b"],
    rtol=args["rtol"],
)

# print out param values
print("\nRunning with parameters\n--------------------")
for key, value in model.__dict__.items():
    print(key, "=", value)
print("\n")

# perform the integration
sol = model.solver(np.linspace(0, 50, 200))

# create data and figure directories
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("figures"):
    os.makedirs("figures")

# save the result
with open("data/emp_solution.pkl", "wb") as f:
    pickle.dump(sol, f)

# plot the result
fig, ax = plt.subplots(1, figsize=(7, 5))
ax.plot(
    sol["tlist"],
    sol["E_norm_at_ground"],
    "-",
    color="k",
    linewidth=1.5,
    markersize=2,
)
ax.set_xlabel(r"$\tau$ \ [ns]")
ax.set_ylabel(r"E \ [V/m]")
plt.minorticks_on()
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.grid(alpha=0.5)
plt.title("Surface EMP Intensity")
plt.savefig("figures/emp_intensity.png", bbox_inches="tight", dpi=600)
plt.show()
