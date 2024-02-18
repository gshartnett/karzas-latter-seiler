"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

# imports
import argparse
import os
import pickle

import numpy as np

import emp.geometry as geometry
from emp.constants import *
from emp.region_scan import (
    contour_plot,
    data_dic_to_xyz,
    region_scan,
)

# argument parsing
parser = argparse.ArgumentParser(
    description="Compute the surface EMP intensity using the Karzas-Latter-Seiler model"
)

parser.add_argument(
    "-phi_B_g",
    default=39.05 * np.pi / 180,
    type=float,
    help="Burst point latitude [radians]",
)

parser.add_argument(
    "-lambd_B_g",
    default=-95.675 * np.pi / 180,
    type=float,
    help="Burst point longitude [radians]",
)

parser.add_argument(
    "-N_pts_phi", default=50, type=int, help="Number of latitude grid points"
)

parser.add_argument(
    "-N_pts_lambd", default=50, type=int, help="Number of longitude grid points"
)

parser.add_argument(
    "-time_max",
    default=100.0,
    type=float,
    help="Total longitude angular spread (degrees)",
)

parser.add_argument(
    "-N_pts_time",
    default=50,
    type=int,
    help="Total longitude angular spread (degrees)",
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
    "-save_str",
    default="",
    type=str,
    help="String used to save results from different data runs",
)

parser.add_argument(
    "-b_field_type",
    default="dipole",
    type=str,
    help="Geomagnetic field model (dipole or igrf).",
)

args = vars(parser.parse_args())
save_str = args.pop("save_str")

# instantiate the burst point, add it to args, and remove the angles
Burst_Point = geometry.Point(
    EARTH_RADIUS + args["HOB"],
    args["phi_B_g"],
    args["lambd_B_g"],
    coordsys="lat/long geo",
)
args["Burst_Point"] = Burst_Point
args.pop("phi_B_g")
args.pop("lambd_B_g")

# print out param values
print("\nModel Parameters\n--------------------")
for key, value in args.items():
    print(key, "=", value)
print("\n")

# perform the region scan
data_dic = region_scan(**args)

# create data and figure directories
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("figures"):
    os.makedirs("figures")

# save the result
if save_str != "":
    save_str = "_" + save_str
with open("data/region_scan" + save_str + ".pkl", "wb") as f:
    pickle.dump(data_dic, f)

x, y, z = data_dic_to_xyz(data_dic)

contourf, _ = contour_plot(
    x, y, z, Burst_Point, save_path="figures/region_scan", grid=False
)