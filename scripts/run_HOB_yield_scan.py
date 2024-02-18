"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

# imports
import argparse
import os
import pickle

from emp.HOB_yield_scan import (
    HOB_yield_scan,
    contour_plot,
    data_dic_to_xyz,
)
from emp.model import EMPMODEL

# argument parsing
model_default = EMPMODEL()
parser = argparse.ArgumentParser(
    description="Compute the surface EMP intensity using the Karzas-Latter-Seiler model"
)

parser.add_argument(
    "-gamma_yield_fraction",
    default=model_default.gamma_yield_fraction,
    type=float,
    help="Fraction of yield corresponding to prompt gamma rays",
)

parser.add_argument(
    "-Compton_KE",
    default=model_default.Compton_KE,
    type=float,
    help="Kinetic energy of Compton electrons [MeV]",
)

parser.add_argument(
    "-Bnorm",
    default=model_default.Bnorm,
    type=float,
    help="Geomagnetic field at target point [T]",
)

parser.add_argument(
    "-theta",
    default=model_default.theta,
    type=float,
    help="Angle between the line-of-sight vector and the geomagnetic field [radians]",
)

parser.add_argument(
    "-A",
    default=model_default.A,
    type=float,
    help="Angle between the line-of-sight vector and the normal of the Earth surface [radians]",
)

parser.add_argument(
    "-pulse_param_a",
    default=model_default.pulse_param_a,
    type=float,
    help="Pulse parameter a [ns^(-1)]",
)

parser.add_argument(
    "-pulse_param_b",
    default=model_default.pulse_param_b,
    type=float,
    help="Pulse parameter b [ns^(-1)]",
)

parser.add_argument(
    "-rtol",
    default=model_default.rtol,
    type=float,
    help="Relative tolerance used in the ODE integration",
)

parser.add_argument(
    "-N_pts_HOB", default=30, type=int, help="Number of HOB values to #"
)

parser.add_argument(
    "-N_pts_yield",
    default=30,
    type=int,
    help="Number of total_yield_kt values to #",
)

parser.add_argument(
    "-HOB_min", default=55.0, type=float, help="Minimum HOB to #"
)

parser.add_argument(
    "-HOB_max", default=400.0, type=float, help="Maximum HOB to #"
)

parser.add_argument(
    "-yield_min", default=1e0, type=float, help="Minimum total_yield_kt to #"
)

parser.add_argument(
    "-yield_max", default=1e3, type=float, help="Maximum total_yield_kt to #"
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
    "-save_str",
    default="",
    type=str,
    help="String used to save results from different data runs",
)

args = vars(parser.parse_args())
save_str = args.pop("save_str")

# print out param values
print("\nModel Parameters\n--------------------")
for key, value in args.items():
    print(key, "=", value)
print("\n")

# perform the scan
data_dic = HOB_yield_scan(**args)

# create data and figure directories
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("figures"):
    os.makedirs("figures")

# save the result
if save_str != "":
    save_str = "_" + save_str
with open("data/HOB_yield_scan" + save_str + ".pkl", "wb") as f:
    pickle.dump(data_dic, f)

x, y, z = data_dic_to_xyz(data_dic)
contourf = contour_plot(
    x, y, z, save_path="figures/HOB_yield_scan" + save_str, ngrid=50, levels=20
)
