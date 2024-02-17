"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

## imports
import argparse

import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import scipy
import scipy.ndimage
from cycler import cycler
from tqdm import tqdm

from emp.model import EMPMODEL

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 5.0
plt.rcParams["xtick.minor.size"] = 3.0
plt.rcParams["ytick.major.size"] = 5.0
plt.rcParams["ytick.minor.size"] = 3.0
plt.rcParams["lines.linewidth"] = 2
plt.rc("font", family="serif", size=16)
matplotlib.rc("text", usetex=True)
matplotlib.rc("legend", fontsize=16)
matplotlib.rcParams["axes.prop_cycle"] = cycler(
    color=["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42", "#FFB5B8"]
)


def data_dic_to_xyz(data_dic, gaussian_smooth=True, field_type="norm"):
    """
    Convert the data into three lists x, y, z, with
        x - yield
        y - HOB
        z - field strength

    Parameters
    ----------
    data_dic : Dict
        A dictionary containing the data.
    gaussian_smooth : bool, optional
        Boolean flag used to control whether Gaussian smoothing is
        applied. By default True
    field_type : str, optional
        Component of E-field to plot, can be 'norm', 'theta', or 'phi'.
        By default 'norm'

    Returns
    -------
    Tupe[List[float], List[float], List[float]]
        Returns the x,y,z lists of the extracted data.
    """
    y = []
    x = []
    z = []

    ## select which component of E-field to plot (norm, theta, or phi)
    if field_type == "norm":
        field_strength = data_dic["max_E_norm_at_ground"]
    elif field_type == "theta":
        field_strength = data_dic["max_E_theta_at_ground"]
    elif field_type == "phi":
        field_strength = data_dic["max_E_phi_at_ground"]

    ## perform gaussian smoothing to make nicer plots
    ## the motivation for this came from this SE post:
    ## https://stackoverflow.com/questions/12274529/how-to-smooth-matplotlib-contour-plot
    if gaussian_smooth:
        field_strength = scipy.ndimage.gaussian_filter(field_strength, 1)

    ## loop over the arrays and extract the points
    for i in range(data_dic["HOB"].shape[0]):
        for j in range(data_dic["HOB"].shape[1]):
            x.append(data_dic["total_yield_kt"][i, j])
            y.append(data_dic["HOB"][i, j])
            z.append(field_strength[i, j])
    x = np.log(x) / np.log(10)
    y = np.asarray(y)
    z = np.asarray(z)
    return x, y, z


def contour_plot(x, y, z, save_path=None, ngrid=50, levels=20):
    """
    Build a contour plot of the x, y, z data.
    Grid interpolation is used.

    TO DO: is the interpolation necessary?

    Parameters
    ----------
    x : List[float]
        A list of the x-values.
    y : List[float]
        A list of the y-values.
    z : List[float]
        A list of the z-values.
    save_path : _type_, optional
        Save path, by default None.
    grid : bool, optional
        Boolean flag used to control whether a grid should
        be displayed. By default False.

    Returns
    -------
    _type_
        A contourf object which can be used by folium.
    """

    fig, ax = plt.subplots(dpi=150, figsize=(7, 5))

    ## create grid values
    xi = np.linspace(np.min(x), np.max(x), ngrid)
    yi = np.linspace(np.min(y), np.max(y), ngrid)

    ## linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    ## other interpolation schemes
    # zi = scipy.interpolate.Rbf(x, y, z, function='linear')(Xi, Yi)

    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    # from scipy.interpolate import griddata
    # zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    ## create the plot
    contourf = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r")
    contour1 = ax.contour(
        xi, yi, zi, levels=levels, linewidths=1, linestyles="-", colors="k"
    )
    # ax.clabel(contour1, inline=True, fontsize=8, colors='k')
    fig.colorbar(contourf, ax=ax)
    # ax.set_xlabel(r'total yield [kt]', labelpad=10, fontsize=16)
    ax.set_xlabel(r"$\log_{10}$ (Y$_{tot}$/(1\, kt))", labelpad=10)
    ax.set_ylabel(r"HOB [km]", labelpad=10)
    # ax.set_title(r'Max EMP Intensity [V/m]')

    ax.set_yticks([55, 100, 150, 200, 250, 300, 350, 400])
    # xtickslocs = ax.get_xticks()
    # print(xtickslocs)
    # ax.set_xticklabels(np.exp(xtickslocs))

    ax.grid(False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.show()

    return contourf


## instantiate a default emp model to copy the default param values
model_default = EMPMODEL()


def HOB_yield_scan(
    gamma_yield_fraction=model_default.gamma_yield_fraction,
    Compton_KE=model_default.Compton_KE,
    Bnorm=model_default.Bnorm,
    theta=model_default.theta,
    A=model_default.A,
    pulse_param_a=model_default.pulse_param_a,
    pulse_param_b=model_default.pulse_param_b,
    rtol=model_default.rtol,
    N_pts_HOB=20,
    N_pts_yield=20,
    HOB_min=55.0,
    HOB_max=400.0,
    yield_min=1.0,
    yield_max=int(1e3),
    time_max=200.0,
    N_pts_time=150,
):
    """
    Performs a scan of the HOB and yield parameters.

    Parameters
    ----------
    gamma_yield_fraction : float, optional
        Fraction of total weapon yield devoted to gamma radiation,
        by default model_default.gamma_yield_fraction.
    Compton_KE : float, optional
        Kinetic energy of Comptons, by default model_default.Compton_KE.
    Bnorm : float, optional
        Norm of B-field, by default model_default.Bnorm.
    theta : float, optional
        Angle b/w line of sight and B-field, by default model_default.theta.
    A : float, optional
        Angle b/w line of sight and vertical, by default model_default.A.
    pulse_param_a : float, optional
        Pulse parameter a, by default model_default.pulse_param_a.
    pulse_param_b : float, optional
        Pulse paramater b, by default model_default.pulse_param_b.
    rtol : float, optional
        Relative tolerance for integration, by default model_default.rtol.
    N_pts_HOB : int, optional
        Number of grid points for the HOB scan, by default 20.
    N_pts_yield : int, optional
        Number of grid points for the yield scan, by default 20.
    HOB_min : float, optional
        Lower limit of HOB values to consider, by default 55.0.
    HOB_max : float, optional
        Upper limit of HOB values to consider, by default 400.0.
    yield_min : float, optional
        Lower limit of yield values to consider, by default 1.0.
    yield_max : _type_, optional
        Upper limit of yield values to consider, by default int(1e3).
    time_max : float, optional
        Maximum integration time, by default 200.0.
    N_pts_time : int, optional
        Number of temporal grid points, by default 150.

    Returns
    -------
    Dict
        A dictionary containing the results.
    """

    ## grids
    ## use log-linear for HOB, yield
    time_list = np.linspace(0, time_max, N_pts_time)
    HOB_list = np.exp(np.linspace(np.log(HOB_min), np.log(HOB_max), N_pts_HOB))
    total_yield_kt_list = np.exp(
        np.linspace(np.log(yield_min), np.log(yield_max), N_pts_yield)
    )

    ## initialize data dictionary
    data_dic = {
        "max_E_norm_at_ground": np.zeros((N_pts_HOB, N_pts_yield)),
        "max_E_theta_at_ground": np.zeros((N_pts_HOB, N_pts_yield)),
        "max_E_phi_at_ground": np.zeros((N_pts_HOB, N_pts_yield)),
        "HOB": np.zeros((N_pts_HOB, N_pts_yield)),
        "total_yield_kt": np.zeros((N_pts_HOB, N_pts_yield)),
    }

    ## loop over HOB
    for i in tqdm(range(N_pts_HOB)):
        ## loop over yield
        for j in tqdm(range(N_pts_yield), leave=bool(i == N_pts_HOB - 1)):
            ## update params
            HOB = HOB_list[i]
            total_yield_kt = total_yield_kt_list[j]

            ## define new EMP model and solve it
            model = EMPMODEL(
                HOB=HOB,
                Compton_KE=Compton_KE,
                total_yield_kt=total_yield_kt,
                gamma_yield_fraction=gamma_yield_fraction,
                pulse_param_a=pulse_param_a,
                pulse_param_b=pulse_param_b,
                rtol=rtol,
                A=A,
                Bnorm=Bnorm,
                theta=theta,
            )
            sol = model.solver(time_list)

            ## store results
            data_dic["max_E_norm_at_ground"][i, j] = np.max(sol["E_norm_at_ground"])
            data_dic["max_E_theta_at_ground"][i, j] = np.max(
                np.abs(sol["E_theta_at_ground"])
            )
            data_dic["max_E_phi_at_ground"][i, j] = np.max(
                np.abs(sol["E_phi_at_ground"])
            )
            data_dic["HOB"][i, j] = model.HOB
            data_dic["total_yield_kt"][i, j] = model.total_yield_kt

    return data_dic


## solve the model for a single line-of-sight integration
if __name__ == "__main__":
    ## argument parsing
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
        "-N_pts_HOB", default=30, type=int, help="Number of HOB values to consider"
    )

    parser.add_argument(
        "-N_pts_yield",
        default=30,
        type=int,
        help="Number of total_yield_kt values to consider",
    )

    parser.add_argument(
        "-HOB_min", default=55.0, type=float, help="Minimum HOB to consider"
    )

    parser.add_argument(
        "-HOB_max", default=400.0, type=float, help="Maximum HOB to consider"
    )

    parser.add_argument(
        "-yield_min", default=1e0, type=float, help="Minimum total_yield_kt to consider"
    )

    parser.add_argument(
        "-yield_max", default=1e3, type=float, help="Maximum total_yield_kt to consider"
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

    ## print out param values
    print("\nModel Parameters\n--------------------")
    for key, value in args.items():
        print(key, "=", value)
    print("\n")

    ## perform the scan
    data_dic = HOB_yield_scan(**args)

    ## create data and figure directories
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("figures"):
        os.makedirs("figures")

    ## save the result
    with open("data/HOB_yield_scan" + save_str + ".pkl", "wb") as f:
        pickle.dump(data_dic, f)

    x, y, z = data_dic_to_xyz(data_dic)
    contourf = contour_plot(
        x, y, z, save_path="figures/HOB_yield_scan" + save_str, ngrid=50, levels=20
    )
