"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing

Scan over the height of burst (HOB) and yield to generate EMP results.
"""

import shutil
from pathlib import Path
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import yaml  # type: ignore
from cycler import cycler
from matplotlib import contour

from emp.config import (
    generate_configs,
    run_configs,
)
from emp.model import EmpLosResult

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


def load_scan_results(
    results_dir: Union[str, Path]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Load all EmpLosResult JSON files in a directory and return HOB, total_yield_kt, max E lists.

    Parameters
    ----------
    results_dir : Union[str, Path]
        Directory containing result JSON files.

    Returns
    -------
    Tuple[List[float], List[float], List[float]]
        HOB, total_yield_kt, max E lists.
    """
    # Extract the results files
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob("config_*_result.json"))
    if not json_files:
        raise ValueError(f"No result JSON files found in {results_dir}")

    HOB_list, total_yield_kt_list, E_max_list = [], [], []

    for file in json_files:
        result = EmpLosResult.load(file)
        model_params = result.model_params

        HOB_list.append(result.model_params["HOB"])
        total_yield_kt_list.append(model_params["total_yield_kt"])
        E_max_list.append(max(result.E_norm_at_ground))

    return HOB_list, total_yield_kt_list, E_max_list


def contour_plot(
    results_dir: Union[str, Path],
    save_path: Optional[str] = None,
    show: bool = True,
    ngrid: int = 50,
    levels: int = 20,
) -> contour.QuadContourSet:
    """
    Create a contour plot of the EMP field strength based on results from a scan.

    Parameters
    ----------
    results_dir : Union[str, Path]
        Directory containing result JSON files.
    save_path : Optional[str], optional
        Path to save the plot, by default None (does not save).
    show : bool, optional
        Whether to display the plot, by default True.
    ngrid : int, optional
        Number of grid points for interpolation, by default 50.
    levels : int, optional
        Number of contour levels, by default 20.

    Returns
    -------
    contour.QuadContourSet
        The contour set object.
    """

    # Load the results
    results_dir = Path(results_dir)
    HOB_list, total_yield_kt_list, E_max_list = load_scan_results(results_dir)

    # Give convenient names
    x = np.log10(total_yield_kt_list).tolist()
    y = HOB_list
    z = E_max_list

    # Create grid values
    xi = np.linspace(np.min(x), np.max(x), ngrid)
    yi = np.linspace(np.min(y), np.max(y), ngrid)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Create the plot
    fig, ax = plt.subplots(dpi=150, figsize=(7, 5))
    contourf = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r")
    ax.contour(xi, yi, zi, levels=levels, linewidths=1, linestyles="-", colors="k")
    clb = fig.colorbar(contourf, ax=ax)
    clb.ax.set_title(r"[V/m]")
    ax.set_xlabel(r"$\log_{10}$ (Y$_{tot}$/(1\, kt))", labelpad=10)
    ax.set_ylabel(r"HOB [km]", labelpad=10)
    ax.set_yticks([55, 100, 150, 200, 250, 300, 350, 400])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()

    return contourf


def HOB_yield_scan(
    base_config_path: str,
    scan_name: str,
    N_pts_HOB: int = 20,
    N_pts_yield: int = 20,
    HOB_min: float = 55.0,
    HOB_max: float = 400.0,
    yield_min: float = 1.0,
    yield_max: float = 1e3,
    num_cores: int = 1,
) -> None:
    """
    Generate and run a regional EMP scan based on a base configuration file.

    Parameters
    ----------
    base_config_path : str
        Path to the base configuration file.
    scan_name : str
        Name for the scan, used to create output directory and config files.
    N_pts_HOB : int, optional
        Number of HOB values to scan, by default 20.
    N_pts_yield : int, optional
        Number of yield values to scan, by default 20.
    HOB_min : float, optional
        Minimum HOB value, by default 55.0.
    HOB_max : float, optional
        Maximum HOB value, by default 400.0.
    yield_min : float, optional
        Minimum yield value, by default 1.0.
    yield_max : float, optional
        Maximum yield value, by default 1e3.
    num_cores : int, optional
        Number of CPU cores to use for parallel processing, by default 1.
    """

    # Delete old configs and results if they exist
    config_dir = Path("configs") / scan_name
    results_dir = Path("results") / scan_name
    for path in [config_dir, results_dir]:
        if path.exists():
            shutil.rmtree(path)

    # Load base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Extract burst point from base config
    burst_cfg = base_config["geometry"]["burst_point"]

    # Generate a list of HOBs ranging from HOB_min to HOB_max
    HOB_list = np.linspace(HOB_min, HOB_max, N_pts_HOB).tolist()

    # Generate a list of total_yield_kt values ranging from yield_min to yield_max
    total_yield_kt_list = np.linspace(yield_min, yield_max, N_pts_yield).tolist()

    # Create all config files
    generate_configs(
        base_config_path=base_config_path,
        output_dir="configs",
        scan_name=scan_name,
        parameters={
            "model_parameters": {
                "total_yield_kt": total_yield_kt_list,
            },
            "geometry": {
                "burst_point": {
                    "latitude_deg": burst_cfg["latitude_deg"],
                    "longitude_deg": burst_cfg["longitude_deg"],
                    "altitude_km": HOB_list,
                }
            },
        },
    )

    # Run all config files
    run_configs(
        config_dir=f"configs/{scan_name}",
        results_dir=f"results/{scan_name}",
        num_cores=num_cores,
    )

    return
