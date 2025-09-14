"""Script for making plots."""
from typing import (
    Optional,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from scipy.optimize import differential_evolution

from emp.constants import (
    ABSORPTION_LAYER_UPPER,
    EARTH_RADIUS,
)
from emp.geomagnetic_field import DipoleMagneticField
from emp.geometry import (
    Point,
    get_line_of_sight_midway_point,
    great_circle_distance,
    line_of_sight_check,
)
from emp.region_scan import (
    compute_horizon_bbox,
    load_scan_results,
    wrap_lon_rad,
    wrap_longitudes,
)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def make_historical_contour_grid(
    save_path: Optional[str] = None,
    show: bool = False,
    gaussian_smooth: bool = False,
    gaussian_sigma: float = 1.0,
    use_log_scale: bool = False,
    level_spacing: float = 5e3,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a 2x3 grid of contour plots for historical nuclear tests.
    Uses consistent color scale and contour levels across all plots.

    Parameters
    ----------
    save_path : Optional[str], optional
        Path to save the figure, by default None.
    show : bool, optional
        Whether to display the plot, by default True.
    gaussian_smooth : bool, optional
        Whether to apply Gaussian smoothing, by default False.
    gaussian_sigma : float, optional
        Standard deviation for Gaussian smoothing, by default 1.0.
    use_log_scale : bool, optional
        Whether to use logarithmic color scale, by default True.
    level_spacing : float, optional
        Spacing between contour levels (only used if use_log_scale=False), by default 5e3.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and array of axes objects.
    """
    fs = 22
    test_names = ["K1", "K2", "K3", "K4", "K5", "StarfishPrime"]
    results_dirs = [f"results/{name}" for name in test_names]

    # First pass: load all data to determine global min/max for consistent levels
    all_E_max = []
    all_data = []

    for results_dir in results_dirs:
        loaded_results = load_scan_results(results_dir)
        E_max_list = loaded_results["E_max_list"]
        all_E_max.extend(E_max_list)
        all_data.append(loaded_results)

    # Define global contour levels
    z_min_global, z_max_global = np.min(all_E_max), np.max(all_E_max)

    if use_log_scale:
        # Logarithmic levels
        # Ensure minimum value is positive
        z_min_log = max(z_min_global, 1e3)  # Minimum 1 kV/m
        log_min = np.log10(z_min_log)
        log_max = np.log10(z_max_global)
        num_levels = 30
        global_levels = np.logspace(log_min, log_max, num_levels)
        norm = LogNorm(vmin=z_min_log, vmax=z_max_global)
    else:
        # Linear levels (original approach)
        level_min = int(np.floor(z_min_global / level_spacing)) - 1
        level_max = int(np.ceil(z_max_global / level_spacing)) + 1
        global_levels = [i * level_spacing for i in range(level_min, level_max + 1)]
        norm = None

    # Create 2x3 subplot grid with better spacing
    fig = plt.figure(figsize=(22, 14), dpi=300)

    # Use gridspec for better control over spacing and colorbar placement
    # Increased hspace for more vertical spacing between rows
    gs = GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.2,
        wspace=0.2,
        left=0.08,
        right=0.88,
        top=0.92,
        bottom=0.08,
    )  # Changed right from 0.92 to 0.88

    axes = []
    for i in range(2):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i, j]))

    # Process each test
    contourf = None  # Will store the last contourf for colorbar
    for idx, (loaded_results, test_name) in enumerate(zip(all_data, test_names)):
        ax = axes[idx]

        lat_list = np.array(loaded_results["lat_list"])
        lon_list_deg = loaded_results["lon_list"]
        E_max_list = loaded_results["E_max_list"]
        burst_point = loaded_results["burst_point"]
        HOB = int(burst_point.r_g - EARTH_RADIUS)

        # Convert and wrap longitudes
        lon_list = np.radians(lon_list_deg)
        lon_wrapped = wrap_longitudes(lon_list, burst_point.lambd_g)
        lon_wrapped_deg = np.degrees(lon_wrapped)

        # Create interpolation grid
        grid_size = 300
        xi = np.linspace(lon_wrapped_deg.min(), lon_wrapped_deg.max(), grid_size)
        yi = np.linspace(lat_list.min(), lat_list.max(), grid_size)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate
        zi = griddata(
            (lon_wrapped_deg, lat_list), E_max_list, (Xi, Yi), method="linear"
        )

        # Mask points outside line of sight
        for i, longitude in enumerate(np.radians(xi)):
            for j, latitude in enumerate(np.radians(yi)):
                target_point = Point(
                    EARTH_RADIUS, latitude, wrap_lon_rad(longitude), "lat/long geo"
                )
                if not line_of_sight_check(burst_point, target_point):
                    zi[j, i] = np.nan

        # Apply Gaussian smoothing if requested
        if gaussian_smooth:
            mask = np.isnan(zi)
            zi = np.nan_to_num(zi, nan=0.0)
            zi = scipy.ndimage.gaussian_filter(zi, sigma=gaussian_sigma)
            zi[mask] = np.nan

        # Create contour plots using global levels and normalization
        if use_log_scale:
            contourf = ax.contourf(
                Xi, Yi, zi, levels=global_levels, cmap="RdBu_r", extend="max", norm=norm
            )
            contour_lines = ax.contour(
                Xi,
                Yi,
                zi,
                levels=global_levels,
                linewidths=0.5,
                colors="k",
                alpha=0.7,
                norm=norm,
            )
        else:
            contourf = ax.contourf(
                Xi, Yi, zi, levels=global_levels, cmap="RdBu_r", extend="max"
            )
            contour_lines = ax.contour(
                Xi, Yi, zi, levels=global_levels, linewidths=0.5, colors="k", alpha=0.7
            )

        # Set labels and title with degree symbols
        # ax.set_xlabel("Longitude [degrees]", fontsize=fs)
        # ax.set_ylabel("Latitude [degrees]", fontsize=fs)
        if idx in [0, 3]:  # Left column
            ax.set_ylabel("Latitude [degrees]", fontsize=fs)
        else:
            ax.set_ylabel("")

        # Only show x-axis labels on bottom-most plots (indices 3, 4, 5)
        if idx in [3, 4, 5]:  # Bottom row
            ax.set_xlabel("Longitude [degrees]", fontsize=fs)
        else:
            ax.set_xlabel("")

        title = f"{test_name}, HOB: {HOB} [km]"
        title = f"{test_name}"
        ax.set_title(title, fontsize=fs, fontweight="bold", pad=10)
        ax.tick_params(labelsize=fs)

    # Add shared colorbar at the bottom with more space
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(contourf, cax=cbar_ax, orientation="vertical")
    cbar.ax.set_xlabel(r"$E$ [V/m]", fontsize=fs, labelpad=10)  # Label below colorbar
    cbar.ax.tick_params(labelsize=fs)

    # Set nice tick locations for colorbar
    if use_log_scale:
        # Define nice round numbers for log scale ticks
        nice_ticks = []
        log_min_nice = int(np.floor(np.log10(z_min_log)))
        log_max_nice = int(np.ceil(np.log10(z_max_global)))

        # Create ticks at powers of 10 and some intermediate values
        for power in range(log_min_nice, log_max_nice + 1):
            base_val = 10**power
            if base_val >= z_min_log and base_val <= z_max_global:
                nice_ticks.append(base_val)
            # Add intermediate values (2x and 5x) if they fit
            for mult in [2, 5]:
                val = mult * base_val
                if val >= z_min_log and val <= z_max_global and val not in nice_ticks:
                    nice_ticks.append(val)

        nice_ticks = sorted(nice_ticks)
        cbar.set_ticks(nice_ticks)
        cbar.set_ticklabels(
            [f"{tick:.0e}" if tick >= 1e4 else f"{tick:.0f}" for tick in nice_ticks]
        )
    else:
        # For linear scale, use round numbers
        tick_spacing = level_spacing * 2  # Every other level
        nice_ticks = np.arange(0, z_max_global + tick_spacing, tick_spacing)
        nice_ticks = nice_ticks[nice_ticks >= z_min_global]
        cbar.set_ticks(nice_ticks)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return


def make_Savage_figure() -> None:
    # Set-up
    phi_B_m_list = [15 * i * np.pi / 180 for i in range(1, 6)]  # burst latitude grid
    N_pts_HOB = 30  # number of heights to consider
    HOB_list = np.linspace(55, 400, N_pts_HOB)  # HOB grid

    phi_min_theta = np.zeros((len(phi_B_m_list), len(HOB_list)))
    phi_max_theta = np.zeros((len(phi_B_m_list), len(HOB_list)))
    ground_dists_min_theta = np.zeros((len(phi_B_m_list), len(HOB_list)))
    ground_dists_max_theta = np.zeros((len(phi_B_m_list), len(HOB_list)))

    magnetic_field = DipoleMagneticField()

    # Loop over latitude and HOB values
    for idx_phi in range(len(phi_B_m_list)):
        for idx_HOB in range(len(HOB_list)):
            Burst_Point = Point(
                EARTH_RADIUS + HOB_list[idx_HOB],
                phi_B_m_list[idx_phi],
                0,
                "lat/long mag",
            )

            def get_sin_theta_lat(phi_T_m):
                # Also check this
                Target_Point = Point(EARTH_RADIUS, phi_T_m, 0, "lat/long mag")
                Midway_Point = get_line_of_sight_midway_point(Burst_Point, Target_Point)
                theta = magnetic_field.get_theta_angle(
                    point_burst=Burst_Point, point_los=Midway_Point
                )
                return np.sin(theta)

            (lat_min, lat_max), (_, _) = compute_horizon_bbox(
                burst_point=Burst_Point, safety_margin=0.9
            )

            # Clip the limits
            lat_min = max(-np.pi / 2, lat_min)
            lat_max = min(np.pi / 2, lat_max)

            # Minimization
            result = scipy.optimize.minimize_scalar(
                lambda x: get_sin_theta_lat(x),
                method="bounded",
                bounds=(lat_min, lat_max),
            )
            Target_Point = Point(EARTH_RADIUS, result.x, 0, "lat/long mag")
            phi_min_theta[idx_phi, idx_HOB] = np.arcsin(result.fun)
            ground_dists_min_theta[idx_phi, idx_HOB] = great_circle_distance(
                Burst_Point, Target_Point
            )

            # Maximization
            # NOTE I found that with minimize_scalar, the resulting plot is screwed up
            # because the wrong maximum is being returned by the solver. The
            # differential_evolution fixed this.

            # result = scipy.optimize.minimize_scalar(lambda x: -get_sin_theta_lat(x), method='bounded', bounds=(lat_min, lat_max))

            result = differential_evolution(
                lambda x: -get_sin_theta_lat(x[0]), bounds=[(lat_min, lat_max)], seed=42
            )

            Target_Point = Point(EARTH_RADIUS, result.x, 0, "lat/long mag")
            phi_max_theta[idx_phi, idx_HOB] = np.arcsin(result.fun)
            ground_dists_max_theta[idx_phi, idx_HOB] = great_circle_distance(
                Burst_Point, Target_Point
            )

    # Make plot
    fig, ax = plt.subplots(figsize=(2 * 7, 5))

    # Loop over latitude values
    for idx_phi in range(len(phi_B_m_list)):
        ax.plot(
            list(-ground_dists_min_theta[idx_phi, ::-1])
            + list(ground_dists_max_theta[idx_phi, :]),
            list(HOB_list)[::-1] + list(HOB_list),
            "-",
            label=r"$\phi_m^B = %.0f^{\circ}$N" % (180 / np.pi * phi_B_m_list[idx_phi]),
        )

    ax.axhline(ABSORPTION_LAYER_UPPER, linestyle="--", linewidth=1, color="k")
    ax.axvline(0, linestyle="-", linewidth=1, color="k")
    ax.legend(bbox_to_anchor=(1.0, 1.05))
    ax.set_xlabel(r"Ground Distance South of GZ \ [km]")
    ax.set_ylabel(r"HOB \ [km]")
    ax.set_ylim([0, max(HOB_list)])
    ax.minorticks_on()

    props = dict(boxstyle="round", facecolor=colors[-1], alpha=0.3, edgecolor="black")
    ax.text(
        0.85,
        0.35,
        "Max Point\nSouth of GZ",
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment="top",
        bbox=props,
    )
    ax.text(
        0.05,
        0.35,
        "Null Point\nNorth of GZ",
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment="top",
        bbox=props,
    )

    plt.savefig("figures/fig_2_11_savage_et_al.png", bbox_inches="tight")


if __name__ == "__main__":
    # make_historical_plot()
    # historical_contour_grid(save_path="figures/historical_contour_grid.png",)
    make_Savage_figure()
