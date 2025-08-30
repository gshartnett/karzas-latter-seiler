"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

import io
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import branca
import folium
import geojsoncontour
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import scipy.ndimage
import yaml  # type: ignore
from cycler import cycler
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

import emp.geometry as geometry
from emp.config import generate_configs
from emp.constants import (
    DEFAULT_HOB,
    EARTH_RADIUS,
    DEFAULT_Compton_KE,
    DEFAULT_gamma_yield_fraction,
    DEFAULT_pulse_param_a,
    DEFAULT_pulse_param_b,
    DEFAULT_rtol,
    DEFAULT_total_yield_kt,
)
from emp.geomagnetic_field import MagneticFieldFactory
from emp.geometry import Point
from emp.model import EmpModel

# Configure matplotlib
plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5.0,
        "xtick.minor.size": 3.0,
        "ytick.major.size": 5.0,
        "ytick.minor.size": 3.0,
        "lines.linewidth": 2,
        "font.family": "serif",
        "font.size": 16,
        "text.usetex": True,
        "legend.fontsize": 16,
        "axes.prop_cycle": cycler(
            color=[
                "#E24A33",
                "#348ABD",
                "#988ED5",
                "#777777",
                "#FBC15E",
                "#8EBA42",
                "#FFB5B8",
            ]
        ),
    }
)


def data_dic_to_xyz(
    data_dic: Dict[str, Any], gaussian_smooth: bool = True, field_type: str = "norm"
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert data dictionary to coordinate arrays for plotting.

    Parameters
    ----------
    data_dic : Dict[str, Any]
        Dictionary containing scan results with field components and coordinates.
    gaussian_smooth : bool, optional
        Whether to apply Gaussian smoothing to reduce noise, by default True.
    field_type : str, optional
        E-field component to extract: 'norm', 'theta', or 'phi', by default 'norm'.

    Returns
    -------
    Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
        Longitude (x), latitude (y), and field strength (z) arrays.

    Raises
    ------
    ValueError
        If field_type is not one of the supported options.
    """
    # Validate field type
    field_map = {
        "norm": "max_E_norm_at_ground",
        "theta": "max_E_theta_at_ground",
        "phi": "max_E_phi_at_ground",
    }

    if field_type not in field_map:
        raise ValueError(
            f"Invalid field_type '{field_type}'. Must be one of {list(field_map.keys())}"
        )

    field_strength = data_dic[field_map[field_type]]

    # Apply Gaussian smoothing if requested
    if gaussian_smooth:
        field_strength = scipy.ndimage.gaussian_filter(field_strength, sigma=1)

    # Extract coordinates and field values
    x_coords = data_dic["lamb_T_g"] * 180 / np.pi  # longitude in degrees
    y_coords = data_dic["phi_T_g"] * 180 / np.pi  # latitude in degrees

    return (
        np.asarray(x_coords.flatten(), dtype=np.floating),
        np.asarray(y_coords.flatten(), dtype=np.floating),
        np.asarray(field_strength.flatten(), dtype=np.floating),
    )


def contour_plot(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    burst_point: Point,
    save_path: Optional[str] = None,
    show_grid: bool = False,
    show: bool = True,
) -> Tuple[matplotlib.contour.QuadContourSet, List[float]]:
    """
    Create a contour plot of EMP field strength.

    Parameters
    ----------
    x : NDArray[np.floating]
        Longitude coordinates in degrees.
    y : NDArray[np.floating]
        Latitude coordinates in degrees.
    z : NDArray[np.floating]
        Field strength values.
    burst_point : Point
        Burst location for line-of-sight calculations.
    save_path : Optional[str], optional
        Path to save the figure, by default None.
    show_grid : bool, optional
        Whether to display grid lines, by default False.
    show : bool, optional
        Whether to display the plot, by default True.

    Returns
    -------
    Tuple[matplotlib.contour.QuadContourSet, List[float]]
        Contour plot object and contour levels for use with folium.
    """
    fig, ax = plt.subplots(dpi=150, figsize=(14, 10))

    # Create interpolation grid
    grid_size = 300
    xi = np.linspace(np.min(x), np.max(x), grid_size)
    yi = np.linspace(np.min(y), np.max(y), grid_size)

    # Interpolate data onto regular grid
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Mask points outside line of sight
    for i, longitude in enumerate(xi):
        for j, latitude in enumerate(yi):
            phi = latitude * np.pi / 180
            lambd = longitude * np.pi / 180
            target_point = Point(EARTH_RADIUS, phi, lambd, "lat/long geo")
            try:
                geometry.line_of_sight_check(burst_point, target_point)
            except Exception:
                zi[j, i] = np.nan

    # Define contour levels
    level_spacing = 5e3  # 5 kV/m spacing
    z_min: float = np.nanmin(z)
    z_max: float = np.nanmax(z)
    level_min = int(np.floor(z_min / level_spacing)) - 1
    level_max = int(np.ceil(z_max / level_spacing)) + 1
    levels = [i * level_spacing for i in range(level_min, level_max + 1)]

    # Create contour plots
    contourf = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r", extend="max")
    contour_lines = ax.contour(xi, yi, zi, levels=levels, linewidths=1, colors="k")

    # Add labels and formatting
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt="%.0f")
    fig.colorbar(contourf, ax=ax, label=r"$E$ [V/m]")
    ax.set_xlabel(r"Longitude [degrees]", labelpad=10)
    ax.set_ylabel(r"Latitude [degrees]", labelpad=10)
    ax.grid(show_grid)

    # Save and/or display
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()

    return contourf, levels


def folium_plot(
    contourf: matplotlib.contour.QuadContourSet,
    lat0: float,
    long0: float,
    levels: List[float],
    save_path: str,
) -> folium.Map:
    """
    Create an interactive map with EMP contours overlaid.

    Parameters
    ----------
    contourf : matplotlib.contour.QuadContourSet
        Matplotlib contour object to convert to map overlay.
    lat0 : float
        Ground zero latitude in degrees.
    long0 : float
        Ground zero longitude in degrees.
    levels : List[float]
        Contour levels for colormap.
    save_path : str
        Path to save the map image.

    Returns
    -------
    folium.Map
        Interactive folium map object.
    """
    # Convert contours to GeoJSON
    geojsonf = geojsoncontour.contourf_to_geojson(contourf=contourf)

    # Set up colormap
    cmap = plt.colormaps["RdBu_r"]

    colors = [cmap(x) for x in np.linspace(0, 1, len(levels))]
    colormap = branca.colormap.LinearColormap(
        colors, vmin=np.min(levels), vmax=np.max(levels)
    ).to_step(index=levels)

    # Create base map
    geomap = folium.Map(
        location=[lat0, long0],
        width=750,
        height=750,
        zoom_start=5,
        tiles="CartoDB positron",
    )

    # Add contour overlay
    folium.GeoJson(
        geojsonf,
        style_function=lambda feature: {
            "color": "gray",
            "weight": 1,
            "fillColor": feature["properties"]["fill"],
            "opacity": 1,
            "fillOpacity": 0.5,
        },
    ).add_to(geomap)

    # Add colorbar and ground zero marker
    colormap.caption = "E-field Strength [V/m]"
    geomap.add_child(colormap)

    folium.Marker(
        location=[lat0, long0],
        popup="Ground Zero",
        icon=folium.Icon(color="red", icon="flash"),
    ).add_to(geomap)

    # Convert to image and save
    _save_map_as_image(geomap, save_path)

    return geomap


def _save_map_as_image(geomap: folium.Map, save_path: str) -> None:
    """Save folium map as a PNG image."""
    img_data = geomap._to_png(delay=10)
    img = Image.open(io.BytesIO(img_data))

    # Crop whitespace
    pix = np.asarray(img)
    non_white_indices = np.where(pix < 255)[:2]  # Find non-white pixels
    if len(non_white_indices[0]) > 0 and len(non_white_indices[1]) > 0:
        y_min, y_max = non_white_indices[0].min(), non_white_indices[0].max()
        x_min, x_max = non_white_indices[1].min(), non_white_indices[1].max()
        cropped = img.crop((x_min, y_min, x_max, y_max))
    else:
        cropped = img

    # Save as high-resolution PNG
    cropped.convert("RGB").save(save_path, dpi=(300, 300))


def region_scan_new(
    base_config_path: str,
    scan_name: str,
    num_points_phi: int = 20,
    num_points_lambda: int = 20,
) -> None:
    # Load base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Extract burst point from base config
    burst_cfg = base_config["geometry"]["burst_point"]

    burst_point = Point.from_gps_coordinates(
        latitude=burst_cfg["latitude_deg"],
        longitude=burst_cfg["longitude_deg"],
        altitude_km=burst_cfg["altitude_km"],
    )

    # Identify grid of angular values
    delta_angle = 2.0 * geometry.compute_max_delta_angle_2d(burst_point)
    print(f"Delta angle: {delta_angle:.6f} radians")

    lat_1d_grid = burst_point.phi_g + np.linspace(
        -delta_angle / 2, delta_angle / 2, num_points_phi
    )
    long_1d_grid = burst_point.lambd_g + np.linspace(
        -delta_angle / 2, delta_angle / 2, num_points_lambda
    )

    # Convert to list of degrees
    lat_1d_grid = ((180 / np.pi) * lat_1d_grid).tolist()
    long_1d_grid = ((180 / np.pi) * long_1d_grid).tolist()

    # Cartesian product → 2 1D arrays
    # lat_grid, long_grid = np.meshgrid(lat_1d_grid, long_1d_grid, indexing="ij")
    # lat_values = lat_grid.ravel().to_list()
    # long_values = long_grid.ravel().to_list()

    # Create all config files
    generate_configs(
        base_config_path=base_config_path,
        scan_name=scan_name,
        parameters={
            "latitude": lat_1d_grid,
            "longitude": long_1d_grid,
        },
        output_dir=f"configs/{scan_name}",
    )

    # Run all config files

    return


def region_scan(
    burst_point: Point,
    HOB: float = DEFAULT_HOB,
    Compton_KE: float = DEFAULT_Compton_KE,
    total_yield_kt: float = DEFAULT_total_yield_kt,
    gamma_yield_fraction: float = DEFAULT_gamma_yield_fraction,
    pulse_param_a: float = DEFAULT_pulse_param_a,
    pulse_param_b: float = DEFAULT_pulse_param_b,
    rtol: float = DEFAULT_rtol,
    N_pts_phi: int = 20,
    N_pts_lambd: int = 20,
    time_max: float = 100.0,
    N_pts_time: int = 50,
    b_field_type: str = "dipole",
) -> Dict[str, Union[NDArray[np.floating], NDArray[Any]]]:
    """
    Perform 2D regional scan of EMP field strength.

    This function computes the maximum EMP field strength at ground level
    over a grid of geographic coordinates around the burst point.

    Parameters
    ----------
    burst_point : Point
        Location of the nuclear burst.
    HOB : float, optional
        Height of burst in km, by default DEFAULT_HOB.
    Compton_KE : float, optional
        Compton electron kinetic energy in MeV, by default DEFAULT_Compton_KE.
    total_yield_kt : float, optional
        Total weapon yield in kilotons, by default DEFAULT_total_yield_kt.
    gamma_yield_fraction : float, optional
        Fraction of yield in gamma rays, by default DEFAULT_gamma_yield_fraction.
    pulse_param_a : float, optional
        Pulse shape parameter in 1/ns, by default DEFAULT_pulse_param_a.
    pulse_param_b : float, optional
        Pulse shape parameter in 1/ns, by default DEFAULT_pulse_param_b.
    rtol : float, optional
        Relative tolerance for ODE solver, by default DEFAULT_rtol.
    N_pts_phi : int, optional
        Number of latitude grid points, by default 20.
    N_pts_lambd : int, optional
        Number of longitude grid points, by default 20.
    time_max : float, optional
        Maximum simulation time in ns, by default 100.0.
    N_pts_time : int, optional
        Number of time points, by default 50.
    b_field_type : str, optional
        Magnetic field model ('dipole' or 'igrf'), by default 'dipole'.

    Returns
    -------
    Dict[str, Union[NDArray[np.floating], NDArray[Any]]]
        Dictionary containing scan results with field components and coordinates.
    """
    # Set up time and spatial grids
    time_list = np.linspace(0, time_max, N_pts_time)

    delta_angle = 2.0 * geometry.compute_max_delta_angle_2d(burst_point)
    phi_grid = burst_point.phi_g + np.linspace(
        -delta_angle / 2, delta_angle / 2, N_pts_phi
    )
    lambd_grid = burst_point.lambd_g + np.linspace(
        -delta_angle / 2, delta_angle / 2, N_pts_lambd
    )

    # Initialize magnetic field model
    geomagnetic_field = MagneticFieldFactory().create(b_field_type)

    # Initialize results storage
    results = {
        "max_E_norm_at_ground": np.zeros((N_pts_phi, N_pts_lambd)),
        "max_E_theta_at_ground": np.zeros((N_pts_phi, N_pts_lambd)),
        "max_E_phi_at_ground": np.zeros((N_pts_phi, N_pts_lambd)),
        "theta": np.zeros((N_pts_phi, N_pts_lambd), dtype=np.float64),
        "A": np.zeros((N_pts_phi, N_pts_lambd), dtype=np.float64),
        "phi_T_g": np.zeros((N_pts_phi, N_pts_lambd)),
        "lamb_T_g": np.zeros((N_pts_phi, N_pts_lambd)),
    }

    # Perform scan
    for i in tqdm(range(N_pts_phi), desc="Latitude"):
        for j in tqdm(range(N_pts_lambd), desc="Longitude", leave=(i == N_pts_phi - 1)):
            # Define target point
            target_point = Point(
                EARTH_RADIUS, phi_grid[i], lambd_grid[j], "lat/long geo"
            )
            midway_point = geometry.get_line_of_sight_midway_point(
                burst_point, target_point
            )

            # Store coordinates
            results["phi_T_g"][i, j] = target_point.phi_g
            results["lamb_T_g"][i, j] = target_point.lambd_g

            try:
                # Calculate geometric and magnetic field parameters
                A_angle = geometry.get_A_angle(burst_point, midway_point)
                theta_angle = geomagnetic_field.get_theta_angle(
                    point_burst=burst_point, point_los=midway_point
                )
                B_magnitude = geomagnetic_field.get_field_magnitude(midway_point)

                # Create and solve EMP model
                model = EmpModel(
                    HOB=HOB,
                    Compton_KE=Compton_KE,
                    total_yield_kt=total_yield_kt,
                    gamma_yield_fraction=gamma_yield_fraction,
                    pulse_param_a=pulse_param_a,
                    pulse_param_b=pulse_param_b,
                    rtol=rtol,
                    A=A_angle,
                    theta=theta_angle,
                    Bnorm=B_magnitude,
                )

                solution = model.solver(time_list)

                # Store model parameters and results
                results["theta"][i, j] = model.theta
                results["A"][i, j] = model.A
                results["max_E_norm_at_ground"][i, j] = np.max(
                    solution["E_norm_at_ground"]
                )
                results["max_E_theta_at_ground"][i, j] = np.max(
                    np.abs(solution["E_theta_at_ground"])
                )
                results["max_E_phi_at_ground"][i, j] = np.max(
                    np.abs(solution["E_phi_at_ground"])
                )

            except Exception:
                # Handle points outside line of sight or other errors
                results["theta"][i, j] = None
                results["A"][i, j] = None
                results["max_E_norm_at_ground"][i, j] = 0.0
                results["max_E_theta_at_ground"][i, j] = 0.0
                results["max_E_phi_at_ground"][i, j] = 0.0

    return results
