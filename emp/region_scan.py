"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

import io
from pathlib import Path
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
from emp.config import (
    generate_configs,
    run_configs,
)
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
from emp.model import (
    EmpLosResult,
    EmpModel,
)

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


def load_scan_results(
    results_dir: Union[str, Path]
) -> Tuple[List[float], List[float], List[float], Point]:
    """
    Load all EmpLosResult JSON files in a directory and return lat, lon, max E lists,
    as well as burst point (verifying all results share the same burst point).

    Parameters
    ----------
    results_dir : Union[str, Path]
        Directory containing result JSON files.

    Returns
    -------
    Tuple[List[float], List[float], List[float], Point]
        latitudes, longitudes, max E-field magnitudes, and burst point.
    """
    # Extract the results files
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob("config_*_result.json"))
    if not json_files:
        raise ValueError(f"No result JSON files found in {results_dir}")

    lat_list, lon_list, E_max_list, burst_points = [], [], [], []

    for file in json_files:
        result = EmpLosResult.load(file)
        target = result.target_point_dict

        lat_list.append(target["latitude_rad"] * 180 / np.pi)
        lon_list.append(target["longitude_rad"] * 180 / np.pi)
        E_max_list.append(max(result.E_norm_at_ground))

        # Nested burst point structure
        burst_dict = result.burst_point_dict

        burst_points.append(
            Point.from_gps_coordinates(
                latitude=burst_dict["latitude_rad"] * 180 / np.pi,
                longitude=burst_dict["longitude_rad"] * 180 / np.pi,
                altitude_km=result.model_params["HOB"],
            )
        )

    # Verify all burst points are the same
    first_burst = burst_points[0]
    for bp in burst_points[1:]:
        if bp != first_burst:
            raise ValueError("Not all results share the same burst point!")

    return lat_list, lon_list, E_max_list, first_burst


'''
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
'''


def contour_plot(
    results_dir: Union[str, Path],
    save_path: Optional[str] = None,
    show_grid: bool = False,
    show: bool = True,
) -> Tuple[matplotlib.contour.QuadContourSet, List[float]]:
    """
    Create a contour plot directly from a directory of EMP result JSON files.

    Parameters
    ----------
    results_dir : Union[str, Path]
        Directory containing result JSON files (EmpLosResult).
    burst_point : Point
        Burst location for line-of-sight calculations.
    save_path : Optional[str], optional
        Path to save the figure, by default None.
    show_grid : bool, optional
        Whether to show the grid, by default False.
    show : bool, optional
        Whether to display the plot, by default True.

    Returns
    -------
    Tuple[matplotlib.contour.QuadContourSet, List[float]]
        Contour plot object and contour levels.
    """
    results_dir = Path(results_dir)
    lat_list, lon_list, E_max_list, burst_point = load_scan_results(results_dir)

    fig, ax = plt.subplots(dpi=150, figsize=(14, 10))

    # Create interpolation grid
    grid_size = 300
    xi = np.linspace(np.min(lon_list), np.max(lon_list), grid_size)
    yi = np.linspace(np.min(lat_list), np.max(lat_list), grid_size)
    triang = tri.Triangulation(lon_list, lat_list)
    interpolator = tri.LinearTriInterpolator(triang, E_max_list)
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
    level_spacing = 5e3
    z_min: float = np.nanmin(E_max_list)
    z_max: float = np.nanmax(E_max_list)
    level_min = int(np.floor(z_min / level_spacing)) - 1
    level_max = int(np.ceil(z_max / level_spacing)) + 1
    levels = [i * level_spacing for i in range(level_min, level_max + 1)]

    contourf = ax.contourf(Xi, Yi, zi, levels=levels, cmap="RdBu_r", extend="max")
    contour_lines = ax.contour(Xi, Yi, zi, levels=levels, linewidths=1, colors="k")
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt="%.0f")

    fig.colorbar(contourf, ax=ax, label=r"$E$ [V/m]")
    ax.set_xlabel("Longitude [degrees]")
    ax.set_ylabel("Latitude [degrees]")
    ax.grid(show_grid)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()

    return contourf, levels


def folium_plot(
    results_dir: Union[str, Path],
    burst_point: Point,
    save_path: str,
) -> folium.Map:
    """
    Create a folium map with EMP contours directly from result JSON files.

    Parameters
    ----------
    results_dir : Union[str, Path]
        Directory containing result JSON files (EmpLosResult).
    burst_point : Point
        Ground zero location.
    save_path : str
        Path to save the map image.

    Returns
    -------
    folium.Map
        Interactive folium map.
    """
    contourf, levels = contour_plot(
        results_dir, burst_point, save_path=None, show=False
    )

    # Convert to GeoJSON
    geojsonf = geojsoncontour.contourf_to_geojson(contourf=contourf)

    # Colormap
    cmap = plt.colormaps["RdBu_r"]
    colors = [cmap(x) for x in np.linspace(0, 1, len(levels))]
    colormap = branca.colormap.LinearColormap(
        colors, vmin=np.min(levels), vmax=np.max(levels)
    ).to_step(index=levels)

    # Create base map
    geomap = folium.Map(
        location=[burst_point.latitude_deg, burst_point.longitude_deg],
        width=750,
        height=750,
        zoom_start=5,
        tiles="CartoDB positron",
    )

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

    colormap.caption = "E-field Strength [V/m]"
    geomap.add_child(colormap)

    folium.Marker(
        location=[burst_point.latitude_deg, burst_point.longitude_deg],
        popup="Ground Zero",
        icon=folium.Icon(color="red", icon="flash"),
    ).add_to(geomap)

    _save_map_as_image(geomap, save_path)

    return geomap


'''
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
'''


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


def region_scan(
    base_config_path: str,
    scan_name: str,
    num_points_phi: int = 20,
    num_points_lambda: int = 20,
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
    num_points_phi : int, optional
        Number of latitude grid points, by default 20.
    num_points_lambda : int, optional
        Number of longitude grid points, by default 20.
    num_cores : int, optional
        Number of CPU cores to use for running the configurations, by default 1.
    """

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
    lat_1d_grid = burst_point.phi_g + np.linspace(
        -delta_angle / 2, delta_angle / 2, num_points_phi
    )
    long_1d_grid = burst_point.lambd_g + np.linspace(
        -delta_angle / 2, delta_angle / 2, num_points_lambda
    )

    # Convert to list of degrees
    lat_1d_grid = ((180 / np.pi) * lat_1d_grid).tolist()
    long_1d_grid = ((180 / np.pi) * long_1d_grid).tolist()

    # Create all config files
    generate_configs(
        base_config_path=base_config_path,
        output_dir="configs",
        scan_name=scan_name,
        parameters={
            "geometry": {
                "target_point": {
                    "latitude_deg": lat_1d_grid,
                    "longitude_deg": long_1d_grid,
                    "altitude_km": 0.0,
                }
            }
        },
    )

    # Run all config files
    run_configs(
        config_dir=f"configs/{scan_name}",
        results_dir=f"results/{scan_name}",
        num_cores=num_cores,
    )

    return
