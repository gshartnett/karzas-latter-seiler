"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

import io
from pathlib import Path
from typing import (
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
from PIL import Image

import emp.geometry as geometry
from emp.config import (
    generate_configs,
    run_configs,
)
from emp.constants import EARTH_RADIUS
from emp.geometry import Point
from emp.model import EmpLosResult

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


def contour_plot(
    results_dir: Union[str, Path],
    save_path: Optional[str] = None,
    show_grid: bool = False,
    show: bool = True,
    gaussian_smooth: bool = False,
    gaussian_sigma: float = 1.0,
) -> Tuple[matplotlib.contour.QuadContourSet, List[float]]:
    """
    Create a contour plot directly from a directory of EMP result JSON files.

    Parameters
    ----------
    results_dir : Union[str, Path]
        Directory containing result JSON files (EmpLosResult).
    save_path : Optional[str], optional
        Path to save the plot image, by default None (does not save).
    show_grid : bool, optional
        Whether to show grid lines on the plot, by default False.
    show : bool, optional
        Whether to display the plot, by default True.
    gaussian_smooth : bool, optional
        Whether to apply Gaussian smoothing to the E-field data, by default False.
    gaussian_sigma : float, optional
        Standard deviation for Gaussian smoothing, by default 1.0.

    Returns
    -------
    Tuple[matplotlib.contour.QuadContourSet, List[float]]
        The contour set and the contour levels used in the plot.
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

    # Apply Gaussian smoothing if requested
    if gaussian_smooth:
        # Replace NaN with 0 temporarily (gaussian_filter doesn’t handle NaN)
        mask = np.isnan(zi)
        zi = np.nan_to_num(zi, nan=0.0)
        zi = scipy.ndimage.gaussian_filter(zi, sigma=gaussian_sigma)
        zi[mask] = np.nan  # restore NaNs

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
    save_path: str,
    gaussian_smooth: bool = True,
    gaussian_sigma: float = 1.0,
) -> folium.Map:
    """
    Create a folium map with EMP contours directly from result JSON files.

    Parameters
    ----------
    results_dir : Union[str, Path]
        Directory containing result JSON files (EmpLosResult).
    save_path : str
        Path to save the map image.
    gaussian_smooth : bool, optional
        Whether to apply Gaussian smoothing to the E-field data, by default True.
    gaussian_sigma : float, optional
        Standard deviation for Gaussian smoothing, by default 1.0.

    Returns
    -------
    folium.Map
        Interactive folium map.
    """

    # Retrieve the burst point
    results_dir = Path(results_dir)
    _, _, _, burst_point = load_scan_results(results_dir)
    lat = burst_point.phi_g * 180 / np.pi
    long = burst_point.lambd_g * 180 / np.pi

    contourf, levels = contour_plot(
        results_dir,
        gaussian_smooth=gaussian_smooth,
        gaussian_sigma=gaussian_sigma,
        show=False,
    )

    # Convert to GeoJSON
    geojsonf = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
    )

    # Colormap
    cmap = plt.colormaps["RdBu_r"]
    colors = [cmap(x) for x in np.linspace(0, 1, len(levels))]
    colormap = branca.colormap.LinearColormap(
        colors, vmin=np.min(levels), vmax=np.max(levels)
    ).to_step(index=levels)

    # Create base map
    geomap = folium.Map(
        location=[lat, long],
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
        location=[lat, long],
        popup="Ground Zero",
        icon=folium.Icon(color="red", icon="flash"),
    ).add_to(geomap)

    _save_map_as_image(geomap, save_path)

    return geomap


def _save_map_as_image(geomap: folium.Map, save_path: str) -> None:
    """
    Save folium map as a PNG image.

    Parameters
    ----------
    geomap : folium.Map
        The folium map to save.
    save_path : str
        Path to save the PNG image.
    """
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
