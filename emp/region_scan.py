"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

# imports
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
import scipy
import scipy.ndimage
from cycler import cycler
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

import emp.geometry as geometry
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
from emp.model import EmpModel

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


def data_dic_to_xyz(
    data_dic: Dict[str, Any], gaussian_smooth: bool = True, field_type: str = "norm"
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert the data into three lists x, y, z, with
        x - latitude
        y - longitude
        z - field strength

    Parameters
    ----------
    data_dic : Dict[str, Any]
        A dictionary containing the data.
    gaussian_smooth : bool, optional
        Boolean flag used to control whether Gaussian smoothing is
        applied. By default True
    field_type : str, optional
        Component of E-field to plot, can be 'norm', 'theta', or 'phi'.
        By default 'norm'

    Returns
    -------
    Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
        Returns the x,y,z arrays of the extracted data.
    """
    y: List[float] = []
    x: List[float] = []
    z: List[float] = []

    # select which component of E-field to plot (norm, theta, or phi)
    if field_type == "norm":
        field_strength = data_dic["max_E_norm_at_ground"]
    elif field_type == "theta":
        field_strength = data_dic["max_E_theta_at_ground"]
    elif field_type == "phi":
        field_strength = data_dic["max_E_phi_at_ground"]
    else:
        raise ValueError(f"Invalid field_type: {field_type}")

    # perform gaussian smoothing to make nicer plots
    # the motivation for this came from this SE post:
    # https://stackoverflow.com/questions/12274529/how-to-smooth-matplotlib-contour-plot
    if gaussian_smooth:
        field_strength = scipy.ndimage.gaussian_filter(field_strength, 1)

    # loop over the arrays and extract the points
    for i in range(data_dic["theta"].shape[0]):
        for j in range(data_dic["theta"].shape[1]):
            x.append(data_dic["lamb_T_g"][i, j] * 180 / np.pi)
            y.append(data_dic["phi_T_g"][i, j] * 180 / np.pi)
            z.append(field_strength[i, j])

    x_array = np.asarray(x, dtype=np.floating)
    y_array = np.asarray(y, dtype=np.floating)
    z_array = np.asarray(z, dtype=np.floating)

    return x_array, y_array, z_array


def contour_plot(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    Burst_Point: geometry.Point,
    save_path: Optional[str] = None,
    grid: bool = False,
    show: bool = True,
) -> Tuple[matplotlib.contour.QuadContourSet, List[float]]:
    """
    Build a contour plot of the x, y, z data.
    Grid interpolation is used.

    TO DO: is the interpolation necessary?

    Parameters
    ----------
    x : NDArray[np.floating]
        Array of the x-values.
    y : NDArray[np.floating]
        Array of the y-values.
    z : NDArray[np.floating]
        Array of the z-values.
    Burst_Point : geometry.Point
        The burst point.
    save_path : Optional[str], optional
        Save path, by default None.
    grid : bool, optional
        Boolean flag used to control whether a grid should
        be displayed. By default False.
    show : bool, optional
        Whether to show the plot. By default True.

    Returns
    -------
    Tuple[matplotlib.contour.QuadContourSet, List[float]]
        A contourf object and levels list which can be used by folium.
    """

    fig, ax = plt.subplots(dpi=150, figsize=(14, 10))

    # create grid values
    ngrid = 300
    xi = np.linspace(np.min(x), np.max(x), ngrid)
    yi = np.linspace(np.min(y), np.max(y), ngrid)

    # linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # set the field value ot nan for any grid points that are outside the horizon
    for i in range(len(xi)):
        for j in range(len(yi)):
            phi = yi[j] * np.pi / 180
            lambd = xi[i] * np.pi / 180
            Target_Point = geometry.Point(EARTH_RADIUS, phi, lambd, "lat/long geo")
            try:
                geometry.line_of_sight_check(Burst_Point, Target_Point)
            except:
                zi[j, i] = np.nan

    # other interpolation schemes
    # zi = scipy.interpolate.Rbf(x, y, z, function='linear')(Xi, Yi)

    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    # from scipy.interpolate import griddata
    # zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    # create the plot
    level_spacing = 5 * 1e3
    levels: List[float] = [
        i * level_spacing
        for i in range(
            int(np.round(np.min(z) / level_spacing)) - 1,
            int(np.round(np.max(z) / level_spacing)) + 1,
        )
    ]

    contourf = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r", extend="max")
    contour1 = ax.contour(
        xi, yi, zi, levels=levels, linewidths=1, linestyles="-", colors="k"
    )

    ax.clabel(contour1, inline=1, fontsize=10)

    # ax.clabel(contour1, inline=True, fontsize=8, colors='k')
    fig.colorbar(contourf, ax=ax, label=r"$E$ [V/m]")
    ax.set_xlabel(r"$\lambda$ (longitude) [degrees]", labelpad=10)
    ax.set_ylabel(r"$\phi$ (latitude) [degrees]", labelpad=10)
    # ax.set_title(r'Max EMP Intensity [V/m]')
    ax.grid(grid)
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
    Super-impose a contour plot on top of a folium map.

    Parameters
    ----------
    contourf : matplotlib.contour.QuadContourSet
        Matplotlib contour object.
    lat0 : float
        Latitude of ground zero in degrees.
    long0 : float
        Longitude of ground zero in degrees.
    levels : List[float]
        Levels for the color map.
    save_path : str
        Save path.

    Returns
    -------
    folium.Map
        A folium geomap object.
    """

    # convert matplotlib contourf to geojson
    # can swap out contourf for contour to get a plot with just the contour lines
    geojson = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
        min_angle_deg=3.0,
        ndigits=5,
        stroke_width=1,
        fill_opacity=0.9,
    )
    geojsonf = geojsoncontour.contourf_to_geojson(contourf=contourf)

    # set up color map
    cmap = plt.colormaps["RdBu_r"]
    colors = [cmap(x) for x in np.linspace(0, 1, len(levels))]
    cm = branca.colormap.LinearColormap(
        colors, vmin=np.min(levels), vmax=np.max(levels)
    ).to_step(index=levels)

    # tile options: "OpenStreetMap", "Stamen Terrain", "Stamen Toner", "Stamen Watercolor", "CartoDB positron", "CartoDB dark_matter"
    geomap = folium.Map(
        [lat0, long0], width=750, height=750, zoom_start=5, tiles="CartoDB positron"
    )

    # plot the contour plot ont folium
    # see here for style function params: https://leafletjs.com/reference-1.6.0.html#path-option
    folium.GeoJson(
        geojsonf,
        style_function=lambda x: {
            "color": "gray",  # x['properties']['stroke'], #color of contour lines
            "weight": 1,  # x['properties']['stroke-width'], #thickness of contour lines
            "fillColor": x["properties"]["fill"],  # color in between contour lines
            "opacity": 1,  # opacity of contour lines
            "fillOpacity": 0.5,
        },
    ).add_to(geomap)

    # add the colormap to the folium map
    cm.caption = "Enorm (V/m)"
    geomap.add_child(cm)

    # feature_group = folium.FeatureGroup("Locations")
    # for lat, lng, name in zip(lat_lst, lng_lst, name_lst):
    #    feature_group.add_child(folium.Marker(location=[lat,lon],popup=name))
    # map.add_child(feature_group)
    geomap.add_child(
        folium.FeatureGroup("Locations").add_child(
            folium.Marker(location=[lat0, long0], popup="Ground Zero")
        )
    )

    # convert the geomap to a PIL image
    # see: https://stackoverflow.com/questions/40208051/selenium-using-python-geckodriver-executable-needs-to-be-in-path
    img_data = geomap._to_png(10)
    img = Image.open(io.BytesIO(img_data))

    # remove any white-space
    # see: https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
    pix = np.asarray(img)
    idx = np.where(pix - 255)[0:2]  # Drop the color when finding edges
    box = list(map(min, idx))[::-1] + list(map(max, idx))[::-1]
    region = img.crop(box)
    region_pix = np.asarray(region)
    img2 = Image.fromarray(region_pix)

    # save
    img2.convert("RGB").save(save_path, dpi=(900, 900))

    # display result
    return geomap


def region_scan(
    Burst_Point: geometry.Point,
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
    Function used to perform the 2d region scan.
    Note: In order to make smooth-looking plots, a grid interpolation is
    applied to the result of this scan. This can introduce numerical
    artifacts if the field values sharply jump to zero as the evaluation
    point moves outside the line-of-sight cone. Therefore, in this scan
    no attempt is made to restrict evaluation points to lie in the line
    of sight. Instead, this constraint is enforced after grid interpolation
    has been applied in `contour_plot`.

    Parameters
    ----------
    Burst_Point : geometry.Point
        Burst point.
    HOB : float, optional
        Height of burst, in km, by default DEFAULT_HOB.
    Compton_KE : float, optional
        Compton kinetic energy, in MeV, by default DEFAULT_Compton_KE.
    total_yield_kt : float, optional
        Total weapon yield in kt, by default DEFAULT_total_yield_kt.
    gamma_yield_fraction : float, optional
        Fraction of yield invested in gamma-rays, by default DEFAULT_gamma_yield_fraction.
    pulse_param_a : float, optional
        Pulse parameter a in 1/ns, by default DEFAULT_pulse_param_a.
    pulse_param_b : float, optional
        Pulse parameter b in 1/ns, by default DEFAULT_pulse_param_b.
    rtol : float, optional
        Relative tolerance parameter, by default DEFAULT_rtol.
    N_pts_phi : int, optional
        Number of latitude grid points, by default 20.
    N_pts_lambd : int, optional
        Number of longitude grid points, by default 20.
    time_max : float, optional
        Max time for simulation in ns, by default 100.0
    N_pts_time : int, optional
        Number of temporal grid points, by default 50.
    b_field_type : str, optional
        The geomagnetic field type, defaults to dipole.

    Returns
    -------
    Dict[str, Union[NDArray[np.floating], NDArray[Any]]]
        A results dictionary.
    """

    time_list = np.linspace(0, time_max, N_pts_time)

    # angular grid
    Delta_angle = geometry.compute_max_delta_angle_2d(Burst_Point)
    Delta_angle = 2.0 * Delta_angle
    phi_T_g_list = Burst_Point.phi_g + np.linspace(
        -Delta_angle / 2, Delta_angle / 2, N_pts_phi
    )
    lambd_T_g_list = Burst_Point.lambd_g + np.linspace(
        -Delta_angle / 2, Delta_angle / 2, N_pts_phi
    )

    # geomagnetic field
    geomagnetic_field = MagneticFieldFactory().create(b_field_type)

    # initialize data dictionary
    data_dic: Dict[str, Union[NDArray[np.floating], NDArray[Any]]] = {
        "max_E_norm_at_ground": np.zeros((N_pts_phi, N_pts_lambd)),
        "max_E_theta_at_ground": np.zeros((N_pts_phi, N_pts_lambd)),
        "max_E_phi_at_ground": np.zeros((N_pts_phi, N_pts_lambd)),
        "theta": np.zeros((N_pts_phi, N_pts_lambd), dtype=object),
        "A": np.zeros((N_pts_phi, N_pts_lambd), dtype=object),
        "phi_T_g": np.zeros((N_pts_phi, N_pts_lambd)),
        "lamb_T_g": np.zeros((N_pts_phi, N_pts_lambd)),
    }

    # perform the scan over location
    for i in tqdm(range(len(phi_T_g_list))):
        for j in tqdm(
            range(len(lambd_T_g_list)), leave=bool(i == len(phi_T_g_list) - 1)
        ):
            # update target and midway points
            Target_Point = geometry.Point(
                EARTH_RADIUS, phi_T_g_list[i], lambd_T_g_list[j], "lat/long geo"
            )
            Midway_Point = geometry.get_line_of_sight_midway_point(
                Burst_Point, Target_Point
            )

            try:
                # line of sight check
                # geometry.line_of_sight_check(Burst_Point, Target_Point)

                # Get geometrical quantities
                A_angle = geometry.get_A_angle(Burst_Point, Midway_Point)
                theta_angle = geomagnetic_field.get_theta_angle(
                    point_burst=Burst_Point, point_los=Midway_Point
                )
                Bnorm = geomagnetic_field.get_field_magnitude(Midway_Point)

                # define new EMP model and solve it
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
                    Bnorm=Bnorm,
                )

                sol = model.solver(time_list)
                data_dic["theta"][i, j] = model.theta
                data_dic["A"][i, j] = model.A

            except:
                # print('overshot LOS')
                data_dic["theta"][i, j] = None
                data_dic["A"][i, j] = None
                sol = {
                    "E_norm_at_ground": np.zeros(len(time_list)),
                    "E_theta_at_ground": np.zeros(len(time_list)),
                    "E_phi_at_ground": np.zeros(len(time_list)),
                }

            # store results
            data_dic["phi_T_g"][i, j] = Target_Point.phi_g
            data_dic["lamb_T_g"][i, j] = Target_Point.lambd_g
            data_dic["max_E_norm_at_ground"][i, j] = np.max(sol["E_norm_at_ground"])
            data_dic["max_E_theta_at_ground"][i, j] = np.max(
                np.abs(sol["E_theta_at_ground"])
            )
            data_dic["max_E_phi_at_ground"][i, j] = np.max(
                np.abs(sol["E_phi_at_ground"])
            )

    return data_dic
