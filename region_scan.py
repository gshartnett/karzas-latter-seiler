'''
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
'''

## imports
import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import folium
from folium import Map
from folium.plugins import HeatMap
import branca
import pandas as pd
import geojsoncontour
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib
from cycler import cycler

## my imports
from constants import *
import geometry
import emp

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['xtick.minor.size'] = 3.0
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['lines.linewidth'] = 2
plt.rc('font', family='serif',size=16)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=[
    '#E24A33',
    '#348ABD',
    '#988ED5',
    '#777777',
    '#FBC15E',
    '#8EBA42',
    '#FFB5B8'
    ])


def data_dic_to_xyz(data_dic, gaussian_smooth=True, field_type='norm'):
    """
    Convert the data into three lists x, y, z, with
        x - latitude
        y - longitude
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
    if field_type == 'norm':
        field_strength = data_dic['max_E_norm_at_ground']
    elif field_type == 'theta':
        field_strength = data_dic['max_E_theta_at_ground']
    elif field_type == 'phi':
        field_strength = data_dic['max_E_phi_at_ground']

    ## perform gaussian smoothing to make nicer plots
    ## the motivation for this came from this SE post:
    ## https://stackoverflow.com/questions/12274529/how-to-smooth-matplotlib-contour-plot
    if gaussian_smooth:
        field_strength = scipy.ndimage.gaussian_filter(field_strength, 1)

    ## loop over the arrays and extract the points
    for i in range(data_dic['theta'].shape[0]):
        for j in range(data_dic['theta'].shape[1]):
            x.append(data_dic['lamb_T_g'][i,j] * 180/np.pi)
            y.append(data_dic['phi_T_g'][i,j] * 180/np.pi )
            z.append(field_strength[i,j])
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    return x, y, z


def contour_plot(x, y, z, Burst_Point, save_path=None, grid=False):
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

    fig, ax = plt.subplots(figsize=(14,10))

    ## create grid values
    ngrid = 300
    xi = np.linspace(np.min(x), np.max(x), ngrid)
    yi = np.linspace(np.min(y), np.max(y), ngrid)

    ## linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    ## set the field value ot nan for any grid points that are outside the horizon
    for i in range(len(xi)):
        for j in range(len(yi)):
            phi = yi[j] * np.pi/180
            lambd = xi[i] * np.pi/180
            Target_Point = geometry.Point(EARTH_RADIUS, phi, lambd, 'lat/long geo')
            try:
                geometry.line_of_sight_check(Burst_Point, Target_Point)
            except:
                zi[j,i] = 'nan'

    ## other interpolation schemes
    #zi = scipy.interpolate.Rbf(x, y, z, function='linear')(Xi, Yi)

    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    #from scipy.interpolate import griddata
    #zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    ## create the plot
    level_spacing = 5 * 1e3
    levels = [i * level_spacing for i in range(
        int(np.round(np.min(z)/level_spacing)) -1,
        int(np.round(np.max(z)/level_spacing)) +1
        )]
    levels
    contourf = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r", extend='max')
    contour1 = ax.contour(xi, yi, zi, levels=levels, linewidths=1, linestyles='-', colors='k')

    ax.clabel(contour1, inline=1, fontsize=10)

    #ax.clabel(contour1, inline=True, fontsize=8, colors='k')
    fig.colorbar(contourf, ax=ax, label=r'$E$ [V/m]')
    ax.set_xlabel(r'$\lambda$ (longitude) [degrees]', labelpad=10)
    ax.set_ylabel(r'$\phi$ (latitude) [degrees]', labelpad=10)
    #ax.set_title(r'Max EMP Intensity [V/m]')
    ax.grid(grid)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

    return contourf, levels


def folium_plot(contourf, lat0, long0, levels, gz_name):
    """
    Super-impose a contour plot on top of a folium map.

    Parameters
    ----------
    contourf : _type_
        _description_
    lat0 : float
        Latitude of ground zero in degrees.
    long0 : float
        Longitude of ground zero in degrees.
    levels : List[float]
        Levels for the color map.
    min_value : float
        Minimum field value, in V/m.
    gz_name : str
        Name of ground zero, used for saving.

    Returns
    -------
    _type_
        A folium geomap object.
    """

    ## convert matplotlib contourf to geojson
    ## can swap out contourf for contour to get a plot with just the contour lines
    geojson = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
        min_angle_deg=3.0,
        ndigits=5,
        stroke_width=1,
        fill_opacity=0.9)
    geojsonf = geojsoncontour.contourf_to_geojson(contourf=contourf)

    ## set up color map
    #cmap = plt.cm.get_cmap('Spectral')
    cmap = plt.cm.get_cmap('RdBu_r')
    colors = [cmap(x) for x in np.linspace(0, 1, len(levels))]
    cm = branca.colormap.LinearColormap(
        colors, vmin=np.min(levels), vmax=np.max(levels)
        ).to_step(index=levels)

    ## tile options: "OpenStreetMap", "Stamen Terrain", "Stamen Toner", "Stamen Watercolor", "CartoDB positron", "CartoDB dark_matter"
    geomap = folium.Map([lat0, long0], width=750, height=750, zoom_start=5, tiles="CartoDB positron")

    ## plot the contour plot ont folium
    # see here for style function params: https://leafletjs.com/reference-1.6.0.html#path-option
    folium.GeoJson(
        geojsonf,
        style_function=lambda x: {
            'color':     'gray', #x['properties']['stroke'], #color of contour lines
            'weight':    1, #x['properties']['stroke-width'], #thickness of contour lines
            'fillColor': x['properties']['fill'], #color in between contour lines
            'opacity':   1, #opacity of contour lines
            'fillOpacity': 0.5,
        }).add_to(geomap)

    ## add the colormap to the folium map
    cm.caption = 'Enorm (V/m)'
    geomap.add_child(cm)

    #feature_group = folium.FeatureGroup("Locations")
    #for lat, lng, name in zip(lat_lst, lng_lst, name_lst):
    #    feature_group.add_child(folium.Marker(location=[lat,lon],popup=name))
    #map.add_child(feature_group)
    geomap.add_child(folium.FeatureGroup("Locations").add_child(folium.Marker(location=[lat0, long0], popup='Ground Zero')))

    ## convert the geomap to a PIL image
    ## see: https://stackoverflow.com/questions/40208051/selenium-using-python-geckodriver-executable-needs-to-be-in-path
    img_data = geomap._to_png(10)
    img = Image.open(io.BytesIO(img_data))

    ## remove any white-space
    ## see: https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
    pix = np.asarray(img)
    idx = np.where(pix-255)[0:2] # Drop the color when finding edges
    box = list(map(min,idx))[::-1] + list(map(max,idx))[::-1]
    region = img.crop(box)
    region_pix = np.asarray(region)
    img2 = Image.fromarray(region_pix)

    ## save
    img2.convert("RGB").save(f'figures/{gz_name}_smile.png', dpi=(900, 900))

    ## display result
    return geomap


def region_scan(
    Burst_Point,
    HOB = DEFAULT_HOB,
    Compton_KE = DEFAULT_Compton_KE,
    total_yield_kt = DEFAULT_total_yield_kt,
    gamma_yield_fraction = DEFAULT_gamma_yield_fraction,
    pulse_param_a = DEFAULT_pulse_param_a,
    pulse_param_b = DEFAULT_pulse_param_b,
    rtol = DEFAULT_rtol,
    N_pts_phi = 20,
    N_pts_lambd = 20,
    time_max = 100.0,
    N_pts_time = 50,
    b_field_type = 'dipole'
    ):
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
    Burst_Point : Point
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
    Dict
        A results dictionary.
    """

    time_list = np.linspace(0, time_max, N_pts_time)

    ## angular grid
    Delta_angle = geometry.compute_max_delta_angle_2d(Burst_Point)
    Delta_angle = 2.0 * Delta_angle
    phi_T_g_list = Burst_Point.phi_g + np.linspace(-Delta_angle/2, Delta_angle/2, N_pts_phi)
    lambd_T_g_list = Burst_Point.lambd_g + np.linspace(-Delta_angle/2, Delta_angle/2, N_pts_phi)

    ## initialize data dictionary
    data_dic = {
        'max_E_norm_at_ground':np.zeros((N_pts_phi, N_pts_lambd)),
        'max_E_theta_at_ground':np.zeros((N_pts_phi, N_pts_lambd)),
        'max_E_phi_at_ground':np.zeros((N_pts_phi, N_pts_lambd)),
        'theta':np.zeros((N_pts_phi, N_pts_lambd)),
        'A':np.zeros((N_pts_phi, N_pts_lambd)),
        'phi_T_g':np.zeros((N_pts_phi, N_pts_lambd)),
        'lamb_T_g':np.zeros((N_pts_phi, N_pts_lambd)),
        }

    ## perform the scan over location
    for i in tqdm(range(len(phi_T_g_list))):

        for j in tqdm(range(len(lambd_T_g_list)), leave=bool(i==len(phi_T_g_list)-1)):

            ## update target and midway points
            Target_Point = geometry.Point(EARTH_RADIUS, phi_T_g_list[i], lambd_T_g_list[j], 'lat/long geo')
            Midway_Point = geometry.get_line_of_sight_midway_point(Burst_Point, Target_Point)

            try:
                ## line of sight check
                #geometry.line_of_sight_check(Burst_Point, Target_Point)

                ## define new EMP model and solve it
                model = emp.EMPMODEL(
                    HOB = HOB,
                    Compton_KE = Compton_KE,
                    total_yield_kt = total_yield_kt,
                    gamma_yield_fraction = gamma_yield_fraction,
                    pulse_param_a = pulse_param_a,
                    pulse_param_b = pulse_param_b,
                    rtol = rtol,
                    A = geometry.get_A(Burst_Point, Midway_Point),
                    theta = geometry.get_theta(Burst_Point, Midway_Point, b_field_type=b_field_type),
                    Bnorm = geometry.get_geomagnetic_field_norm(Midway_Point, b_field_type=b_field_type)
                    )
                sol = model.solver(time_list)
                data_dic['theta'][i,j] = model.theta
                data_dic['A'][i,j] = model.A

            except:
                #print('overshot LOS')
                data_dic['theta'][i,j] = None
                data_dic['A'][i,j] = None
                sol = {
                        'E_norm_at_ground':np.zeros(len(time_list)),
                        'E_theta_at_ground':np.zeros(len(time_list)),
                        'E_phi_at_ground':np.zeros(len(time_list))
                        }

            ## store results
            data_dic['phi_T_g'][i,j] = Target_Point.phi_g
            data_dic['lamb_T_g'][i,j] = Target_Point.lambd_g
            data_dic['max_E_norm_at_ground'][i,j] = np.max(sol['E_norm_at_ground'])
            data_dic['max_E_theta_at_ground'][i,j] = np.max(np.abs(sol['E_theta_at_ground']))
            data_dic['max_E_phi_at_ground'][i,j] = np.max(np.abs(sol['E_phi_at_ground']))

    return data_dic


## solve the model for a single line-of-sight integration
if __name__ == "__main__":

    ## argument parsing
    parser = argparse.ArgumentParser(description='Compute the surface EMP intensity using the Karzas-Latter-Seiler model')

    parser.add_argument(
        '-phi_B_g',
        default=39.05 * np.pi/180,
        type=float,
        help='Burst point latitude [radians]'
        )

    parser.add_argument(
        '-lambd_B_g',
        default=-95.675 * np.pi/180,
        type=float,
        help='Burst point longitude [radians]'
        )

    parser.add_argument(
        '-N_pts_phi',
        default=50,
        type=int,
        help='Number of latitude grid points'
        )

    parser.add_argument(
        '-N_pts_lambd',
        default=50,
        type=int,
        help='Number of longitude grid points'
        )

    parser.add_argument(
        '-time_max',
        default=100.0,
        type=float,
        help='Total longitude angular spread (degrees)'
        )

    parser.add_argument(
        '-N_pts_time',
        default=50,
        type=int,
        help='Total longitude angular spread (degrees)'
        )

    parser.add_argument(
        '-HOB',
        default=DEFAULT_HOB,
        type=float,
        help='Height of burst [km]'
        )

    parser.add_argument(
        '-Compton_KE',
        default=DEFAULT_Compton_KE,
        type=float,
        help='Kinetic energy of Compton electrons [MeV]'
        )

    parser.add_argument(
        '-total_yield_kt',
        default=DEFAULT_total_yield_kt,
        type=float,
        help='Total weapon yield [kt]'
        )

    parser.add_argument(
        '-gamma_yield_fraction',
        default=DEFAULT_gamma_yield_fraction,
        type=float,
        help='Fraction of yield corresponding to prompt gamma rays'
        )

    parser.add_argument(
        '-pulse_param_a',
        default=DEFAULT_pulse_param_a,
        type=float,
        help='Pulse parameter a [ns^(-1)]'
        )

    parser.add_argument(
        '-pulse_param_b',
        default=DEFAULT_pulse_param_b,
        type=float,
        help='Pulse parameter b [ns^(-1)]'
        )

    parser.add_argument(
        '-rtol',
        default=DEFAULT_rtol,
        type=float,
        help='Relative tolerance used in the ODE integration'
        )

    parser.add_argument(
        '-save_str',
        default='',
        type=str,
        help='String used to save results from different data runs'
        )

    parser.add_argument(
        '-b_field_type',
        default='dipole',
        type=str,
        help='Geomagnetic field model (dipole or igrf).'
        )

    args = vars(parser.parse_args())
    save_str = args.pop('save_str')

    ## instantiate the burst point, add it to args, and remove the angles
    Burst_Point = geometry.Point(
        EARTH_RADIUS + args['HOB'],
        args['phi_B_g'],
        args['lambd_B_g'],
        coordsys='lat/long geo'
        )
    args['Burst_Point'] = Burst_Point
    args.pop('phi_B_g')
    args.pop('lambd_B_g')

    ## print out param values
    print('\nModel Parameters\n--------------------')
    for key, value in args.items():
        print(key, '=', value)
    print('\n')

    ## perform the region scan
    data_dic = region_scan(**args)

    ## create data and figure directories
    if not os.path.exists('data'):
       os.makedirs('data')
    if not os.path.exists('figures'):
       os.makedirs('figures')

    ## save the result
    with open('data/region_scan' + save_str + '.pkl', 'wb') as f:
        pickle.dump(data_dic, f)

    x, y, z = data_dic_to_xyz(data_dic)
    contourf = contour_plot(x, y, z, save_path='figures/region_scan' + save_str, ngrid=50, levels=20)