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
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib
from cycler import cycler
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
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8'])


def data_dic_to_xyz(data_dic, gaussian_smooth=True, field_type='norm'):
    '''
    Convert the data into three lists x, y, z, with
        x - latitude
        y - longitude
        z - field strength
    '''
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


def contour_plot(x, y, z, save_path=None, grid=False):
    '''
    Build a contour plot of the x, y, z data.
    Grid interpolation is used (which I don't think is necessary?).
    '''

    fig, ax = plt.subplots(figsize=(7,5))

    ## create grid values
    ngrid = 250
    xi = np.linspace(np.min(x), np.max(x), ngrid)
    yi = np.linspace(np.min(y), np.max(y), ngrid)

    ## linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    ## other interpolation schemes
    #zi = scipy.interpolate.Rbf(x, y, z, function='linear')(Xi, Yi)
    
    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    #from scipy.interpolate import griddata
    #zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    ## create the plot
    level_spacing = 5e3
    levels = [i * level_spacing for i in range(1, 1+int(np.round(np.max(zi)/level_spacing)))]
    contourf = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r", extend='max')
    contour1 = ax.contour(xi, yi, zi, levels=levels, linewidths=1, linestyles='-', colors='k')
    #ax.clabel(contour1, inline=True, fontsize=8, colors='k')
    fig.colorbar(contourf, ax=ax, label=r'$E$ [V/m]')
    ax.set_xlabel(r'$\lambda$ (longitude) [degrees]', labelpad=10)
    ax.set_ylabel(r'$\phi$ (latitude) [degrees]', labelpad=10)
    #ax.set_title(r'Max EMP Intensity [V/m]')
    ax.grid(grid)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    return contourf


## instantiate a default emp model to copy the default param values
model_default = emp.EMPMODEL()   


def region_scan(
    phi_B_g,
    lambd_B_g,
    HOB = model_default.HOB,
    Compton_KE = model_default.Compton_KE,
    total_yield_kt = model_default.total_yield_kt,
    gamma_yield_fraction = model_default.gamma_yield_fraction,
    pulse_param_a = model_default.pulse_param_a,
    pulse_param_b = model_default.pulse_param_b,
    rtol = model_default.rtol,
    N_pts_phi = 20,
    N_pts_lambd = 20,
    time_max = 100.0,
    N_pts_time = 50
    ):

    time_list = np.linspace(0, time_max, N_pts_time)

    emp.check_geo_coords(phi_B_g, lambd_B_g)
    
    ## angular grid
    Delta_angle = emp.compute_max_delta_angle_2d(HOB, phi_B_g, lambd_B_g)
    Delta_angle = 2.0 * Delta_angle
    phi_T_g_list = phi_B_g + np.linspace(-Delta_angle/2, Delta_angle/2, N_pts_phi)
    lambd_T_g_list = lambd_B_g + np.linspace(-Delta_angle/2, Delta_angle/2, N_pts_phi)
    
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

            ## update target coords
            phi_T_g = phi_T_g_list[i] 
            lambd_T_g = lambd_T_g_list[j]         
            emp.check_geo_coords(phi_T_g, lambd_T_g)
            phi_T_m, _ = emp.geo2mag(phi_T_g, lambd_T_g)
            
            try:
                ## define new EMP model and solve it
                model = emp.EMPMODEL(
                    HOB = HOB,
                    Compton_KE = Compton_KE,
                    total_yield_kt = total_yield_kt,
                    gamma_yield_fraction = gamma_yield_fraction,
                    pulse_param_a = pulse_param_a,
                    pulse_param_b = pulse_param_b,
                    rtol = rtol,
                    A = emp.A_angle_latlong(HOB, phi_B_g, lambd_B_g, phi_T_g, lambd_T_g), 
                    theta = emp.theta_angle(HOB, phi_B_g, lambd_B_g, phi_T_g, lambd_T_g, B_eval_point='T'),
                    B = emp.B0 * np.sqrt(1 + 3*np.sin(phi_T_m)**2) #dipole B-field magnitude at target
                    )
                sol = model.solver(time_list)
                data_dic['theta'][i,j] = model.theta
                data_dic['A'][i,j] = model.A
    
            except:
                data_dic['theta'][i,j] = None
                data_dic['A'][i,j] = None
                sol = {
                        'E_norm_at_ground':np.zeros(len(time_list)),
                        'E_theta_at_ground':np.zeros(len(time_list)),
                        'E_phi_at_ground':np.zeros(len(time_list))
                        }
                    
            ## store results
            data_dic['phi_T_g'][i,j] = phi_T_g
            data_dic['lamb_T_g'][i,j] = lambd_T_g
            #data_dic['theta'][i,j] = model.theta
            #data_dic['A'][i,j] = model.A
            data_dic['max_E_norm_at_ground'][i,j] = np.max(sol['E_norm_at_ground'])
            data_dic['max_E_theta_at_ground'][i,j] = np.max(np.abs(sol['E_theta_at_ground']))
            data_dic['max_E_phi_at_ground'][i,j] = np.max(np.abs(sol['E_phi_at_ground']))

    return data_dic

        
## solve the model for a single line-of-sight integration
if __name__ == "__main__":

    ## argument parsing
    model_default = emp.EMPMODEL()   
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
        default=model_default.HOB, 
        type=float, 
        help='Height of burst [km]'
        )

    parser.add_argument(
        '-Compton_KE', 
        default=model_default.Compton_KE, 
        type=float, 
        help='Kinetic energy of Compton electrons [MeV]'
        )

    parser.add_argument(
        '-total_yield_kt', 
        default=model_default.total_yield_kt, 
        type=float, 
        help='Total weapon yield [kt]'
        )

    parser.add_argument(
        '-gamma_yield_fraction', 
        default=model_default.gamma_yield_fraction, 
        type=float, 
        help='Fraction of yield corresponding to prompt gamma rays'
        )

    parser.add_argument(
        '-pulse_param_a', 
        default=model_default.pulse_param_a, 
        type=float, 
        help='Pulse parameter a [ns^(-1)]'
        )

    parser.add_argument(
        '-pulse_param_b', 
        default=model_default.pulse_param_b, 
        type=float, 
        help='Pulse parameter b [ns^(-1)]'
        )

    parser.add_argument(
        '-rtol', 
        default=model_default.rtol, 
        type=float, 
        help='Relative tolerance used in the ODE integration'
        )
    
    parser.add_argument(
        '-save_str', 
        default='', 
        type=str, 
        help='String used to save results from different data runs'
        )

    args = vars(parser.parse_args())
    save_str = args.pop('save_str')
    
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