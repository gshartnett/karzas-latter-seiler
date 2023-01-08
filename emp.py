'''
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
'''


## imports
import os
import pickle 
import argparse
import numpy as np
from scipy.integrate import quad, solve_ivp
import warnings


## plotting settings
import matplotlib 
import matplotlib.pyplot as plt
from cycler import cycler
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['xtick.minor.size'] = 3.0
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rc('font', family='serif',size=14)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=14)
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', 
                                                       '#348ABD', 
                                                       '#988ED5', 
                                                       '#777777', 
                                                       '#FBC15E', 
                                                       '#8EBA42', 
                                                       '#FFB5B8']
                                                )


## constants of nature
SCALE_HEIGHT = 7 #atmospheric scale height in km (the 7km comes from Seiler pg. 24, paragraph 1, I was using 10.4 previously)
AIR_DENSITY_AT_SEA_LEVEL = 1.293 #air density in kg/m^3
ELECTRON_MASS = 0.511 #electron rest mass in MeV
SPEED_OF_LIGHT = 3 * 1e8 #speed of light in m/s
MEAN_FREE_PATH_AT_SEA_LEVEL = 0.3 #atmospheric mean free path of light at sea-level in km (value taken from KL paper)
ELECTRON_CHARGE = 1.602 * 1e-19 #electric charge in C
VACUUM_PERMEABILITY= 1.257 * 1e-6 #vacuum permeability in H/m
MEV_TO_KG = 1.79 * 1e-30 #conversion factor from MeV/c^2 to kg
KT_TO_MEV = 2.611e+25 #conversion factor from kt to Mev
EARTH_RADIUS = 6378 #Earth's radius in km
ABSORPTION_LAYER_UPPER = 50 #altitude in km of the upper boundary of the absorption layer
ABSORPTION_LAYER_LOWER = 20 #altitude in km of the upper boundary of the absorption layer
PHI_MAGNP = 86.294 * np.pi / 180 #latitude of magnetic North Pole (radians)
LAMBDA_MAGNP = 151.948 * np.pi / 180 #longitude of magnetic North Pole (radians)
B0 = 3.12 * 1e-5 # proportionality constant for the dipole geomagnetic field in Tesla


'''
def _warning(msg, *args, **kwargs):
#    Monkeypatch for warnings: https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
    return str(msg) + '\n'
warnings.warn = _warning
warnings.formatwarning = custom_formatwarning
'''

def Rot(theta, vx, vy, vz):
    '''
    Rotation matrix for angle θ and axis (vx, vy, vz)
    https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_rotation_matrix_to_axis%E2%80%93angle
    '''
    R = np.zeros((3, 3))
    
    R[0,0] = np.cos(theta) + vx**2*(1-np.cos(theta))
    R[0,1] = vx*vy*(1-np.cos(theta)) - vz*np.sin(theta)
    R[0,2] = vx*vz*(1-np.cos(theta)) + vy*np.sin(theta)

    R[1,0] = vx*vy*(1-np.cos(theta)) + vz*np.sin(theta)
    R[1,1] = np.cos(theta) + vy**2*(1-np.cos(theta))
    R[1,2] = vy*vz*(1-np.cos(theta)) - vx*np.sin(theta)

    R[2,0] = vx*vz*(1-np.cos(theta)) - vy*np.sin(theta)
    R[2,1] = vy*vz*(1-np.cos(theta)) + vx*np.sin(theta)
    R[2,2] = np.cos(theta) + vz**2*(1-np.cos(theta))

    return R


ROTATION_MATRIX = Rot(
    np.pi/2 - PHI_MAGNP, 
    -np.sin(LAMBDA_MAGNP), 
    np.cos(LAMBDA_MAGNP),
    0.0
    )

INV_ROTATION_MATRIX = Rot(
    - np.pi/2 + PHI_MAGNP, 
    -np.sin(LAMBDA_MAGNP), 
    np.cos(LAMBDA_MAGNP),
    0.0
    )

def great_circle_distance(phi_1, lambd_1, phi_2, lambd_2):
    '''
    Computes the great-circle distance bewteen two points
    see: http://www.movable-type.co.uk/scripts/latlong.html
    '''
    delta_phi = phi_2 - phi_1
    delta_lambd = lambd_2 - lambd_1
    a = np.sin(delta_phi/2)**2 + np.cos(phi_1)*np.cos(phi_2)*np.sin(delta_lambd/2)**2
    c = 2*np.arctan2( np.sqrt(a), np.sqrt(1-a))
    return EARTH_RADIUS * c


def compute_max_delta_angle_1d(HOB, phi_B, lambd_B, Delta_angle=25*np.pi/180, N_pts=150):
    '''
    Compute the largest delta angle such that a 1d grid
    of lat points deviating from the burst point by at
    most Delta_angle radians will be entirely contained within
    the line-of-sight cone of the burst point.
    '''
    ## perform the scan over location
    cond = False
    while not cond:

        ## build the angular grid
        phi_T_list = phi_B + np.linspace(-Delta_angle/2, Delta_angle/2, N_pts)

        ## perform the scan over location
        try:
            for i in range(len(phi_T_list)):
                r_T = EARTH_RADIUS
                r_B = EARTH_RADIUS + HOB
                line_of_sight_check(r_T, phi_T_list[i], lambd_B, r_B, phi_B, lambd_B)
            cond = True
            return Delta_angle
        except:
            ## decay the delta angle
            Delta_angle = 0.99 * Delta_angle


def compute_max_delta_angle_2d(HOB, phi_B, lambd_B, Delta_angle=25*np.pi/180, N_pts=150):
    '''
    Compute the largest Delta_angle such that a 2d square grid
    of lat/long points deviating from the burst point by at
    most Delta_angle radians will be entirely contained within
    the line-of-sight cone of the burst point.
    '''
    ## perform the scan over location
    cond = False
    while not cond:

        ## build the angular grid
        phi_T_list = phi_B + np.linspace(-Delta_angle/2, Delta_angle/2, N_pts)
        lambd_T_list = lambd_B + np.linspace(-Delta_angle/2, Delta_angle/2, N_pts)

        ## perform the scan over location
        try:
            for i in range(len(phi_T_list)):
                for j in range(len(lambd_T_list)):
                    r_T = EARTH_RADIUS
                    r_B = EARTH_RADIUS + HOB
                    line_of_sight_check(r_T, phi_T_list[i], lambd_T_list[j], r_B, phi_B, lambd_B)
            cond = True
            return Delta_angle
        except:
            ## decay the delta angle
            Delta_angle = 0.99 * Delta_angle
        

def get_geomagnetic_field(r_m, phi_m, lambd_m):
    '''
    Returns the geomagnetic field vector, evaluated at the point 
    (r, phi, lambd) in magnetic coordinates
    '''
    check_geo_coords(phi_m, lambd_m)
    B_r = - 2 * B0 * (EARTH_RADIUS/r_m)**3 * np.sin(phi_m)
    B_phi = B0 * (EARTH_RADIUS/r_m)**3 * np.cos(phi_m)
    B_lambd = 0
    return B_r, B_phi, B_lambd


def get_inclination_angle(phi_m, lambd_m):
    '''
    Returns the inclination angle of the geomagnetic field (in radians),
    evaluated at the point (r, phi, lambd) in magnetic coordinates.
    
    As a check, the angle is computed two ways:
        (1) tan I = |B_r| / |B_phi|
        (2) tan I = 2 tan( |phi| )
    If these two expressions do not agree, an error is raised.
    Computed this way, I is always positive.
    '''
    B_r, B_phi, B_lambd = get_geomagnetic_field(1.0, phi_m, lambd_m)
    expression_1 = np.arctan( np.abs(B_r) / np.abs(B_phi) )
    expression_2 = np.arctan( 2*np.tan( np.abs(phi_m) ) ) 
    #assert expression_1 == expression_2
    return expression_1


def get_geomagnetic_field_Cartesian(r_m, phi_m, lambd_m):
    '''
    Returns the geomagnetic field vector in *geographic* Cartesian coordinates,
    for the point (r, phi, lambd) in magnetic lat/long coordinates.
    '''
    B_r_m, B_phi_m, B_lambd_m = get_geomagnetic_field(r_m, phi_m, lambd_m)
        
    ## unit vectors used to convert from lat/long to Cartesian
    x_hat = np.asarray([1,0,0])
    y_hat = np.asarray([0,1,0])
    z_hat = np.asarray([0,0,1])
    
    ## unit vectors for lat/long coordinates
    r_hat = np.cos(phi_m)*np.cos(lambd_m)*x_hat + np.cos(phi_m)*np.sin(lambd_m)*y_hat + np.sin(phi_m)*z_hat
    phi_hat = -np.sin(phi_m)*np.cos(lambd_m)*x_hat - np.sin(phi_m)*np.sin(lambd_m)*y_hat + np.cos(phi_m)*z_hat
    lambd_hat = -np.sin(lambd_m)*x_hat + np.cos(lambd_m)*y_hat

    ## build the magnetic Cartesian vector for B    
    B_vec_m = B_r_m * r_hat + B_phi_m * phi_hat + B_lambd_m * lambd_hat

    ## convert to the geographic Cartesian vector
    B_vec_g = np.dot(ROTATION_MATRIX, B_vec_m)
    
    return B_vec_g


def theta_angle(HOB, phi_B_g, lambd_B_g, r_S, phi_S_g, lambd_S_g):
    '''
    The angle b/w the line of sight radial vector and the local magnetic field.
    The two sets of coordinates are:
        B: burst
        S: evaluation point
    The evaluation point could be a point on the Earth's surface, but it seems more
    appropriate to pick a point lying in the absorption zone.
    The same geographic lat/long (r,phi,lambda) coordinate system is used for all points.
    '''
    ## convert to magnetic coordinates for the evaluation point S
    phi_S_m, lambd_S_m = geo2mag(phi_S_g, lambd_S_g)   

    ## obtain the magnetic field vector at the evaluation point S
    B_vec_g = get_geomagnetic_field_Cartesian(r_S, phi_S_m, lambd_S_m)

    ## get vector pointing from B to S
    r_B = EARTH_RADIUS + HOB
    X_BS_g = get_X_A_to_B(r_B, phi_B_g, lambd_B_g, r_S, phi_S_g, lambd_S_g)
    
    return np.arccos( np.dot(X_BS_g, B_vec_g) / (np.linalg.norm(X_BS_g) * np.linalg.norm(B_vec_g)) )


def get_X_at_point(r, phi, lambd):
    '''Get the Cartesian vectors for a point'''
    x, y, z = geo2cartesian(1, phi, lambd)
    X = r * np.asarray([x, y, z])
    return X
        
    
def get_X_A_to_B(r_A, phi_A, lambd_A, r_B, phi_B, lambd_B):
    '''
    Compute the vector pointing from A to B.
    '''
    X_A = get_X_at_point(r_A, phi_A, lambd_A)
    X_B = get_X_at_point(r_B, phi_B, lambd_B)    
    X_A_to_B = X_B - X_A
    return X_A_to_B


def line_of_sight_check(r_A, phi_A, lambd_A, r_B, phi_B, lambd_B):
    '''
    Given a burst point B and an arbitrary point A on the Earth's surface, 
    compute the vector pointing from B to A and confirm that this vector's 
    length is less than the length of the tangent vector pointing from B 
    to a point on the surface.
    '''
    HOB = r_B - EARTH_RADIUS
    Amax = np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + HOB) )    
    X_B_to_A = get_X_A_to_B(r_B, phi_B, lambd_B, r_A, phi_A, lambd_A)
    r = np.linalg.norm(X_B_to_A) # distance from B to surface point A
    rmax = (EARTH_RADIUS + HOB) * np.cos(Amax)
    try:
        assert r <= rmax
        #assert r <= HOB / np.cos(Amax) # i think this is wrong! DELETE
    except: 
        raise ValueError('Coordinates have overshot the horizon!')
    return

    
def A_angle_spherical(HOB, theta_B, phi_B, theta_T, phi_T):
    '''
    The angle A between the target-burst and origin-burst vectors.
    The same spherical (r,theta,phi) coordinate system is used for all points.
    '''
    Delta_phi = phi_B - phi_T
    cos_c = np.cos(theta_B)*np.cos(theta_T) + np.sin(theta_B)*np.sin(theta_T)*np.cos(Delta_phi) #spherical law of cosine term
    num = HOB + EARTH_RADIUS - EARTH_RADIUS*cos_c
    den = np.sqrt(HOB**2 + 2*HOB*EARTH_RADIUS + 2*EARTH_RADIUS**2 - 2*EARTH_RADIUS*(HOB + EARTH_RADIUS)*cos_c)    
    return np.arccos(num/den)
    
    
def A_angle_dotproduct(HOB, phi_B, lambd_B, phi_T, lambd_T):
    '''
    The angle A between the target-burst and origin-burst vectors.
    The same lat/long (r,phi,lambda) coordinate system is used for all points.
    Computed directly from the dot product formula.
    '''
    ## get the Cartesian vectors for B
    r_B = HOB + EARTH_RADIUS
    x_B, y_B, z_B = geo2cartesian(1, phi_B, lambd_B)
    X_B = r_B * np.asarray([x_B, y_B, z_B])

    ## get the Cartesian vector for X_B_to_T
    r_T = EARTH_RADIUS
    X_B_to_T = get_X_A_to_B(r_B, phi_B, lambd_B, r_T, phi_T, lambd_T)

    ## compute A. The - sign comes from the fact that X_B = vector from origin to B, but we want B to origin    
    return np.arccos( - np.dot(X_B_to_T, X_B) / (np.linalg.norm(X_B_to_T) * np.linalg.norm(X_B)) )
    
    
def A_angle_latlong(HOB, phi_B, lambd_B, phi_T, lambd_T):
    '''
    The angle A between the target-burst and origin-burst vectors.
    The same lat/long (r,phi,lambda) coordinate system is used for all points.
    '''
    Delta_lambd = lambd_B - lambd_T
    cos_c = np.sin(phi_B)*np.sin(phi_T) + np.cos(phi_B)*np.cos(phi_T)*np.cos(Delta_lambd) #spherical law of cosine term
    num = HOB + EARTH_RADIUS - EARTH_RADIUS*cos_c
    den = np.sqrt(HOB**2 + 2*HOB*EARTH_RADIUS + 2*EARTH_RADIUS**2 - 2*EARTH_RADIUS*(HOB + EARTH_RADIUS)*cos_c)    
    return np.arccos(num/den)


def get_X_midway(HOB, phi_B, lambd_B, phi_T, lambd_T):
    '''
    Get the position vector to a point mid-way between the 
    upper and lower absorption layers, along the line of sight ray.
    '''
    A = A_angle_latlong(HOB, phi_B, lambd_B, phi_T, lambd_T)
    rmin = (HOB - ABSORPTION_LAYER_UPPER) / np.cos(A) #distance from burst point (r=0) to top of absorption layer
    rmax = (HOB - ABSORPTION_LAYER_LOWER) / np.cos(A) #distance from burst point (r=0) to bottom of absorption layer
    
    r_B = EARTH_RADIUS + HOB    
    r_T = EARTH_RADIUS
    
    ## compute the vector from B to T
    X_B_to_T = get_X_A_to_B(r_B, phi_B, lambd_B, r_T, phi_T, lambd_T)
    ## rescale the length to be (rmin+rmax)/2, producing X_B_to_midway
    X_B_to_midway = ( (rmin + rmax) / 2 ) * X_B_to_T / np.linalg.norm(X_B_to_T)
    ## get vector from origin to B
    X_B = get_X_at_point(r_B, phi_B, lambd_B)
    ## combine to get vector from origin to midway
    return X_B + X_B_to_midway

    
def get_latlong_midway(HOB, phi_B, lambd_B, phi_T, lambd_T):
    '''
    Get the lat/long of the mid-way point.
    '''
    X_midway = get_X_midway(HOB, phi_B, lambd_B, phi_T, lambd_T)
    r_midway, phi_midway, lambd_midway = cartesian2geo(X_midway[0], X_midway[1], X_midway[2])
    check_geo_coords(phi_midway, lambd_midway)
    return r_midway, phi_midway, lambd_midway


def spherical2cartesian(r, theta, phi):
    '''
    Convert spherical coordinates (r,θ,ϕ) to Cartesian coordinates (x,y,z)
    ''' 
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cartesian2spherical(x, y, z):
    '''
    Convert Cartesian coordinates (x,y,z) to spherical coordinates (r,θ,ϕ)
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    '''
    if x > 0:
        phi = np.arctan2(y,x)
    elif x < 0 and y >= 0:
        phi = np.arctan2(y,x) + np.pi
    elif x < 0 and y < 0:
        phi = np.arctan2(y,x) - np.pi
    elif x == 0 and y > 0:
        phi = np.pi/2
    elif x == 0 and y < 0:
        phi = -np.pi/2
    else:
        phi = NaN
    '''
    phi = np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))
    return r, theta, phi


def geo2cartesian(r, phi, lambd):
    '''
    Convert geographic coordinates (r,ϕ,λ) to Cartesian coordinates (x,y,z)
    ''' 
    x = r * np.cos(phi) * np.cos(lambd)
    y = r * np.cos(phi) * np.sin(lambd)
    z = r * np.sin(phi)
    return x, y, z


def cartesian2geo(x, y, z):
    '''
    Convert Cartesian coordinates (x,y,z) to geographic coordinates (r,ϕ,λ)
    The choice of a pi offset for y <= 0, as opposed to 2*pi, corresponds to
    lambda \in [-pi, pi].
    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(z , np.sqrt(x**2 + y**2))
    
    if y > 0:
        lambd = np.arccos(x / np.sqrt(x**2 + y**2))
    else:
        lambd = -np.arccos(x / np.sqrt(x**2 + y**2))

    return r, phi, lambd


def mag2geo(phi_m, lambd_m): 
    ''' 
    Convert the magnetic coordinates to geographic coordinates
    ''' 
    x_m, y_m, z_m = geo2cartesian(1, phi_m, lambd_m)
    x_g, y_g, z_g = np.dot(ROTATION_MATRIX, np.asarray([x_m, y_m, z_m]))
    phi_g, lambd_g = cartesian2geo(x_g, y_g, z_g)[1:] 
    return phi_g, lambd_g


def geo2mag(phi_g, lambd_g):
    ''' 
    Convert the geographic coordinates to magnetic coordinates
    ''' 
    x_g, y_g, z_g = geo2cartesian(1, phi_g, lambd_g)
    x_m, y_m, z_m = np.dot(INV_ROTATION_MATRIX, np.asarray([x_g, y_g, z_g]))
    phi_m, lambd_m = cartesian2geo(x_m, y_m, z_m)[1:] 
    return phi_m, lambd_m


def check_geo_coords(phi, lambd):
    '''
    Confirm that the geographic coordinates lie within the proper range.
    '''
    assert (-np.pi/2 <= phi) and (phi <= np.pi/2)
    assert (-np.pi <= lambd) and (lambd < np.pi)
    
    
def check_spherical_coords(theta, phi):
    '''
    Confirm that the spherical coordinates lie within the proper range.
    '''
    assert (0 <= theta) and (theta <= np.pi)
    assert (-np.pi <= phi) and (phi < np.pi)
    

def check_cartesian_geo(rng, Ntrials=10000):
    '''
    Check the cartesian-geo coordinate transforms.
    '''
    TOL = 1e-10

    ## check cartesian -> geo -> cartesian
    for _ in range(Ntrials):
        x, y, z = rng.uniform(low=-10, high=10, size=3)        
        r, phi, lambd = cartesian2geo(x, y, z)
        x2, y2, z2 = geo2cartesian(r, phi, lambd)
        assert (np.abs(x-x2) < TOL) and (np.abs(y-y2) < TOL) and (np.abs(z-z2) < TOL)
        
    ## check geo -> cartesian -> geo
    for _ in range(Ntrials):
        r = 1
        phi = rng.uniform(-np.pi/2, np.pi/2)
        lambd = rng.uniform(-np.pi, np.pi)
        x, y, z = geo2cartesian(r, phi, lambd)
        _, phi2, lambd2 = cartesian2geo(x, y, z)
        assert (np.abs(phi-phi2) < TOL) and (np.abs(lambd-lambd2) < TOL)
        
        
def check_cartesian_spherical(rng, Ntrials=10000):
    '''
    Check the cartesian-spherical coordinate transforms.
    '''
    TOL = 1e-10

    ## check cartesian -> spherical -> cartesian
    for _ in range(Ntrials):
        x, y, z = rng.uniform(low=-10, high=10, size=3)        
        r, theta, phi = cartesian2spherical(x, y, z)
        x2, y2, z2 = spherical2cartesian(r, theta, phi)
        assert (np.abs(x-x2) < TOL) and (np.abs(y-y2) < TOL) and (np.abs(z-z2) < TOL)
        
    ## check spherical -> cartesian -> spherical
    for _ in range(Ntrials):
        r = 1
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(-np.pi, np.pi)
        x, y, z = spherical2cartesian(r, theta, phi)
        _, theta2, phi2 = cartesian2spherical(x, y, z)
        assert (np.abs(theta-theta2) < TOL) and (np.abs(phi-phi2) < TOL)
        
        
def check_mag_geo(rng, Ntrials=10000):
    '''
    Check the mag-geo coordinate transforms
    '''
    TOL = 1e-10
    
    ## check geo -> mag -> geo
    for _ in range(Ntrials):
        phi_g = rng.uniform(-np.pi/2, np.pi/2)
        lambd_g = rng.uniform(-np.pi, np.pi)

        phi_m, lambd_m = geo2mag(phi_g, lambd_g)
        phi_g2, lambd_g2 = mag2geo(phi_m, lambd_m)
        assert (np.abs(phi_g - phi_g2) < TOL) and (np.abs(lambd_g - lambd_g2) < TOL)
        
    ## check mag -> geo -> mag
    for _ in range(Ntrials):
        phi_m = rng.uniform(-np.pi/2, np.pi/2)
        lambd_m = rng.uniform(-np.pi, np.pi)
        phi_g, lambd_g = mag2geo(phi_m, lambd_m)
        phi_m2, lambd_m2 = geo2mag(phi_g, lambd_g)
        assert (np.abs(phi_m - phi_m2) < TOL) and (np.abs(lambd_m - lambd_m2) < TOL)
        
        
class EMPMODEL:
    def __init__(self, 
                 total_yield_kt = 5.0, #total yield in kilotons
                 gamma_yield_fraction = 0.05, #fraction of yield deposited in prompt Gamma radiation
                 Compton_KE = 1.28, #kinetic energy of Compton electron in MeV
                 HOB = 100.0, #heigh of burst in km
                 B = B0, #geomagnetic field strength in Teslas
                 theta = np.pi/2, #angle between line of sight vector and magnetic field
                 A = 0, #angle between radial ray from burst point to target and normal in radians
                 pulse_param_a = 1e7 * 1e-9, #pulse parameter in 1/ns
                 pulse_param_b = 3.7 * 1e8 * 1e-9, #pulse parameter in 1/ns
                 rtol = 1e-4 # relative tolerance for ODE integration
                 ):
        
        ## variable input parameters
        self.total_yield_kt = total_yield_kt
        self.gamma_yield_fraction = gamma_yield_fraction
        self.Compton_KE = Compton_KE
        self.HOB = HOB
        self.B = B
        self.theta = theta
        self.A = A
        self.pulse_param_a = pulse_param_a
        self.pulse_param_b = pulse_param_b
        self.rtol = rtol
        
        ## secondary/derivative parameters
        self.Amax = np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + self.HOB) ) #max A angle (line of sight is tangent to horizon)
        self.R0 = 412 * self.Compton_KE**(1.265 - 0.0954*np.log(self.Compton_KE)) / AIR_DENSITY_AT_SEA_LEVEL * 1e-2 #range of Compton electrons at sea level in m
        self.total_yield_MeV = self.total_yield_kt * KT_TO_MEV #yield in MeV
        self.V0 = SPEED_OF_LIGHT * np.sqrt( (self.Compton_KE**2 + 2*ELECTRON_MASS*self.Compton_KE) / (self.Compton_KE + ELECTRON_MASS)**2 ) #initial Compton velocity in m/s
        self.beta = self.V0 / SPEED_OF_LIGHT #velocity in units of c
        self.gamma = np.sqrt(1/(1 - self.beta**2)) #Lorentz factor
        self.omega = ELECTRON_CHARGE * self.B / ( self.gamma * ELECTRON_MASS * MEV_TO_KG) # units of 1/s
        self.q = self.Compton_KE / (33 * 1e-6) #number of secondary electrons
        self.rmin = (self.HOB - ABSORPTION_LAYER_UPPER) / np.cos(self.A) #distance from burst point (r=0) to top of absorption layer
        self.rmax = (self.HOB - ABSORPTION_LAYER_LOWER) / np.cos(self.A) #distance from burst point (r=0) to bottom of absorption layer
        self.rtarget = self.HOB / np.cos(self.A) #distance from burst point (r=0) to bottom of absorption layer
                
        ## check that the angle A lies in the correct range
        assert (0 <= self.A) or (self.A >= self.Amax)
        
        ## check that the distance to the target lies in the correct range
        
    def RCompton(self, r):
        '''
        The Compton electron stopping distance, in m
        '''
        R = self.R0/self.rho_divided_by_rho0(r)
        return R
    
    def TCompton(self, r):
        '''
        The (scaled) Compton electron lifetime, in ns
        '''
        T = 1e9 * self.RCompton(r) / self.V0
        T = min(T, 1e3) #don't let T exceed 1 μs = 10^3 ns
        T = (1-self.beta)*T
        return T

    def f_pulse(self, t):
        '''
        Normalized gamma pulse.
        Units in 1/ns
        '''
        mask = 1.0 * (t >= 0) #if t<0, define the pulse to be zero
        prefactor = (self.pulse_param_a * self.pulse_param_b) / (self.pulse_param_b - self.pulse_param_a)
        out = mask * prefactor * (np.exp(-self.pulse_param_a * t) - np.exp(-self.pulse_param_b * t))
        return out
    
    def rho_divided_by_rho0(self, r):
        '''
        Ratio of air density at radius r to air density at sea level
        Units are in km
        '''
        return np.exp( -(self.HOB - r*np.cos(self.A)) / SCALE_HEIGHT )
    
    def mean_free_path(self, r):
        '''    
        Mean free path
        Units (of both r and λ) are km
        '''
        return MEAN_FREE_PATH_AT_SEA_LEVEL / self.rho_divided_by_rho0(r)
    
    def gCompton(self, r):
        '''
        The rate function for the creation of primary electrons
        The radius is measured from the burst
        Units are in km^(-3)
        '''
        out = self.gamma_yield_fraction * self.total_yield_MeV / self.Compton_KE / (4 * np.pi * r**2 * self.mean_free_path(r))
        out *= np.exp( - np.exp(-self.HOB/SCALE_HEIGHT) / MEAN_FREE_PATH_AT_SEA_LEVEL * SCALE_HEIGHT / np.cos(self.A) * (np.exp(r * np.cos(self.A)/SCALE_HEIGHT) - 1))
        return out

    def gCompton_numerical_integration(self, r):
        '''
        The rate function for the creation of primary electrons
        The radius is measured from the burst.

        Here g is computed using a numerical integration.
        
        To Do:
        - can we vectorize the numerical integral w/ scipy?
        - should we increase the precision?
        '''
        integral = np.asarray([quad(lambda x: 1/self.mean_free_path(x), 0, ri)[0] for ri in r])
        out = self.gamma_yield_fraction * self.total_yield_MeV / self.Compton_KE * np.exp(-integral) / (4 * np.pi * r**2 * self.mean_free_path(r))
        return out
    
    def electron_collision_freq_at_sea_level(self, E, t):
        '''
        Electron collision frequency at sea level, in 1/ns.
        Defined in Seiler eq 55.
        I modified the expression so t is in units of ns.
        '''
        nu_1 = (-2.5 * 1e20 * 1e-9*t) + (4.5 * 1e12)
        if E < 5e4:
            nu_2 = (0.43 * 1e8)*E + (1.6 * 1e12)
        else:
            nu_2 = (6 * 1e7)*E + (0.8 * 1e12)
        nu_3 = 2.8 * 1e12
        nu = min(4.4 * 1e12, max(nu_1, nu_2, nu_3))
        return 1e-9 * nu 
    
    def conductivity(self, r, t, nuC_0):
        '''
        The conductivity computed using Seiler's approximation.
        '''
        T = self.TCompton(r) # integration time, as a function of r, in units of ns
        nuC = nuC_0 * self.rho_divided_by_rho0(r)
        prefactor = (ELECTRON_CHARGE**2 * self.q/ELECTRON_MASS) * self.gCompton(r)/nuC * 1/((self.pulse_param_b - self.pulse_param_a)*T)
        if t < T:
            main_term = (self.pulse_param_a*t - 1 + np.exp(-self.pulse_param_a*t))*(self.pulse_param_b/self.pulse_param_a) - (self.pulse_param_b*t - 1 + np.exp(-self.pulse_param_b*t))*(self.pulse_param_a/self.pulse_param_b)
        else:
            main_term = (self.pulse_param_b/self.pulse_param_a)*(self.pulse_param_a*T - 1 + np.exp(-self.pulse_param_a*T) - (np.exp(self.pulse_param_a*T) - 1)*(np.exp(-self.pulse_param_a*t) - np.exp(-self.pulse_param_a*T)))
            main_term -= (self.pulse_param_a/self.pulse_param_b)*(self.pulse_param_b*T - 1 + np.exp(-self.pulse_param_b*T) - (np.exp(self.pulse_param_b*T) - 1)*(np.exp(-self.pulse_param_b*t) - np.exp(-self.pulse_param_b*T)))
        units_conversion_factor = 1/MEV_TO_KG * (1/1000)**3 * (1e-9)
        return units_conversion_factor * prefactor * main_term
    
    def JCompton_theta(self, r, t):
        '''
        The polar angle component of the Compton current
        computed using Seiler's approximation
        '''
        T = self.TCompton(r) # integration time, as a function of z, in units of ns
        prefactor = ELECTRON_CHARGE * self.gCompton(r) * np.sin(2*self.theta) * (self.omega**2/4) 
        prefactor *= self.V0/(self.pulse_param_b-self.pulse_param_a) * (1-self.beta)**(-3)
        if t < T:
            main_term = ((self.pulse_param_a*t)**2 - 2*self.pulse_param_a*t + 2 - 2*np.exp(-self.pulse_param_a*t))*(self.pulse_param_b/self.pulse_param_a**2)
            main_term -= ((self.pulse_param_b*t)**2 - 2*self.pulse_param_b*t + 2 - 2*np.exp(-self.pulse_param_b*t))*(self.pulse_param_a/self.pulse_param_b**2)
        else:
            main_term = np.exp(-self.pulse_param_a*t)*(np.exp(self.pulse_param_a*T)*((self.pulse_param_a*T)**2 - 2*self.pulse_param_a*T + 2) - 2)*(self.pulse_param_b/self.pulse_param_a**2)
            main_term -= np.exp(-self.pulse_param_b*t)*(np.exp(self.pulse_param_b*T)*((self.pulse_param_b*T)**2 - 2*self.pulse_param_b*T + 2) - 2)*(self.pulse_param_a/self.pulse_param_b**2)
        units_conversion_factor = 1e-27
        return units_conversion_factor * prefactor * main_term
    
    def JCompton_phi(self, r, t):
        '''
        The azimuthal angle component of the Compton current
        computed using Seiler's approximation
        '''
        T = self.TCompton(r) # integration time, as a function of z, in units of ns
        prefactor = - ELECTRON_CHARGE * self.gCompton(r) * np.sin(self.theta) * self.omega * self.V0/(self.pulse_param_b-self.pulse_param_a) * (1-self.beta)**(-2)
        if t < T:
            main_term = (self.pulse_param_a*t - 1 + np.exp(-self.pulse_param_a*t))*(self.pulse_param_b/self.pulse_param_a) - (self.pulse_param_b*t - 1 + np.exp(-self.pulse_param_b*t))*(self.pulse_param_a/self.pulse_param_b)
        else:
            main_term = np.exp(-self.pulse_param_a*t)*(np.exp(self.pulse_param_a*T)*(self.pulse_param_a*T - 1) + 1)*(self.pulse_param_b/self.pulse_param_a) - np.exp(-self.pulse_param_b*t)*(np.exp(self.pulse_param_b*T)*(self.pulse_param_b*T - 1) + 1)*(self.pulse_param_a/self.pulse_param_b)
        units_conversion_factor = 1e-18
        return units_conversion_factor * prefactor * main_term
    
    def F_theta_Seiler(self, E, r, t, nuC_0):
        '''
        The theta-component of the Maxwell equations. 
        Expressed as dE/dr = F(E)
        '''
        return -E/r - (1e3 * VACUUM_PERMEABILITY * SPEED_OF_LIGHT / 2) * (self.JCompton_theta(r, t) + self.conductivity(r, t, nuC_0(r))*E)

    def F_phi_Seiler(self, E, r, t, nuC_0):
        '''
        The theta-component of the Maxwell equations. 
        Expressed as dE/dr = F(E)
        '''
        return -E/r - (1e3 * VACUUM_PERMEABILITY * SPEED_OF_LIGHT / 2) * (self.JCompton_phi(r, t) + self.conductivity(r, t, nuC_0(r))*E)

    def ODE_solve(self, t, nuC_0):
        '''
        Solve the angular components of the KL ODEs.
                
        To Do: consider adding radial component
        '''
        sol_theta = solve_ivp(lambda r, e: self.F_theta_Seiler(e, r, t, nuC_0), 
                              [self.rmin, self.rmax], 
                              [0],
                              rtol=self.rtol)
        sol_phi = solve_ivp(lambda r, e: self.F_phi_Seiler(e, r, t, nuC_0), 
                            [self.rmin, self.rmax], 
                            [0],
                            rtol=self.rtol)
        return sol_theta, sol_phi
    
    def solver(self, tlist):
        '''
        Solve the KL equations for a range of retarded times and 
        return the angular components of the electric field for 
        r = r_target (at the Earth's surface).
        '''
        
        out = {'tlist':tlist, 'E_theta_at_ground':[], 'E_phi_at_ground':[], 'E_norm_at_ground':[]}
        rlist = np.linspace(self.rmin, self.rmax, 200)
        #E_norm_at_rmax = 0.0
        E_norm_interp = lambda x: 0.0
        
        for t in tlist:
            ## compute the electron collision freq.
            #nuC_0 = self.electron_collision_freq_at_sea_level(E_norm_at_rmax * self.rmax/self.rtarget, t)
            nuC_0_points = np.asarray([self.electron_collision_freq_at_sea_level(E_norm_interp(r), t) for r in rlist])
            nuC_0 = lambda x: np.interp(x, rlist, nuC_0_points)
            
            ## solve the KL equations
            sol_theta, sol_phi = self.ODE_solve(t, nuC_0)

            ## build an interpolation of E_norm(r)
            E_theta_interp = lambda x: np.interp(x, sol_theta.t, sol_theta.y[0])
            E_phi_interp = lambda x: np.interp(x, sol_phi.t, sol_phi.y[0])
            E_norm_interp = lambda x: np.sqrt(E_theta_interp(x)**2 + E_phi_interp(x)**2)

            ## record the value at rmax
            out['E_theta_at_ground'].append(sol_theta.y[0,-1] * self.rmax/self.rtarget)
            out['E_phi_at_ground'].append(sol_phi.y[0,-1] * self.rmax/self.rtarget)
            out['E_norm_at_ground'].append(np.sqrt(sol_theta.y[0,-1]**2 + sol_phi.y[0,-1]**2) * self.rmax/self.rtarget)
            
        ## check that the time of max EMP intensity is not the last time considered
        i_max = max(np.argmax(np.abs(out['E_theta_at_ground'])), np.argmax(np.abs(out['E_phi_at_ground'])))
        if i_max == len(tlist) - 1:
            warnings.warn("Warning, evolution terminated before max EMP intensity has been reached.")

        return out


## solve the model for a single line-of-sight integration
if __name__ == "__main__":

    ## argument parsing
    model_default = EMPMODEL()   
    parser = argparse.ArgumentParser(description='Compute the surface EMP intensity using the Karzas-Latter-Seiler model')

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
        '-B', 
        default=model_default.B, 
        type=float, 
        help='Local value of the geomagnetic field [T]'
        )
    
    parser.add_argument(
        '-theta', 
        default=model_default.theta, 
        type=float, 
        help='Angle between the line-of-sight vector and the geomagnetic field'
        )
        
    parser.add_argument(
        '-A', 
        default=model_default.A, 
        type=float, 
        help='Angle between the line-of-sight vector and the vector normal to the surface of the Earth'
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

    args = vars(parser.parse_args())

    ## define the model
    model = EMPMODEL(
        HOB=args['HOB'], 
        Compton_KE=args['Compton_KE'],
        total_yield_kt=args['total_yield_kt'],
        gamma_yield_fraction=args['gamma_yield_fraction'],
        B=args['B'],
        A=args['A'],
        theta=args['theta'],
        pulse_param_a=args['pulse_param_a'],
        pulse_param_b=args['pulse_param_b'],
        rtol=args['rtol']
        )

    ## print out param values
    print('\nModel Parameters\n--------------------')
    for key, value in model.__dict__.items():
        print(key, '=', value)
    print('\n')

    ## perform the integration
    sol = model.solver(np.linspace(0, 50, 200))   

    ## create data and figure directories
    if not os.path.exists('data'):
       os.makedirs('data')
    if not os.path.exists('figures'):
       os.makedirs('figures')
   
    ## save the result
    with open('data/emp_solution.pkl', 'wb') as f:
        pickle.dump(sol, f)

    ## plot the result
    fig, ax = plt.subplots(1, figsize=(7,5))
    ax.plot(sol['tlist'], sol['E_norm_at_ground'], '-', color='k', linewidth=1.5, markersize=2)
    ax.set_xlabel(r'$\tau$ \ [ns]')
    ax.set_ylabel('E \ [V/m]')
    plt.minorticks_on()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid(alpha=0.5)
    plt.title('Surface EMP Intensity')
    plt.savefig('figures/emp_intensity.png', bbox_inches='tight', dpi=600)
    plt.show()