'''
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
'''

import numpy as np
from constants import * 


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


## build the rotation matrices
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


class Point:
    '''
    A point class used to handle the many different coordinate
    transformations used.
    '''
    def __init__(self, coord1, coord2, coord3, coordsys, consistency_check=True):
        '''Initialize by recording the point in all 4 coordinate systems.'''

        assert coordsys in ['lat/long geo', 'cartesian geo', 'lat/long mag', 'cartesian mag']

        ## point initially defined using geographic coords
        if 'geo' in coordsys:
            if coordsys == 'lat/long geo':
                self.r_g, self.phi_g, self.lambd_g = coord1, coord2, coord3
                self.x_g, self.y_g, self.z_g = latlong2cartesian(self.r_g, self.phi_g, self.lambd_g)
            elif coordsys == 'cartesian geo':
                self.x_g, self.y_g, self.z_g = coord1, coord2, coord3
                self.r_g, self.phi_g, self.lambd_g = cartesian2latlong(self.x_g, self.y_g, self.z_g)
            ## convert to magnetic coords
            self.r_m, self.phi_m, self.lambd_m = latlong_geo2mag(self.r_g, self.phi_g, self.lambd_g)
            self.x_m, self.y_m, self.z_m = latlong2cartesian(self.r_m, self.phi_m, self.lambd_m)

        ## point initially defined using magnetic coords
        else:
            if coordsys == 'lat/long mag':
                self.r_m, self.phi_m, self.lambd_m = coord1, coord2, coord3
                self.x_m, self.y_m, self.z_m = latlong2cartesian(self.r_m, self.phi_m, self.lambd_m)
            elif coordsys == 'cartesian mag':
                self.x_m, self.y_m, self.z_m = coord1, coord2, coord3
                self.r_m, self.phi_m, self.lambd_m = cartesian2latlong(self.x_m, self.y_m, self.z_m)
            ## convert to geographic coords
            self.r_g, self.phi_g, self.lambd_g = latlong_mag2geo(self.r_m, self.phi_m, self.lambd_m)
            self.x_g, self.y_g, self.z_g = latlong2cartesian(self.r_g, self.phi_g, self.lambd_g)
            
        ## check the coordinates
        check_latlong_coords(self.r_g, self.phi_g, self.lambd_g)
        check_latlong_coords(self.r_m, self.phi_m, self.lambd_m)
        
        ## confirm that the coordinates are all consistent
        if consistency_check:
            A_latlong_geo = Point(self.r_g, self.phi_g, self.lambd_g, 'lat/long geo', consistency_check=False)
            A_cartesian_geo = Point(self.x_g, self.y_g, self.z_g, 'cartesian geo', consistency_check=False)
            A_latlong_mag = Point(self.r_m, self.phi_m, self.lambd_m, 'lat/long mag', consistency_check=False)
            A_cartesian_mag = Point(self.x_m, self.y_m, self.z_m, 'cartesian mag', consistency_check=False)

            assert are_two_points_equal(A_latlong_geo, A_cartesian_geo)
            assert are_two_points_equal(A_latlong_mag, A_cartesian_mag)
            assert are_two_points_equal(A_latlong_geo, A_latlong_mag)


def check_latlong_coords(r, phi, lambd):
    '''
    Confirm that the (r, phi, lambd) coordinates lie within the proper range.
    '''
    assert r >= 0
    assert (-np.pi/2 <= phi) and (phi <= np.pi/2)
    assert (-np.pi <= lambd) and (lambd < np.pi)
        
        
def latlong_geo2mag(r_g, phi_g, lambd_g):
    ''' 
    Convert the geographic lat/long to magnetic lat/long
    ''' 
    x_g, y_g, z_g = latlong2cartesian(r_g, phi_g, lambd_g)
    x_m, y_m, z_m = np.dot(INV_ROTATION_MATRIX, np.asarray([x_g, y_g, z_g]))
    r_m, phi_m, lambd_m = cartesian2latlong(x_m, y_m, z_m) 
    return r_m, phi_m, lambd_m


def latlong_mag2geo(r_m, phi_m, lambd_m): 
    ''' 
    Convert the geographic lat/long to magnetic lat/long
    ''' 
    x_m, y_m, z_m = latlong2cartesian(r_m, phi_m, lambd_m)
    x_g, y_g, z_g = np.dot(ROTATION_MATRIX, np.asarray([x_m, y_m, z_m]))
    r_g, phi_g, lambd_g = cartesian2latlong(x_g, y_g, z_g) 
    return r_g, phi_g, lambd_g


def check_spherical_coords(theta, phi):
    '''
    Confirm that the spherical coordinates lie within the proper range.
    No longer used.
    '''
    assert (0 <= theta) and (theta <= np.pi)
    assert (-np.pi <= phi) and (phi < np.pi)
    
    
def latlong2cartesian(r, phi, lambd):
    '''
    Convert lat/long coordinates (r,ϕ,λ) to Cartesian coordinates
    (x,y,z)
    ''' 
    x = r * np.cos(phi) * np.cos(lambd)
    y = r * np.cos(phi) * np.sin(lambd)
    z = r * np.sin(phi)
    return x, y, z


def cartesian2latlong(x, y, z):
    '''
    Convert Cartesian coordinates (x,y,z) to lat/long coordinates 
    (r,ϕ,λ). The choice of a pi offset for y <= 0, as opposed to 2*pi,
    corresponds to lambda \in [-pi, pi].
    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(z , np.sqrt(x**2 + y**2))
    if y > 0:
        lambd = np.arccos(x / np.sqrt(x**2 + y**2))
    else:
        lambd = -np.arccos(x / np.sqrt(x**2 + y**2))
    return r, phi, lambd
    
    
def spherical2cartesian(r, theta, phi):
    '''
    Convert spherical coordinates (r,θ,ϕ) to Cartesian coordinates (x,y,z).
    No longer used.
    ''' 
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cartesian2spherical(x, y, z):
    '''
    Convert Cartesian coordinates (x,y,z) to spherical coordinates (r,θ,ϕ)
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
    No longer used.
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
        
        
def are_two_points_equal(A: Point, B: Point):
    '''Return true if A and B are the same point.'''
    TOL = 1e-6
    #print( [np.abs(A.__dict__[key] - B.__dict__[key]) for key in A.__dict__.keys()] )
    return np.alltrue([np.abs(A.__dict__[key] - B.__dict__[key]) < TOL for key in A.__dict__.keys()])


def check_cartesian_geo(rng, Ntrials=10000):
    '''
    Check the cartesian-geo coordinate transforms.
    '''
    TOL = 1e-10

    ## check cartesian -> geo -> cartesian
    for _ in range(Ntrials):
        x, y, z = rng.uniform(low=-10, high=10, size=3)        
        r, phi, lambd = cartesian2latlong(x, y, z)
        x2, y2, z2 = latlong2cartesian(r, phi, lambd)
        assert (np.abs(x-x2) < TOL) and (np.abs(y-y2) < TOL) and (np.abs(z-z2) < TOL)
        
    ## check geo -> cartesian -> geo
    for _ in range(Ntrials):
        r = 1
        phi = rng.uniform(-np.pi/2, np.pi/2)
        lambd = rng.uniform(-np.pi, np.pi)
        x, y, z = latlong2cartesian(r, phi, lambd)
        _, phi2, lambd2 = cartesian2latlong(x, y, z)
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

    
def get_Xvec_g_from_A_to_B(A: Point, B: Point):
    '''
    Compute the vector pointing from A to B in 
    geographic cartesian coordinates.
    '''
    Xvec_g_from_O_to_A = np.asarray([A.x_g, A.y_g, A.z_g])
    Xvec_g_from_O_to_B = np.asarray([B.x_g, B.y_g, B.z_g])
    return Xvec_g_from_O_to_B - Xvec_g_from_O_to_A


def great_circle_distance(A: Point, B: Point):
    '''
    Computes the great-circle distance bewteen points A, B
    on the sphere.
    see: http://www.movable-type.co.uk/scripts/latlong.html
    '''
    delta_phi = B.phi_g - A.phi_g
    delta_lambd = B.lambd_g - A.lambd_g
    a = np.sin(delta_phi/2)**2 + np.cos(A.phi_g)*np.cos(B.phi_g)*np.sin(delta_lambd/2)**2
    c = 2*np.arctan2( np.sqrt(a), np.sqrt(1-a))
    return EARTH_RADIUS * c


def get_geomagnetic_field_latlong(A: Point):
    '''
    Returns the geomagnetic field vector, evaluated at the point 
    (r, phi, lambd) in magnetic coordinates
    '''
    B_r = - 2 * B0 * (EARTH_RADIUS/A.r_m)**3 * np.sin(A.phi_m)
    B_phi = B0 * (EARTH_RADIUS/A.r_m)**3 * np.cos(A.phi_m)
    B_lambd = 0
    return B_r, B_phi, B_lambd


def get_geomagnetic_field_cartesian(A: Point):
    '''
    Returns the geomagnetic field vector in *geographic* cartesian coordinates,
    for the point (r, phi, lambd) in magnetic lat/long coordinates.
    '''
    B_r_m, B_phi_m, B_lambd_m = get_geomagnetic_field_latlong(A)
        
    ## unit vectors used to convert from lat/long to Cartesian
    x_hat = np.asarray([1,0,0])
    y_hat = np.asarray([0,1,0])
    z_hat = np.asarray([0,0,1])
    
    ## unit vectors for lat/long coordinates
    r_hat = np.cos(A.phi_m)*np.cos(A.lambd_m)*x_hat + np.cos(A.phi_m)*np.sin(A.lambd_m)*y_hat + np.sin(A.phi_m)*z_hat
    phi_hat = -np.sin(A.phi_m)*np.cos(A.lambd_m)*x_hat - np.sin(A.phi_m)*np.sin(A.lambd_m)*y_hat + np.cos(A.phi_m)*z_hat
    lambd_hat = -np.sin(A.lambd_m)*x_hat + np.cos(A.lambd_m)*y_hat

    ## build the magnetic Cartesian vector for B    
    B_vec_m = B_r_m * r_hat + B_phi_m * phi_hat + B_lambd_m * lambd_hat

    ## convert to the geographic Cartesian vector
    B_vec_g = np.dot(ROTATION_MATRIX, B_vec_m)
    
    return B_vec_g


def get_geomagnetic_field_norm(A: Point):
    '''
    Returns the geomagnetic field norm at point A.
    '''
    expression_1 = B0 * (EARTH_RADIUS/A.r_m)**3 * np.sqrt(1 + 3*np.sin(A.phi_m)**2)
    expression_2 = np.linalg.norm( get_geomagnetic_field_cartesian(A) )
    assert np.abs(expression_1 - expression_2) < 1e-10
    return expression_1


def get_inclination_angle(A: Point):
    '''
    Returns the inclination angle of the geomagnetic field (in radians),
    evaluated at the point (r, phi, lambd) in magnetic coordinates.
    
    As a check, the angle is computed two ways:
        (1) tan I = |B_r| / |B_phi|
        (2) tan I = 2 tan( |phi| )
    If these two expressions do not agree, an error is raised.
    Computed this way, I is always positive.
    '''
    B_r, B_phi, _ = get_geomagnetic_field_latlong(A)
    expression_1 = np.arctan( np.abs(B_r) / np.abs(B_phi) )
    expression_2 = np.arctan( 2*np.tan( np.abs(A.phi_m) ) ) 
    assert expression_1 == expression_2
    return expression_1


def get_theta(B: Point, S: Point):
    '''
    The angle b/w the line of sight radial vector and the local magnetic field.
    The two points are:
        B: burst
        S: evaluation point along the line of sight
    The evaluation point could be a point on the Earth's surface, but it seems more
    appropriate to pick a point lying in the absorption zone.
    '''
    B_vec_g = get_geomagnetic_field_cartesian(S)
    Xvec_g_from_B_to_S = get_Xvec_g_from_A_to_B(B, S)    
    num = np.dot(Xvec_g_from_B_to_S, B_vec_g)
    den = np.linalg.norm(Xvec_g_from_B_to_S) * np.linalg.norm(B_vec_g)
    return np.arccos(num/den)


def get_A(B: Point, S: Point):
    '''
    The angle A b/w the line of sight radial vector and the 
    vertical/origin-burst vector.
    The two points are:
        B: burst
        S: evaluation point along the line of sight
    '''
    Xvec_g_from_O_to_B = np.asarray([B.x_g, B.y_g, B.z_g])
    Xvec_g_from_B_to_S = get_Xvec_g_from_A_to_B(B, S) 
    num = - np.dot(Xvec_g_from_B_to_S, Xvec_g_from_O_to_B)
    den = np.linalg.norm(Xvec_g_from_B_to_S) * np.linalg.norm(Xvec_g_from_O_to_B)
    ## A = 0 sometimes fails due to rounding errors in the coord conversions
    #print(num, den, num/den)
    if num/den > 1 and num/den < 1 + 1e-5:
        return 0
    else:
        return np.arccos(num/den)


def get_line_of_sight_midway_point(B: Point, T: Point):
    '''
    Get the position vector in geographic cartesian coordinates 
    to a point mid-way between the upper and lower absorption layers, 
    along the line of sight ray.
    The two points are:
        B: burst
        T: target
    '''
    A = get_A(B, T)
    HOB = np.linalg.norm(np.asarray([B.x_g, B.y_g, B.z_g])) - EARTH_RADIUS
    rmin = (HOB - ABSORPTION_LAYER_UPPER) / np.cos(A) #distance from burst point (r=0) to top of absorption layer
    rmax = (HOB - ABSORPTION_LAYER_LOWER) / np.cos(A) #distance from burst point (r=0) to bottom of absorption layer
    Xvec_g_from_B_to_T = get_Xvec_g_from_A_to_B(B, T)  # compute the vector from B to T
    Xvec_g_from_B_to_M = ( (rmin + rmax) / 2 ) * Xvec_g_from_B_to_T / np.linalg.norm(Xvec_g_from_B_to_T) # rescale the length to be (rmin+rmax)/2, producing X_B_to_midway
    Xvec_g_from_O_to_B = np.asarray([B.x_g, B.y_g, B.z_g]) # get vector from origin to B
    Xvec_g_from_O_to_M = Xvec_g_from_O_to_B + Xvec_g_from_B_to_M
    M = Point(Xvec_g_from_O_to_M[0], Xvec_g_from_O_to_M[1], Xvec_g_from_O_to_M[2], coordsys='cartesian geo')
    return M


def line_of_sight_check(B: Point, T: Point):
    '''
    Given a burst and target point on the Earth's surface, compute 
    the vector pointing from B to T and confirm that this vector's 
    length is less than the length of the tangent vector pointing from B 
    to a point on the surface.
    The two points are:
        B: burst
        T: target
    '''
    HOB = np.linalg.norm(np.asarray([B.x_g, B.y_g, B.z_g])) - EARTH_RADIUS
    Amax = np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + HOB) )    
    rmax = (EARTH_RADIUS + HOB) * np.cos(Amax)
    Xvec_g_B_to_T = get_Xvec_g_from_A_to_B(B, T)
    try:
        ## check distance to target
        assert np.linalg.norm(Xvec_g_B_to_T) <= rmax
        ## check angle A
        HOB = np.linalg.norm(np.asarray([B.x_g, B.y_g, B.z_g])) - EARTH_RADIUS
        Amax = np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + HOB) )
        A = get_A(B, T)
        assert (0 <= A and A <= Amax)
    except:
        raise ValueError('Coordinates have overshot the horizon!')
    return


def compute_max_delta_angle_1d(B: Point, Delta_angle=25*np.pi/180, N_pts=150):
    '''
    Compute the largest delta angle such that a 1d grid
    of lat points deviating from the burst point by at
    most Delta_angle radians will be entirely contained within
    the line-of-sight cone of the burst point.
    Geographic coordinates are assumed.
    '''
    ## perform the scan over location
    while Delta_angle > 1e-3:
        ## build the angular grid and scan
        phi_T_list = B.phi_g + np.linspace(-Delta_angle/2, Delta_angle/2, N_pts)
        try:
            for i in range(len(phi_T_list)):
                ## check whether the target point is within the line of sight
                T = Point(EARTH_RADIUS, phi_T_list[i], B.lambd_g, coordsys='lat/long geo')
                line_of_sight_check(B, T)
            return Delta_angle
        except:
            ## decay the delta angle
            Delta_angle = 0.95 * Delta_angle
            

def compute_max_delta_angle_2d(B: Point, Delta_angle=25*np.pi/180, N_pts=150):
    '''
    Compute the largest Delta_angle such that a 2d square grid
    of lat/long points deviating from the burst point by at
    most Delta_angle radians will be entirely contained within
    the line-of-sight cone of the burst point.
    Geographic coordinates are assumed.    
    '''
    ## perform the scan over location
    while Delta_angle > 1e-3:
        ## build the angular grid and scan
        phi_T_list = B.phi_g + np.linspace(-Delta_angle/2, Delta_angle/2, N_pts)
        lambd_T_list = B.lambd_g + np.linspace(-Delta_angle/2, Delta_angle/2, N_pts)
        try:
            for i in range(len(phi_T_list)):
                for j in range(len(lambd_T_list)):
                    T = Point(EARTH_RADIUS, phi_T_list[i], lambd_T_list[j], coordsys='lat/long geo')
                    line_of_sight_check(B, T)
            return Delta_angle
        except:
            ## decay the delta angle
            Delta_angle = 0.95 * Delta_angle