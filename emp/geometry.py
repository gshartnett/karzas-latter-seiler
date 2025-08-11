"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

from datetime import datetime
from typing import Tuple

import numpy as np
import ppigrf

from emp.constants import *


def get_rotation_matrix(theta: float, axis: np.ndarray) -> np.ndarray:
    """
    Rotation matrix for angle theta and axis (vx, vy, vz).
    https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_rotation_matrix_to_axis%E2%80%93angle

    Parameters
    ----------
    theta : float
        Rotation angle, in radians.

    Returns
    -------
    np.ndarray
        Rotation matrix.
    """
    if len(axis) != 3:
        raise ValueError("Axis must be a 3-vector.")

    # Get the components of the rotation axis
    vec_x, vec_y, vec_z = axis / np.linalg.norm(axis)

    # Build the rotation matrix
    rotation_matrix = np.zeros((3, 3))

    rotation_matrix[0, 0] = np.cos(theta) + vec_x**2 * (1 - np.cos(theta))
    rotation_matrix[0, 1] = vec_x * vec_y * (1 - np.cos(theta)) - vec_z * np.sin(theta)
    rotation_matrix[0, 2] = vec_x * vec_z * (1 - np.cos(theta)) + vec_y * np.sin(theta)

    rotation_matrix[1, 0] = vec_x * vec_y * (1 - np.cos(theta)) + vec_z * np.sin(theta)
    rotation_matrix[1, 1] = np.cos(theta) + vec_y**2 * (1 - np.cos(theta))
    rotation_matrix[1, 2] = vec_y * vec_z * (1 - np.cos(theta)) - vec_x * np.sin(theta)

    rotation_matrix[2, 0] = vec_x * vec_z * (1 - np.cos(theta)) - vec_y * np.sin(theta)
    rotation_matrix[2, 1] = vec_y * vec_z * (1 - np.cos(theta)) + vec_x * np.sin(theta)
    rotation_matrix[2, 2] = np.cos(theta) + vec_z**2 * (1 - np.cos(theta))

    return rotation_matrix


# Build the rotation matrices
ROTATION_AXIS = np.asarray([-np.sin(LAMBDA_MAGNP), np.cos(LAMBDA_MAGNP), 0.0])
ROTATION_MATRIX = get_rotation_matrix(np.pi / 2 - PHI_MAGNP, ROTATION_AXIS)
INV_ROTATION_MATRIX = get_rotation_matrix(-np.pi / 2 + PHI_MAGNP, ROTATION_AXIS)


class Point:
    """
    A point class used to handle the many different coordinate
    transformations used.
    """

    def __init__(self, coord1, coord2, coord3, coordsys, consistency_check=True):
        """
        Initialize a point by recording the point in all 4 coordinate systems.

        Parameters
        ----------
        coord1 : float
            First coordinate.
        coord2 : float
            Second coordinate.
        coord3 : float
            Third coordinate.
        coordsys : str
            Coordinate system. Must be one of
            ['lat/long geo', 'cartesian geo', 'lat/long mag', 'cartesian mag'].
        consistency_check : bool, optional
            Once the point has been transformed to each of the four coordinate systems,
            perform a check by performing all possible coordinate transformations and
            checking that they all agree.
            By default True
        """
        assert coordsys in [
            "lat/long geo",
            "cartesian geo",
            "lat/long mag",
            "cartesian mag",
        ]

        # point initially defined using geographic coords
        if "geo" in coordsys:
            if coordsys == "lat/long geo":
                self.r_g, self.phi_g, self.lambd_g = coord1, coord2, coord3
                self.x_g, self.y_g, self.z_g = latlong2cartesian(
                    self.r_g, self.phi_g, self.lambd_g
                )
            elif coordsys == "cartesian geo":
                self.x_g, self.y_g, self.z_g = coord1, coord2, coord3
                self.r_g, self.phi_g, self.lambd_g = cartesian_to_latlong(
                    self.x_g, self.y_g, self.z_g
                )
            # convert to magnetic coords
            self.r_m, self.phi_m, self.lambd_m = latlong_geo2mag(
                self.r_g, self.phi_g, self.lambd_g
            )
            self.x_m, self.y_m, self.z_m = latlong2cartesian(
                self.r_m, self.phi_m, self.lambd_m
            )

        # point initially defined using magnetic coords
        else:
            if coordsys == "lat/long mag":
                self.r_m, self.phi_m, self.lambd_m = coord1, coord2, coord3
                self.x_m, self.y_m, self.z_m = latlong2cartesian(
                    self.r_m, self.phi_m, self.lambd_m
                )
            elif coordsys == "cartesian mag":
                self.x_m, self.y_m, self.z_m = coord1, coord2, coord3
                self.r_m, self.phi_m, self.lambd_m = cartesian_to_latlong(
                    self.x_m, self.y_m, self.z_m
                )
            # convert to geographic coords
            self.r_g, self.phi_g, self.lambd_g = latlong_mag2geo(
                self.r_m, self.phi_m, self.lambd_m
            )
            self.x_g, self.y_g, self.z_g = latlong2cartesian(
                self.r_g, self.phi_g, self.lambd_g
            )

        # check the coordinates
        validate_latlong_coords(self.r_g, self.phi_g, self.lambd_g)
        validate_latlong_coords(self.r_m, self.phi_m, self.lambd_m)

        # confirm that the coordinates are all consistent
        if consistency_check:
            point_latlong_geo = Point(
                self.r_g,
                self.phi_g,
                self.lambd_g,
                "lat/long geo",
                consistency_check=False,
            )
            point_cartesian_geo = Point(
                self.x_g, self.y_g, self.z_g, "cartesian geo", consistency_check=False
            )
            point_latlong_mag = Point(
                self.r_m,
                self.phi_m,
                self.lambd_m,
                "lat/long mag",
                consistency_check=False,
            )
            point_cartesian_mag = Point(
                self.x_m, self.y_m, self.z_m, "cartesian mag", consistency_check=False
            )

            assert point_latlong_geo == point_cartesian_geo
            assert point_latlong_mag == point_cartesian_mag
            assert point_latlong_geo == point_latlong_mag

    def __str__(self):
        return (
            f"Point(rg={self.r_g}, phi_g={self.phi_g:.4f}, lambd_g={self.lambd_g:.4f})"
        )

    def __eq__(self, other):
        """
        Return true if two points are the same.

        Parameters
        ----------
        other : Point
            Another point to compare with.

        Returns
        -------
        bool
            Are the two points equivalent?
        """
        if not isinstance(other, Point):
            return False

        tol = 1e-6
        return np.all(
            [
                np.abs(self.__dict__[key] - other.__dict__[key]) < tol
                for key in self.__dict__.keys()
            ]
        )

    def __hash__(self):
        """
        Make Point hashable by using a tuple of rounded coordinate values.
        This allows Points to be used in sets and as dictionary keys.
        """
        # Round to avoid floating point precision issues
        return hash(
            (round(self.r_g, 10), round(self.phi_g, 10), round(self.lambd_g, 10))
        )


def validate_latlong_coords(r: float, phi: float, lambd: float) -> None:
    """
    Confirm that the (r, phi, lambd) coordinates lie within the proper
    range.

    Parameters
    ----------
    r : float
        Radius, any units.
    phi : float
        Latitude, in radians.
    lambd : float
        Longitude, in radians.
    """
    assert r >= 0
    assert (-np.pi / 2 <= phi) and (phi <= np.pi / 2)
    assert (-np.pi <= lambd) and (lambd < np.pi)


def validate_spherical_coords(theta: float, phi: float) -> None:
    """
    Check that the spherical coordinates lie within the proper range.
    No longer used.

    Parameters
    ----------
    theta : float
        Polar spherical coordinate, in radians.
    phi : phi
        Azimuthal spherical coordinate, in radians.
    """
    assert (0 <= theta) and (theta <= np.pi)
    assert (-np.pi <= phi) and (phi < np.pi)


def latlong_geo2mag(
    r_g: float, phi_g: float, lambd_g: float
) -> Tuple[float, float, float]:
    """
    Convert the geographic lat/long to magnetic lat/long.

    Parameters
    ----------
    r_g : float
        Radius, any units.
    phi_g : float
        Latitude, in radians.
    lambd_g : float
        Longitude, in radians.

    Returns
    -------
    Tuple[float, float, float]
        The three coordinates.
    """
    x_g, y_g, z_g = latlong2cartesian(r_g, phi_g, lambd_g)
    x_m, y_m, z_m = np.dot(INV_ROTATION_MATRIX, np.asarray([x_g, y_g, z_g]))
    r_m, phi_m, lambd_m = cartesian_to_latlong(x_m, y_m, z_m)
    return r_m, phi_m, lambd_m


def latlong_mag2geo(
    r_m: float, phi_m: float, lambd_m: float
) -> Tuple[float, float, float]:
    """
    Convert the geographic lat/long to magnetic lat/long.

    Parameters
    ----------
    r_m : _type_
        Radius, any units.
    phi_m : _type_
        Latitude, in radians.
    lambd_m : _type_
        Longitude, in radians.

    Returns
    -------
    Tuple[float, float, float]
        The three coordinates.
    """
    x_m, y_m, z_m = latlong2cartesian(r_m, phi_m, lambd_m)
    x_g, y_g, z_g = np.dot(ROTATION_MATRIX, np.asarray([x_m, y_m, z_m]))
    r_g, phi_g, lambd_g = cartesian_to_latlong(x_g, y_g, z_g)
    return r_g, phi_g, lambd_g


def latlong2cartesian(r: float, phi: float, lambd: float) -> Tuple[float, float, float]:
    """
    Convert lat/long coordinates (r,ϕ,λ) to Cartesian coordinates
    (x,y,z)

    Parameters
    ----------
    r : float
        Radius, any units.
    phi : float
       Latitude, in radians.
    lambd : float
       Latitude, in radians.

    Returns
    -------
    Tuple[float, float, float]
        The three Cartesian coordinates. Same units as r.
    """
    x = r * np.cos(phi) * np.cos(lambd)
    y = r * np.cos(phi) * np.sin(lambd)
    z = r * np.sin(phi)
    return x, y, z


def cartesian_to_latlong(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates (x,y,z) to lat/long coordinates
    (r,ϕ,λ). The choice of a pi offset for y <= 0, as opposed to 2*pi,
    corresponds to lambda in [-pi, pi].

    Parameters
    ----------
    x : float
        Cartesian x coordinate, in any units.
    y : float
        Cartesian y coordinate, in any units.
    z : float
        Cartesian z coordinate, in any units.

    Returns
    -------
    Tuple[float, float, float]
        The three coordinates. The radius is measured in the same units
        as x,y,z, and the angles are measured in radians.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))
    if y > 0:
        lambd = np.arccos(x / np.sqrt(x**2 + y**2))
    else:
        lambd = -np.arccos(x / np.sqrt(x**2 + y**2))
    return r, phi, lambd


def spherical_to_cartesian(
    r: float, theta: float, phi: float
) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates (r,θ,ϕ) to Cartesian coordinates (x,y,z).
    No longer used.

    Parameters
    ----------
    r : float
        The radius, in any units.
    theta : float
        Polar coordinate, in radians.
    phi : float
        Azimuthal coordinate, in radians.

    Returns
    -------
    Tuple[float, float, float]
        The three coordinates, measured in the same units as r.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates (x,y,z) to spherical coordinates (r,θ,ϕ)
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
    No longer used.

    Parameters
    ----------
    x : float
        Cartesian x coordinate, in any units.
    y : float
        Cartesian y coordinate, in any units.
    z : float
        Cartesian z coordinate, in any units.

    Returns
    -------
    Tuple[float, float, float]
        The three coordinates. r is measured in same units as x,y,z, inputs,
        and the angles are measured in radians.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    """
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
    """
    phi = np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))
    return r, theta, phi


def get_xvec_g_from_A_to_B(pointA: Point, pointB: Point) -> np.ndarray:
    """
    Compute the vector pointing from A to B in geographic cartesian
    coordinates.

    Parameters
    ----------
    pointA : Point
        Starting point A.
    pointB : Point
        Destination point B.

    Returns
    -------
    np.ndarray
        The vector pointing from point A to point B.
    """
    xvec_g_from_O_to_A = np.asarray([pointA.x_g, pointA.y_g, pointA.z_g])
    xvec_g_from_O_to_B = np.asarray([pointB.x_g, pointB.y_g, pointB.z_g])
    return np.asarray(xvec_g_from_O_to_B - xvec_g_from_O_to_A)


def great_circle_distance(pointA: Point, pointB: Point) -> float:
    """
    Computes the great-circle distance bewteen points A, B
    on the sphere.
    see: http://www.movable-type.co.uk/scripts/latlong.html

    Parameters
    ----------
    pointA : Point
        A point.
    pointB : Point
        A point.

    Returns
    -------
    float
        The great circle distance between points A, B.
    """
    delta_phi = pointB.phi_g - pointA.phi_g
    delta_lambd = pointB.lambd_g - pointA.lambd_g
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(pointA.phi_g) * np.cos(pointB.phi_g) * np.sin(delta_lambd / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(EARTH_RADIUS * c)


def get_geomagnetic_field_latlong(pointA: Point) -> Tuple[float, float, float]:
    """
    Returns the geomagnetic field vector, evaluated at the point
    (r, phi, lambd) in magnetic coordinates.

    Assumes a dipole model for the geomagnetic field.

    Parameters
    ----------
    pointA : Point
        The evaluation point.

    Returns
    -------
    Tuple[float, float, float]
       The geomagnetic field vector.
    """
    B_r = -2 * B0 * (EARTH_RADIUS / pointA.r_m) ** 3 * np.sin(pointA.phi_m)
    B_phi = B0 * (EARTH_RADIUS / pointA.r_m) ** 3 * np.cos(pointA.phi_m)
    B_lambd = 0
    return B_r, B_phi, B_lambd


def get_geomagnetic_field_cartesian_dipole(point: Point) -> np.ndarray:
    """
    Returns the geomagnetic field vector in *geographic* cartesian
    coordinates, for the point (r, phi, lambd) in magnetic lat/long
    coordinates.

    Assumes a dipole model for the geomagnetic field.

    Parameters
    ----------
    point : Point
        The evaluation point.

    Returns
    -------
    ndarray
        The geomagnetic field as a Cartesian vector.
    """
    B_r_m, B_phi_m, B_lambd_m = get_geomagnetic_field_latlong(point)

    # unit vectors used to convert from lat/long to Cartesian
    x_hat = np.asarray([1, 0, 0])
    y_hat = np.asarray([0, 1, 0])
    z_hat = np.asarray([0, 0, 1])

    # unit vectors for lat/long coordinates
    r_hat = (
        np.cos(point.phi_m) * np.cos(point.lambd_m) * x_hat
        + np.cos(point.phi_m) * np.sin(point.lambd_m) * y_hat
        + np.sin(point.phi_m) * z_hat
    )
    phi_hat = (
        -np.sin(point.phi_m) * np.cos(point.lambd_m) * x_hat
        - np.sin(point.phi_m) * np.sin(point.lambd_m) * y_hat
        + np.cos(point.phi_m) * z_hat
    )
    lambd_hat = -np.sin(point.lambd_m) * x_hat + np.cos(point.lambd_m) * y_hat

    # build the magnetic Cartesian vector for B
    B_vec_m = B_r_m * r_hat + B_phi_m * phi_hat + B_lambd_m * lambd_hat

    # convert to the geographic Cartesian vector
    B_vec_g = np.dot(ROTATION_MATRIX, B_vec_m)

    return np.asarray(B_vec_g)


def get_geomagnetic_field_cartesian_igrf(point: Point) -> np.ndarray:
    """
    Returns the IGRF geomagnetic field vector in geographic cartesian
    coordinates, for the point (r, phi, lambd) in geographic lat/long
    coordinates.

    The ppigrf package is used, which uses geographic spherical coordinates
    (r, theta, phi), with the convention that the angles are measured in
    degrees, not radians. Care must be taken to not confuse the azimuthal
    phi angle here with the latitude phi angle. Therefore, within this
    function phi_azimuthal will be used.

    Parameters
    ----------
    point : Point
        The evaluation point.

    Returns
    -------
    ndarray
        The geomagnetic field as a Cartesian vector.
    """

    theta = np.pi / 2 - point.phi_g
    phi_azimuthal = point.lambd_g

    B_r, B_theta, B_phi_azimuthal = ppigrf.igrf_gc(
        point.r_g, 180 / np.pi * theta, 180 / np.pi * phi_azimuthal, datetime.today()
    )
    B_r = B_r[0]
    B_theta = B_theta[0]
    B_phi_azimuthal = B_phi_azimuthal[0]

    # unit vectors used to convert from (r, theta, phi)) to Cartesian
    x_hat = np.asarray([1, 0, 0])
    y_hat = np.asarray([0, 1, 0])
    z_hat = np.asarray([0, 0, 1])

    # unit vectors for lat/long coordinates
    r_hat = (
        np.sin(theta) * np.cos(phi_azimuthal) * x_hat
        + np.sin(theta) * np.sin(phi_azimuthal) * y_hat
        + np.cos(theta) * z_hat
    )

    theta_hat = (
        np.cos(theta) * np.cos(phi_azimuthal) * x_hat
        + np.cos(theta) * np.sin(phi_azimuthal) * y_hat
        - np.sin(theta) * z_hat
    )

    phi_azimuthal_hat = -np.sin(phi_azimuthal) * x_hat + np.cos(phi_azimuthal) * y_hat

    # build the magnetic Cartesian vector for B
    B_vec_g = np.asarray(
        B_r * r_hat + B_theta * theta_hat + B_phi_azimuthal * phi_azimuthal_hat
    )

    return 1e-9 * B_vec_g  # convert to Teslas


def get_geomagnetic_field_norm(point: Point, b_field_type="dipole") -> float:
    """
    The geomagnetic field norm at point A.

    Parameters
    ----------
    pointA : Point
        The evaluation point.
    b_field_type : str, optional
        The geomagnetic field type, defaults to dipole.

    Returns
    -------
    float
        The norm of the geomagnetic field.
    """
    if b_field_type == "dipole":
        B_vec_g = get_geomagnetic_field_cartesian_dipole(point)
        norm_dipole = (
            B0
            * (EARTH_RADIUS / point.r_m) ** 3
            * np.sqrt(1 + 3 * np.sin(point.phi_m) ** 2)
        )
    elif b_field_type == "igrf":
        B_vec_g = get_geomagnetic_field_cartesian_igrf(point)
    else:
        raise NotImplementedError

    norm = np.linalg.norm(B_vec_g)

    if b_field_type == "dipole":
        assert np.abs(norm - norm_dipole) < 1e-10

    return float(np.linalg.norm(B_vec_g))


def get_inclination_angle(point: Point, b_field_type="dipole") -> float:
    """
    The inclination angle of the geomagnetic field,
    evaluated at the point (r, phi, lambd) in magnetic coordinates.

    As a check, the angle is computed two ways:
        (1) tan I = |B_r| / |B_phi|
        (2) tan I = 2 tan( |phi| )
    If these two expressions do not agree, an error is raised.
    Computed this way, I is always positive.

    Parameters
    ----------
    point : Point
        The evaluation point.
    b_field_type : str, optional
        The geomagnetic field type, defaults to dipole.

    Returns
    -------
    float
        The inclination angle, in radians.
    """
    if b_field_type != "dipole":
        raise NotImplementedError
    B_r, B_phi, _ = get_geomagnetic_field_latlong(point)
    expression_1 = np.arctan(np.abs(B_r) / np.abs(B_phi))
    expression_2 = np.arctan(2 * np.tan(np.abs(point.phi_m)))
    assert expression_1 == expression_2
    return float(expression_1)


def get_theta_angle(point_b: Point, point_s: Point, b_field_type="dipole") -> float:
    """
    The angle b/w the line of sight radial vector and the local magnetic
    field. The evaluation point could be a point on the Earth's surface,
    but it seems more appropriate to pick a point lying in the absorption
    zone.

    Parameters
    ----------
    point_b : Point
        Burst point.
    point_s : Point
        Evaluation point along the line of sight.
    b_field_type : str, optional
        The geomagnetic field type, defaults to dipole.

    Returns
    -------
    float
        The angle theta, in radians.
    """
    if b_field_type == "dipole":
        B_vec_g = get_geomagnetic_field_cartesian_dipole(point_s)
    elif b_field_type == "igrf":
        B_vec_g = get_geomagnetic_field_cartesian_igrf(point_s)
    else:
        raise NotImplementedError
    xvec_g_from_B_to_S = get_xvec_g_from_A_to_B(point_b, point_s)
    num = np.dot(xvec_g_from_B_to_S, B_vec_g)
    den = np.linalg.norm(xvec_g_from_B_to_S) * np.linalg.norm(B_vec_g)
    return float(np.arccos(num / den))


def get_A_angle(point_b: Point, point_s: Point) -> float:
    """
    The angle A b/w the line of sight radial vector and the
    vertical/origin-burst vector.

    Parameters
    ----------
    point_b : Point
        The burst point.
    point_s : Point
        Evaluation point along the line of sight.

    Returns
    -------
    float
        The angle A, in radians.
    """
    xvec_g_from_O_to_B = np.asarray([point_b.x_g, point_b.y_g, point_b.z_g])
    xvec_g_from_B_to_S = get_xvec_g_from_A_to_B(point_b, point_s)
    num = -1.0 * np.dot(xvec_g_from_B_to_S, xvec_g_from_O_to_B)
    den = np.linalg.norm(xvec_g_from_B_to_S) * np.linalg.norm(xvec_g_from_O_to_B)
    # A = 0 sometimes fails due to rounding errors in the coord conversions
    if num / den > 1 and num / den < 1 + 1e-5:
        return 0.0
    return float(np.arccos(num / den))


def get_line_of_sight_midway_point(point_b: Point, point_t: Point) -> Point:
    """
    Get the position vector in geographic cartesian coordinates
    to a point mid-way between the upper and lower absorption layers,
    along the line of sight ray.

    Parameters
    ----------
    point_b : Point
        Burst point.
    point_t : Point
        Target point.

    Returns
    -------
    Point
        The midway point.
    """
    A = get_A_angle(point_b, point_t)
    HOB = (
        np.linalg.norm(np.asarray([point_b.x_g, point_b.y_g, point_b.z_g]))
        - EARTH_RADIUS
    )

    # distance from burst point (r=0) to top of absorption layer
    rmin = (HOB - ABSORPTION_LAYER_UPPER) / np.cos(A)

    # distance from burst point (r=0) to bottom of absorption layer
    rmax = (HOB - ABSORPTION_LAYER_LOWER) / np.cos(A)

    # compute the vector from B to T
    xvec_g_from_B_to_T = get_xvec_g_from_A_to_B(point_b, point_t)

    # rescale the length to be (rmin+rmax)/2, producing X_B_to_midway
    xvec_g_from_B_to_M = (
        ((rmin + rmax) / 2) * xvec_g_from_B_to_T / np.linalg.norm(xvec_g_from_B_to_T)
    )

    # compute the vector from O to B
    xvec_g_from_O_to_B = np.asarray([point_b.x_g, point_b.y_g, point_b.z_g])

    # compute the vector from O to M
    xvec_g_from_O_to_M = xvec_g_from_O_to_B + xvec_g_from_B_to_M

    # compute the midway point M
    M = Point(
        xvec_g_from_O_to_M[0],
        xvec_g_from_O_to_M[1],
        xvec_g_from_O_to_M[2],
        coordsys="cartesian geo",
    )

    return M


def line_of_sight_check(pointB: Point, pointT: Point) -> None:
    """
    Given a burst and target point on the Earth's surface, compute
    the vector pointing from B to T and confirm that this vector's
    length is less than the length of the tangent vector pointing from B
    to a point on the surface.

    Parameters
    ----------
    pointB : Point
        Burst point.
    pointT : Point
        Target point.

    Raises
    ------
    Union[None, ValueError]
        Raise an error if the coordinates have overshot the horizon.
    """
    HOB = (
        np.linalg.norm(np.asarray([pointB.x_g, pointB.y_g, pointB.z_g])) - EARTH_RADIUS
    )
    Amax = np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + HOB))
    rmax = (EARTH_RADIUS + HOB) * np.cos(Amax)
    xvec_g_B_to_T = get_xvec_g_from_A_to_B(pointB, pointT)
    try:
        # check distance to target
        assert np.linalg.norm(xvec_g_B_to_T) <= rmax
        # check angle A
        Amax = np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + HOB))
        A = get_A_angle(pointB, pointT)
        assert 0 <= A and A <= Amax
    except:
        raise ValueError("Coordinates have overshot the horizon!")
    return


def compute_max_delta_angle_1d(
    pointB: Point, Delta_angle=25 * np.pi / 180, N_pts=150
) -> float:
    """
    Compute the largest delta angle such that a 1d grid
    of lat points deviating from the burst point by at
    most Delta_angle radians will be entirely contained within
    the line-of-sight cone of the burst point.
    Geographic coordinates are assumed.

    Parameters
    ----------
    pointB : Point
        Burst point
    Delta_angle : _type_, optional
        Starting delta latitude angle, in radians. By default 25*np.pi/180
    N_pts : int, optional
        Grid points, by default 150

    Returns
    -------
    float
        A delta latitude, in radians.
    """
    # perform the scan over location
    while Delta_angle > 1e-3:
        # build the angular grid and scan
        phi_T_list = pointB.phi_g + np.linspace(
            -Delta_angle / 2, Delta_angle / 2, N_pts
        )
        try:
            for i in range(len(phi_T_list)):
                # check whether the target point is within the line of sight
                pointT = Point(
                    EARTH_RADIUS, phi_T_list[i], pointB.lambd_g, coordsys="lat/long geo"
                )
                line_of_sight_check(pointB, pointT)
            return float(Delta_angle)
        except:
            # decay the delta angle
            Delta_angle = 0.95 * Delta_angle

    return float(Delta_angle)


def compute_max_delta_angle_2d(
    pointB: Point, Delta_angle=25 * np.pi / 180, N_pts=150
) -> float:
    """
    Compute the largest Delta_angle such that a 2d square grid
    of lat/long points deviating from the burst point by at
    most Delta_angle radians will be entirely contained within
    the line-of-sight cone of the burst point.
    Geographic coordinates are assumed.

    Parameters
    ----------
    pointB : Point
        Burst point.
    Delta_angle : float, optional
        Initial delta angle, in radians. By default 25*np.pi/180
    N_pts : int, optional
        Grid points, by default 150

    Returns
    -------
    float
        A delta angle (used for both lat and long), in radians.
    """
    # perform the scan over location
    while Delta_angle > 1e-3:
        # build the angular grid and scan
        phi_T_list = pointB.phi_g + np.linspace(
            -Delta_angle / 2, Delta_angle / 2, N_pts
        )
        lambd_T_list = pointB.lambd_g + np.linspace(
            -Delta_angle / 2, Delta_angle / 2, N_pts
        )
        try:
            for i in range(len(phi_T_list)):
                for j in range(len(lambd_T_list)):
                    pointT = Point(
                        EARTH_RADIUS,
                        phi_T_list[i],
                        lambd_T_list[j],
                        coordsys="lat/long geo",
                    )
                    line_of_sight_check(pointB, pointT)
            return float(Delta_angle)  # Return if all points are valid
        except:
            # decay the delta angle
            Delta_angle = 0.95 * Delta_angle

    return float(Delta_angle)
