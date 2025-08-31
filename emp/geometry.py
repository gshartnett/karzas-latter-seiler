"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing.

Contains the Point class describe locations in a coordinate-independent manner.
"""

from typing import (
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from emp.constants import (
    ABSORPTION_LAYER_LOWER,
    ABSORPTION_LAYER_UPPER,
    EARTH_RADIUS,
    LAMBDA_MAGNP,
    PHI_MAGNP,
)


def get_rotation_matrix(theta: float, axis: np.ndarray) -> NDArray[np.float64]:
    """
    Rotation matrix for angle theta and axis (vx, vy, vz).
    https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_rotation_matrix_to_axis%E2%80%93angle

    Parameters
    ----------
    theta : float
        Rotation angle, in radians.

    Returns
    -------
    NDArray[np.float64]
        Rotation matrix.
    """
    if len(axis) != 3:
        raise ValueError("Axis must be a 3-vector.")

    # Get the components of the rotation axis
    vec_x, vec_y, vec_z = axis / np.linalg.norm(axis)

    # Build the rotation matrix
    rotation_matrix: NDArray[np.float64] = np.zeros((3, 3))

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

    def __init__(
        self,
        coord1: float,
        coord2: float,
        coord3: float,
        coordsys: str,
        consistency_check: bool = True,
    ) -> None:
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
                self.x_g, self.y_g, self.z_g = _latlong_to_cartesian(
                    self.r_g, self.phi_g, self.lambd_g
                )
            elif coordsys == "cartesian geo":
                self.x_g, self.y_g, self.z_g = coord1, coord2, coord3
                self.r_g, self.phi_g, self.lambd_g = _cartesian_to_latlong(
                    self.x_g, self.y_g, self.z_g
                )
            # convert to magnetic coords
            self.r_m, self.phi_m, self.lambd_m = _latlong_geo_to_mag(
                self.r_g, self.phi_g, self.lambd_g
            )
            self.x_m, self.y_m, self.z_m = _latlong_to_cartesian(
                self.r_m, self.phi_m, self.lambd_m
            )

        # point initially defined using magnetic coords
        else:
            if coordsys == "lat/long mag":
                self.r_m, self.phi_m, self.lambd_m = coord1, coord2, coord3
                self.x_m, self.y_m, self.z_m = _latlong_to_cartesian(
                    self.r_m, self.phi_m, self.lambd_m
                )
            elif coordsys == "cartesian mag":
                self.x_m, self.y_m, self.z_m = coord1, coord2, coord3
                self.r_m, self.phi_m, self.lambd_m = _cartesian_to_latlong(
                    self.x_m, self.y_m, self.z_m
                )
            # convert to geographic coords
            self.r_g, self.phi_g, self.lambd_g = _latlong_mag_to_geo(
                self.r_m, self.phi_m, self.lambd_m
            )
            self.x_g, self.y_g, self.z_g = _latlong_to_cartesian(
                self.r_g, self.phi_g, self.lambd_g
            )

        # check the coordinates
        self.validate_latlong_coords(self.r_g, self.phi_g, self.lambd_g)
        self.validate_latlong_coords(self.r_m, self.phi_m, self.lambd_m)

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

    def __str__(self) -> str:
        return (
            f"Point(rg={self.r_g}, phi_g={self.phi_g:.4f}, lambd_g={self.lambd_g:.4f})"
        )

    def __eq__(self, other: object) -> bool:
        """
        Return true if two points are the same.

        Parameters
        ----------
        other : object
            Another point to compare with.

        Returns
        -------
        bool
            Are the two points equivalent?
        """
        if not isinstance(other, Point):
            return False

        tol = 1e-6
        return bool(
            np.all(
                [
                    np.abs(self.__dict__[key] - other.__dict__[key]) < tol
                    for key in self.__dict__.keys()
                ]
            )
        )

    def __hash__(self) -> int:
        """
        Make Point hashable by using a tuple of rounded coordinate values.
        This allows Points to be used in sets and as dictionary keys.
        """
        # Round to avoid floating point precision issues
        return hash(
            (round(self.r_g, 10), round(self.phi_g, 10), round(self.lambd_g, 10))
        )

    @classmethod
    def from_gps_coordinates(
        cls,
        latitude: Union[float, str],
        longitude: Union[float, str],
        altitude_km: float = 0.0,
    ) -> "Point":
        """
        Create a Point from GPS-style coordinates.

        Parameters
        ----------
        latitude : float | str
            Latitude in degrees (e.g. 40.7128) or as a string with N/S suffix.
        longitude : float | str
            Longitude in degrees (e.g. -74.0060) or as a string with E/W suffix.
        altitude_km : float
            Altitude above mean sea level in kilometers.

        Returns
        -------
        Point
            A Point object in geographic lat/long coordinates.
        """

        # If inputs are strings like "40.7N", "74W", handle parsing:
        def parse_coord(coord: Union[float, str], is_lat: bool) -> float:
            if isinstance(coord, (int, float)):
                return float(coord)
            c = coord.strip().upper()
            if c.endswith("N") and is_lat:
                return float(c[:-1])
            if c.endswith("S") and is_lat:
                return -float(c[:-1])
            if c.endswith("E") and not is_lat:
                return float(c[:-1])
            if c.endswith("W") and not is_lat:
                return -float(c[:-1])
            return float(c)

        lat_deg = parse_coord(latitude, True)
        lon_deg = parse_coord(longitude, False)

        r = EARTH_RADIUS + altitude_km
        phi = np.radians(lat_deg)  # latitude in radians
        lambd = np.radians(lon_deg)  # longitude in radians

        return cls(r, phi, lambd, coordsys="lat/long geo")

    @staticmethod
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

    @staticmethod
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


def _latlong_geo_to_mag(
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
    x_g, y_g, z_g = _latlong_to_cartesian(r_g, phi_g, lambd_g)
    x_m, y_m, z_m = np.dot(INV_ROTATION_MATRIX, np.asarray([x_g, y_g, z_g]))
    r_m, phi_m, lambd_m = _cartesian_to_latlong(x_m, y_m, z_m)
    return r_m, phi_m, lambd_m


def _latlong_mag_to_geo(
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
    x_m, y_m, z_m = _latlong_to_cartesian(r_m, phi_m, lambd_m)
    x_g, y_g, z_g = np.dot(ROTATION_MATRIX, np.asarray([x_m, y_m, z_m]))
    r_g, phi_g, lambd_g = _cartesian_to_latlong(x_g, y_g, z_g)
    return r_g, phi_g, lambd_g


def _latlong_to_cartesian(
    r: float, phi: float, lambd: float
) -> Tuple[float, float, float]:
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


def _cartesian_to_latlong(x: float, y: float, z: float) -> Tuple[float, float, float]:
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


def _spherical_to_cartesian(
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


def _cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
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
    phi = np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))
    return r, theta, phi


def get_xvec_g_from_A_to_B(pointA: Point, pointB: Point) -> NDArray[np.float64]:
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
    NDArray[np.float64]
        The vector pointing from point A to point B.
    """
    xvec_g_from_O_to_A: NDArray[np.float64] = np.asarray(
        [pointA.x_g, pointA.y_g, pointA.z_g]
    )
    xvec_g_from_O_to_B: NDArray[np.float64] = np.asarray(
        [pointB.x_g, pointB.y_g, pointB.z_g]
    )
    return xvec_g_from_O_to_B - xvec_g_from_O_to_A


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

    # TODO Remove this restriction
    if HOB < ABSORPTION_LAYER_UPPER:
        raise ValueError(
            "Burst height must be above the upper absorption layer "
            f"({ABSORPTION_LAYER_UPPER} km)"
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


def line_of_sight_check(burst_point: Point, target_point: Point) -> None:
    """
    Given a burst and target point on the Earth's surface, compute
    the vector pointing from B to T and confirm that this vector's
    length is less than the length of the tangent vector pointing from B
    to a point on the surface.

    Parameters
    ----------
    burst_point : Point
        Burst point.
    target_point : Point
        Target point.

    Raises
    ------
    ValueError
        If the coordinates have overshot the horizon.
    """
    HOB = (
        np.linalg.norm(np.asarray([burst_point.x_g, burst_point.y_g, burst_point.z_g]))
        - EARTH_RADIUS
    )
    Amax = np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + HOB))
    rmax = (EARTH_RADIUS + HOB) * np.cos(Amax)
    xvec_g_B_to_T = get_xvec_g_from_A_to_B(burst_point, target_point)

    # Check distance to target
    distance = np.linalg.norm(xvec_g_B_to_T)
    if distance > rmax:
        raise ValueError(
            f"Target distance ({distance:.3f}) exceeds maximum line-of-sight distance ({rmax:.3f}). "
            "Coordinates have overshot the horizon!"
        )

    # Check angle A
    A = get_A_angle(burst_point, target_point)
    if not (0 <= A <= Amax):
        raise ValueError(
            f"Line-of-sight angle A ({A:.6f} rad) is outside valid range [0, {Amax:.6f}]. "
            "Coordinates have overshot the horizon!"
        )
    return
