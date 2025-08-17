from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Tuple, Union

import numpy as np
import ppigrf
from emp.geometry import (
    Point
    )
from emp.geometry import (
    INV_ROTATION_MATRIX,
    ROTATION_MATRIX,
    Point,
    get_xvec_g_from_A_to_B,
)
from emp.constants import (
    ABSORPTION_LAYER_LOWER,
    ABSORPTION_LAYER_UPPER,
    B0,
    EARTH_RADIUS,
    LAMBDA_MAGNP,
    PHI_MAGNP,
)


def get_geomagnetic_field_latlong(pointA: Point) -> Tuple[float, float, float]:
    """
    Returns the geomagnetic field vector, evaluated at the point
    (r, phi, lambd) in magnetic coordinates.

    Assumes a dipole model for the geomagnetic field.

    TODO Delete

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

    TODO Delete

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

    TODO Delete

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

    TODO Delete

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

    TODO Delete

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

    TODO Delete

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
