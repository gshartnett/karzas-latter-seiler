import numpy as np

from emp.geometry import (
    Point,
    cartesian2latlong,
    cartesian2spherical,
    get_rotation_matrix,
    latlong2cartesian,
    spherical2cartesian,
)


def test_rotation_matrix_properties() -> None:
    """Test that rotation matrices have correct mathematical properties."""
    # Test with a non-trivial set of angles
    axis = np.asarray([0.1, 0.2, 0.3])
    theta = np.pi / 2
    R = get_rotation_matrix(theta, axis)

    # Check that it's a 3x3 matrix
    assert R.shape == (3, 3)

    # Check orthogonality: R^T * R = I
    np.testing.assert_allclose(
        R.T @ R, np.eye(3), rtol=1e-10, atol=1e-12, err_msg="Matrix is not orthogonal"
    )

    # Check that determinant is +1 (proper rotation, not reflection)
    det = np.linalg.det(R)
    np.testing.assert_allclose(
        det,
        1.0,
        rtol=1e-10,
        atol=1e-12,
        err_msg="Determinant is not 1 - not a proper rotation matrix",
    )


def test_point_equality_across_coordinate_systems() -> None:
    """Test that the same point created in different coordinate systems are equal."""
    # Test case 1: Point at 45° latitude, 30° longitude, Earth radius
    r = 6371.0  # Earth radius in km
    phi = np.pi / 4  # 45° latitude
    lambd = np.pi / 6  # 30° longitude

    # Create the same point using different coordinate systems
    point_latlong_geo = Point(r, phi, lambd, "lat/long geo")

    # Get the Cartesian coordinates and create point from them
    point_cartesian_geo = Point(
        point_latlong_geo.x_g,
        point_latlong_geo.y_g,
        point_latlong_geo.z_g,
        "cartesian geo",
    )

    # Get the magnetic coordinates and create point from them
    point_latlong_mag = Point(
        point_latlong_geo.r_m,
        point_latlong_geo.phi_m,
        point_latlong_geo.lambd_m,
        "lat/long mag",
    )

    # Get the magnetic Cartesian coordinates and create point from them
    point_cartesian_mag = Point(
        point_latlong_geo.x_m,
        point_latlong_geo.y_m,
        point_latlong_geo.z_m,
        "cartesian mag",
    )

    # All points should be equal
    assert point_latlong_geo == point_cartesian_geo
    assert point_latlong_geo == point_latlong_mag
    assert point_latlong_geo == point_cartesian_mag
    assert point_cartesian_geo == point_latlong_mag
    assert point_cartesian_geo == point_cartesian_mag
    assert point_latlong_mag == point_cartesian_mag

    # Test case 2: Point at equator
    point_equator_latlong = Point(6371.0, 0.0, 0.0, "lat/long geo")
    point_equator_cartesian = Point(
        point_equator_latlong.x_g,
        point_equator_latlong.y_g,
        point_equator_latlong.z_g,
        "cartesian geo",
    )

    assert point_equator_latlong == point_equator_cartesian

    # Test case 3: Point at north pole
    point_pole_latlong = Point(6371.0, np.pi / 2, 0.0, "lat/long geo")
    point_pole_cartesian = Point(
        point_pole_latlong.x_g,
        point_pole_latlong.y_g,
        point_pole_latlong.z_g,
        "cartesian geo",
    )

    assert point_pole_latlong == point_pole_cartesian


def test_point_inequality() -> None:
    """Test that different points are not equal."""
    point1 = Point(6371.0, 0.0, 0.0, "lat/long geo")  # Equator, 0° longitude
    point2 = Point(6371.0, np.pi / 4, 0.0, "lat/long geo")  # 45° latitude, 0° longitude
    point3 = Point(6371.0, 0.0, np.pi / 6, "lat/long geo")  # Equator, 30° longitude
    point4 = Point(6500.0, 0.0, 0.0, "lat/long geo")  # Different radius

    # All points should be different
    assert point1 != point2
    assert point1 != point3
    assert point1 != point4
    assert point2 != point3
    assert point2 != point4
    assert point3 != point4


def test_point_hash_consistency() -> None:
    """Test that equal points have equal hashes (required for hash consistency)."""
    # Create the same point in different ways
    r, phi, lambd = 6371.0, np.pi / 3, np.pi / 4

    point1 = Point(r, phi, lambd, "lat/long geo")
    point2 = Point(point1.x_g, point1.y_g, point1.z_g, "cartesian geo")

    # Equal points must have equal hashes
    assert point1 == point2
    assert hash(point1) == hash(point2)

    # Test that points can be used in sets (requires both __eq__ and __hash__)
    point_set = {point1, point2}
    assert len(point_set) == 1  # Should collapse to single point since they're equal


def test_cartesian_geo_round_trip_transformations() -> None:
    """Test that cartesian <-> geographic coordinate transformations are consistent."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    tol = 1e-10
    num_trials = 1000  # Reduced for faster testing

    # Test cartesian -> geo -> cartesian round trip
    for _ in range(num_trials):
        x, y, z = rng.uniform(low=-10, high=10, size=3)
        r, phi, lambd = cartesian2latlong(x, y, z)
        x2, y2, z2 = latlong2cartesian(r, phi, lambd)

        np.testing.assert_allclose(
            [x, y, z],
            [x2, y2, z2],
            atol=tol,
            err_msg=f"Cartesian round-trip failed for ({x}, {y}, {z})",
        )

    # Test geo -> cartesian -> geo round trip
    for _ in range(num_trials):
        r = 1.0  # Unit sphere
        phi = rng.uniform(-np.pi / 2, np.pi / 2)  # Valid latitude range
        lambd = rng.uniform(-np.pi, np.pi)  # Valid longitude range

        x, y, z = latlong2cartesian(r, phi, lambd)
        r2, phi2, lambd2 = cartesian2latlong(x, y, z)

        np.testing.assert_allclose(
            [r, phi, lambd],
            [r2, phi2, lambd2],
            atol=tol,
            err_msg=f"Geographic round-trip failed for (r={r}, phi={phi}, lambd={lambd})",
        )


def test_cartesian_spherical_round_trip_transformations() -> None:
    """Test that cartesian <-> spherical coordinate transformations are consistent."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    tol = 1e-10
    num_trials = 1000  # Reduced for faster testing

    # Test cartesian -> spherical -> cartesian round trip
    for _ in range(num_trials):
        x, y, z = rng.uniform(low=-10, high=10, size=3)

        # Skip origin to avoid singularities
        if np.sqrt(x**2 + y**2 + z**2) < 1e-10:
            continue

        r, theta, phi = cartesian2spherical(x, y, z)
        x2, y2, z2 = spherical2cartesian(r, theta, phi)

        np.testing.assert_allclose(
            [x, y, z],
            [x2, y2, z2],
            atol=tol,
            err_msg=f"Cartesian->spherical round-trip failed for ({x}, {y}, {z})",
        )

    # Test spherical -> cartesian -> spherical round trip
    for _ in range(num_trials):
        r = 1.0  # Unit sphere
        theta = rng.uniform(0, np.pi)  # Valid polar angle range
        phi = rng.uniform(-np.pi, np.pi)  # Valid azimuthal angle range

        x, y, z = spherical2cartesian(r, theta, phi)
        r2, theta2, phi2 = cartesian2spherical(x, y, z)

        np.testing.assert_allclose(
            [r, theta, phi],
            [r2, theta2, phi2],
            atol=tol,
            err_msg=f"Spherical round-trip failed for (r={r}, theta={theta}, phi={phi})",
        )


def test_specific_coordinate_edge_cases() -> None:
    """Test coordinate transformations at specific edge cases."""
    tol = 1e-12

    # Test points at poles and equator for geographic coordinates
    test_cases_geo = [
        (1.0, 0.0, 0.0),  # Equator, 0° longitude
        (1.0, np.pi / 2, 0.0),  # North pole
        (1.0, -np.pi / 2, 0.0),  # South pole
        (1.0, 0.0, np.pi),  # Equator, 180° longitude
        (1.0, 0.0, -np.pi),  # Equator, -180° longitude
    ]

    for r, phi, lambd in test_cases_geo:
        x, y, z = latlong2cartesian(r, phi, lambd)
        r2, phi2, lambd2 = cartesian2latlong(x, y, z)

        np.testing.assert_allclose(
            [r, phi, lambd],
            [r2, phi2, lambd2],
            atol=tol,
            err_msg=f"Edge case failed for geographic coords (r={r}, phi={phi}, lambd={lambd})",
        )
