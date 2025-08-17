import numpy as np
import pytest

from emp.constants import EARTH_RADIUS
from emp.geometry import (
    Point,
    _cartesian_to_latlong,
    _cartesian_to_spherical,
    _latlong_geo_to_mag,
    _latlong_mag_to_geo,
    _latlong_to_cartesian,
    _spherical_to_cartesian,
    compute_max_delta_angle_1d,
    compute_max_delta_angle_2d,
    get_rotation_matrix,
)


def test_rotation_matrix_properties() -> None:
    """Test that rotation matrices have correct mathematical properties."""
    axis = np.asarray([0.1, 0.2, 0.3])
    theta = np.pi / 2
    R = get_rotation_matrix(theta, axis)

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
    r = 6371.0
    phi = np.pi / 4
    lambd = np.pi / 6

    point_latlong_geo = Point(r, phi, lambd, "lat/long geo")

    point_cartesian_geo = Point(
        point_latlong_geo.x_g,
        point_latlong_geo.y_g,
        point_latlong_geo.z_g,
        "cartesian geo",
    )

    point_latlong_mag = Point(
        point_latlong_geo.r_m,
        point_latlong_geo.phi_m,
        point_latlong_geo.lambd_m,
        "lat/long mag",
    )

    point_cartesian_mag = Point(
        point_latlong_geo.x_m,
        point_latlong_geo.y_m,
        point_latlong_geo.z_m,
        "cartesian mag",
    )

    assert point_latlong_geo == point_cartesian_geo
    assert point_latlong_geo == point_latlong_mag
    assert point_latlong_geo == point_cartesian_mag
    assert point_cartesian_geo == point_latlong_mag
    assert point_cartesian_geo == point_cartesian_mag
    assert point_latlong_mag == point_cartesian_mag

    # Test point at equator
    point_equator_latlong = Point(6371.0, 0.0, 0.0, "lat/long geo")
    point_equator_cartesian = Point(
        point_equator_latlong.x_g,
        point_equator_latlong.y_g,
        point_equator_latlong.z_g,
        "cartesian geo",
    )
    assert point_equator_latlong == point_equator_cartesian

    # Test point near north pole (avoiding singularity)
    point_near_pole_latlong = Point(6371.0, np.pi / 2 - 1e-10, 0.0, "lat/long geo")
    point_near_pole_cartesian = Point(
        point_near_pole_latlong.x_g,
        point_near_pole_latlong.y_g,
        point_near_pole_latlong.z_g,
        "cartesian geo",
    )
    assert point_near_pole_latlong == point_near_pole_cartesian


def test_point_inequality() -> None:
    """Test that different points are not equal."""
    point1 = Point(6371.0, 0.0, 0.0, "lat/long geo")
    point2 = Point(6371.0, np.pi / 4, 0.0, "lat/long geo")
    point3 = Point(6371.0, 0.0, np.pi / 6, "lat/long geo")
    point4 = Point(6500.0, 0.0, 0.0, "lat/long geo")

    assert point1 != point2
    assert point1 != point3
    assert point1 != point4
    assert point2 != point3
    assert point2 != point4
    assert point3 != point4


def test_point_hash_consistency() -> None:
    """Test that equal points have equal hashes."""
    r, phi, lambd = 6371.0, np.pi / 3, np.pi / 4

    point1 = Point(r, phi, lambd, "lat/long geo")
    point2 = Point(point1.x_g, point1.y_g, point1.z_g, "cartesian geo")

    assert point1 == point2
    assert hash(point1) == hash(point2)

    point_set = {point1, point2}
    assert len(point_set) == 1


def test_cartesian_geo_round_trip_transformations() -> None:
    """Test that cartesian <-> geographic coordinate transformations are consistent."""
    rng = np.random.default_rng(42)
    tol = 1e-10
    num_trials = 1000

    # Test cartesian -> geo -> cartesian round trip
    for _ in range(num_trials):
        x, y, z = rng.uniform(low=-10, high=10, size=3)
        r, phi, lambd = _cartesian_to_latlong(x, y, z)
        x2, y2, z2 = _latlong_to_cartesian(r, phi, lambd)

        np.testing.assert_allclose(
            [x, y, z],
            [x2, y2, z2],
            atol=tol,
            err_msg=f"Cartesian round-trip failed for ({x}, {y}, {z})",
        )

    # Test geo -> cartesian -> geo round trip
    for _ in range(num_trials):
        r = 1.0
        phi = rng.uniform(-np.pi / 2, np.pi / 2)
        lambd = rng.uniform(-np.pi, np.pi)

        x, y, z = _latlong_to_cartesian(r, phi, lambd)
        r2, phi2, lambd2 = _cartesian_to_latlong(x, y, z)

        np.testing.assert_allclose(
            [r, phi, lambd],
            [r2, phi2, lambd2],
            atol=tol,
            err_msg=f"Geographic round-trip failed for (r={r}, phi={phi}, lambd={lambd})",
        )


def test_cartesian_spherical_round_trip_transformations() -> None:
    """Test that cartesian <-> spherical coordinate transformations are consistent."""
    rng = np.random.default_rng(42)
    tol = 1e-10
    num_trials = 1000

    # Test cartesian -> spherical -> cartesian round trip
    for _ in range(num_trials):
        x, y, z = rng.uniform(low=-10, high=10, size=3)

        if np.sqrt(x**2 + y**2 + z**2) < 1e-10:
            continue

        r, theta, phi = _cartesian_to_spherical(x, y, z)
        x2, y2, z2 = _spherical_to_cartesian(r, theta, phi)

        np.testing.assert_allclose(
            [x, y, z],
            [x2, y2, z2],
            atol=tol,
            err_msg=f"Cartesian->spherical round-trip failed for ({x}, {y}, {z})",
        )

    # Test spherical -> cartesian -> spherical round trip
    for _ in range(num_trials):
        r = 1.0
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(-np.pi, np.pi)

        x, y, z = _spherical_to_cartesian(r, theta, phi)
        r2, theta2, phi2 = _cartesian_to_spherical(x, y, z)

        np.testing.assert_allclose(
            [r, theta, phi],
            [r2, theta2, phi2],
            atol=tol,
            err_msg=f"Spherical round-trip failed for (r={r}, theta={theta}, phi={phi})",
        )


def test_specific_coordinate_edge_cases() -> None:
    """Test coordinate transformations at specific edge cases."""
    tol = 1e-12

    test_cases_geo = [
        (1.0, 0.0, 0.0),  # Equator, 0° longitude
        (1.0, np.pi / 2, 0.0),  # North pole
        (1.0, -np.pi / 2, 0.0),  # South pole
        (1.0, 0.0, np.pi),  # Equator, 180° longitude
        (1.0, 0.0, -np.pi),  # Equator, -180° longitude
    ]

    for r, phi, lambd in test_cases_geo:
        x, y, z = _latlong_to_cartesian(r, phi, lambd)
        r2, phi2, lambd2 = _cartesian_to_latlong(x, y, z)

        np.testing.assert_allclose(
            [r, phi, lambd],
            [r2, phi2, lambd2],
            atol=tol,
            err_msg=f"Edge case failed for geographic coords (r={r}, phi={phi}, lambd={lambd})",
        )


class TestComputeMaxDeltaAngle:
    """Tests for compute_max_delta_angle functions."""

    def test_compute_max_delta_angle_1d_basic(self) -> None:
        """Test basic functionality of 1D delta angle computation."""
        burst_point = Point(EARTH_RADIUS + 400.0, 0.0, 0.0, "lat/long geo")
        max_angle = compute_max_delta_angle_1d(burst_point, n_grid_points=20)

        assert isinstance(max_angle, float)
        assert max_angle > 0
        assert max_angle < np.pi / 2

    def test_compute_max_delta_angle_2d_basic(self) -> None:
        """Test basic functionality of 2D delta angle computation."""
        burst_point = Point(EARTH_RADIUS + 400.0, 0.0, 0.0, "lat/long geo")
        max_angle = compute_max_delta_angle_2d(burst_point, n_grid_points=20)

        assert isinstance(max_angle, float)
        assert max_angle > 0
        assert max_angle < np.pi / 2

    def test_1d_vs_2d_relationship(self) -> None:
        """Test that 2D delta angle is smaller than or equal to 1D delta angle."""
        burst_point = Point(EARTH_RADIUS + 500.0, np.pi / 6, np.pi / 4, "lat/long geo")

        max_angle_1d = compute_max_delta_angle_1d(burst_point, n_grid_points=15)
        max_angle_2d = compute_max_delta_angle_2d(burst_point, n_grid_points=15)

        assert max_angle_2d <= max_angle_1d

    def test_higher_altitude_larger_angle(self) -> None:
        """Test that higher altitude allows larger delta angles."""
        burst_low = Point(EARTH_RADIUS + 200.0, 0.0, 0.0, "lat/long geo")
        burst_high = Point(EARTH_RADIUS + 800.0, 0.0, 0.0, "lat/long geo")

        max_angle_low = compute_max_delta_angle_1d(burst_low, n_grid_points=15)
        max_angle_high = compute_max_delta_angle_1d(burst_high, n_grid_points=15)

        assert max_angle_high >= max_angle_low

    def test_input_validation_burst_below_surface(self) -> None:
        """Test that burst point below Earth surface raises ValueError."""
        burst_below = Point(EARTH_RADIUS - 100.0, 0.0, 0.0, "lat/long geo")

        with pytest.raises(
            ValueError, match="Burst point must be above Earth's surface"
        ):
            compute_max_delta_angle_1d(burst_below)

    def test_input_validation_negative_angle(self) -> None:
        """Test that negative initial delta angle raises ValueError."""
        burst_point = Point(EARTH_RADIUS + 400.0, 0.0, 0.0, "lat/long geo")

        with pytest.raises(ValueError, match="Initial delta angle must be positive"):
            compute_max_delta_angle_1d(burst_point, initial_delta_angle=-0.1)

    def test_input_validation_few_grid_points(self) -> None:
        """Test that too few grid points raises ValueError."""
        burst_point = Point(EARTH_RADIUS + 400.0, 0.0, 0.0, "lat/long geo")

        with pytest.raises(ValueError, match="Need at least 3 grid points"):
            compute_max_delta_angle_1d(burst_point, n_grid_points=2)

    def test_custom_parameters(self) -> None:
        """Test functions work with custom parameters."""
        burst_point = Point(EARTH_RADIUS + 300.0, np.pi / 8, np.pi / 6, "lat/long geo")

        max_angle_1d = compute_max_delta_angle_1d(
            burst_point,
            initial_delta_angle=np.pi / 12,
            n_grid_points=10,
            tolerance=1e-5,
        )

        max_angle_2d = compute_max_delta_angle_2d(
            burst_point,
            initial_delta_angle=np.pi / 12,
            n_grid_points=10,
            tolerance=1e-5,
        )

        assert isinstance(max_angle_1d, float)
        assert isinstance(max_angle_2d, float)
        assert max_angle_1d > 0
        assert max_angle_2d > 0

    def test_convergence(self) -> None:
        """Test that functions converge and don't hit iteration limits."""
        burst_point = Point(EARTH_RADIUS + 600.0, 0.0, 0.0, "lat/long geo")

        max_angle_1d = compute_max_delta_angle_1d(
            burst_point,
            initial_delta_angle=np.pi / 18,
            max_iterations=25,
            n_grid_points=10,
            tolerance=1e-4,
        )
        max_angle_2d = compute_max_delta_angle_2d(
            burst_point,
            initial_delta_angle=np.pi / 18,
            max_iterations=25,
            n_grid_points=8,
            tolerance=1e-4,
        )

        assert max_angle_1d > 0
        assert max_angle_2d > 0
        assert max_angle_1d > 1e-5
        assert max_angle_2d > 1e-5

    def test_different_coordinate_systems(self) -> None:
        """Test that burst point coordinate system doesn't affect results."""
        burst_latlong = Point(
            EARTH_RADIUS + 400.0, np.pi / 6, np.pi / 4, "lat/long geo"
        )
        burst_cartesian = Point(
            burst_latlong.x_g, burst_latlong.y_g, burst_latlong.z_g, "cartesian geo"
        )

        max_angle_ll = compute_max_delta_angle_1d(burst_latlong, n_grid_points=10)
        max_angle_cart = compute_max_delta_angle_1d(burst_cartesian, n_grid_points=10)

        np.testing.assert_allclose(max_angle_ll, max_angle_cart, rtol=1e-8)

    def test_compute_max_delta_angle_2d_comprehensive(self) -> None:
        """Test 2D computation with larger grid parameters."""
        burst_point = Point(EARTH_RADIUS + 400.0, 0.0, 0.0, "lat/long geo")

        max_angle = compute_max_delta_angle_2d(
            burst_point, n_grid_points=20, tolerance=1e-7
        )

        assert isinstance(max_angle, float)
        assert max_angle > 0
        assert max_angle < np.pi / 6


class TestCoordinateConversions:
    """Test round-trip conversions between coordinate systems."""

    def test_cartesian_spherical_round_trip(self) -> None:
        """Test cartesian <-> spherical round-trip conversions."""
        rng = np.random.default_rng(42)
        tol = 1e-10

        for _ in range(100):
            x, y, z = rng.uniform(low=-10, high=10, size=3)
            if np.sqrt(x**2 + y**2 + z**2) < 1e-10:
                continue

            r, theta, phi = _cartesian_to_spherical(x, y, z)
            x2, y2, z2 = _spherical_to_cartesian(r, theta, phi)

            np.testing.assert_allclose([x, y, z], [x2, y2, z2], atol=tol)

    def test_cartesian_latlong_round_trip(self) -> None:
        """Test cartesian <-> lat/long round-trip conversions."""
        rng = np.random.default_rng(42)
        tol = 1e-10

        for _ in range(100):
            x, y, z = rng.uniform(low=-10, high=10, size=3)
            if np.sqrt(x**2 + y**2 + z**2) < 1e-10:
                continue

            r, phi, lambd = _cartesian_to_latlong(x, y, z)
            x2, y2, z2 = _latlong_to_cartesian(r, phi, lambd)

            np.testing.assert_allclose([x, y, z], [x2, y2, z2], atol=tol)

    def test_latlong_geo_mag_round_trip(self) -> None:
        """Test geographic <-> magnetic lat/long round-trip conversions."""
        rng = np.random.default_rng(42)
        tol = 1e-10

        for _ in range(100):
            r_g = rng.uniform(0.1, 10.0)
            phi_g = rng.uniform(-np.pi / 2, np.pi / 2)
            lambd_g = rng.uniform(-np.pi, np.pi)

            r_m, phi_m, lambd_m = _latlong_geo_to_mag(r_g, phi_g, lambd_g)
            r_g2, phi_g2, lambd_g2 = _latlong_mag_to_geo(r_m, phi_m, lambd_m)

            np.testing.assert_allclose(
                [r_g, phi_g, lambd_g], [r_g2, phi_g2, lambd_g2], atol=tol
            )
