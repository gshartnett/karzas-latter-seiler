import numpy as np
from datetime import datetime
from unittest.mock import patch

from emp.constants import EARTH_RADIUS
from emp.geomagnetic_field import (
    MagneticFieldFactory,
    MagneticFieldModel,
    CoordinateSystem,
)
from emp.geometry import (
    Point
    )

from emp.geometry_legacy import (
    get_geomagnetic_field_latlong,
    get_geomagnetic_field_cartesian_dipole,
    get_geomagnetic_field_cartesian_igrf,
    get_geomagnetic_field_norm,
    get_inclination_angle,
    get_theta_angle,
)


class TestLegacyCompatibility:
    """Test that new magnetic field classes match legacy functions."""

    def test_dipole_latlong_components_match_legacy(self) -> None:
        """Test that dipole lat/long components match legacy function."""
        dipole_field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)

        test_points = [
            Point(EARTH_RADIUS + 100.0, 0.0, 0.0, "lat/long geo"),
            Point(EARTH_RADIUS + 400.0, np.pi/4, np.pi/6, "lat/long geo"),
            Point(EARTH_RADIUS + 800.0, np.pi/3, -np.pi/4, "lat/long geo"),
            Point(EARTH_RADIUS + 1200.0, -np.pi/6, np.pi/3, "lat/long geo"),
        ]

        for point in test_points:
            # New implementation
            B_new = dipole_field.get_field_vector(point, CoordinateSystem.LATLONG_MAG)

            # Legacy implementation
            B_legacy = get_geomagnetic_field_latlong(point)

            np.testing.assert_allclose(
                B_new,
                B_legacy,
                rtol=1e-12,
                err_msg=f"Dipole lat/long components mismatch at point {point}"
            )

    def test_dipole_cartesian_geo_matches_legacy(self) -> None:
        """Test that dipole geographic Cartesian field matches legacy function."""
        dipole_field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)

        test_points = [
            Point(EARTH_RADIUS + 100.0, 0.0, 0.0, "lat/long geo"),
            Point(EARTH_RADIUS + 400.0, np.pi/4, np.pi/6, "lat/long geo"),
            Point(EARTH_RADIUS + 800.0, np.pi/3, -np.pi/4, "lat/long geo"),
            Point(EARTH_RADIUS + 1200.0, -np.pi/6, np.pi/3, "lat/long geo"),
            Point(EARTH_RADIUS + 200.0, np.pi/6, 0.0, "lat/long geo"),
        ]

        for point in test_points:
            # New implementation
            B_new = dipole_field.get_field_vector(point, CoordinateSystem.CARTESIAN_GEO)

            # Legacy implementation
            B_legacy = get_geomagnetic_field_cartesian_dipole(point)

            np.testing.assert_allclose(
                B_new,
                B_legacy,
                rtol=1e-12,
                err_msg=f"Dipole Cartesian geo field mismatch at point {point}"
            )

    def test_igrf_cartesian_geo_matches_legacy(self) -> None:
        """Test that IGRF geographic Cartesian field matches legacy function."""
        # Use same date for both to ensure consistency
        test_date = datetime(2020, 1, 1)
        igrf_field = MagneticFieldFactory.create(MagneticFieldModel.IGRF, date=test_date)

        test_points = [
            Point(EARTH_RADIUS + 100.0, 0.0, 0.0, "lat/long geo"),
            Point(EARTH_RADIUS + 400.0, np.pi/4, np.pi/6, "lat/long geo"),
            Point(EARTH_RADIUS + 800.0, np.pi/3, -np.pi/4, "lat/long geo"),
            Point(EARTH_RADIUS + 600.0, -np.pi/6, np.pi/3, "lat/long geo"),
        ]

        for point in test_points:
            # New implementation
            B_new = igrf_field.get_field_vector(point, CoordinateSystem.CARTESIAN_GEO)

            # Legacy implementation with mocked date
            with patch('emp.geometry_legacy.datetime') as mock_datetime:
                mock_datetime.today.return_value = test_date
                B_legacy = get_geomagnetic_field_cartesian_igrf(point)

            np.testing.assert_allclose(
                B_new,
                B_legacy,
                rtol=1e-10,  # Slightly looser tolerance for IGRF
                err_msg=f"IGRF Cartesian geo field mismatch at point {point}"
            )

    def test_dipole_field_magnitude_matches_legacy(self) -> None:
        """Test that dipole field magnitude matches legacy function."""
        dipole_field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)

        test_points = [
            Point(EARTH_RADIUS + 100.0, 0.0, 0.0, "lat/long geo"),
            Point(EARTH_RADIUS + 400.0, np.pi/4, np.pi/6, "lat/long geo"),
            Point(EARTH_RADIUS + 800.0, np.pi/3, -np.pi/4, "lat/long geo"),
            Point(EARTH_RADIUS + 1200.0, -np.pi/6, np.pi/3, "lat/long geo"),
        ]

        for point in test_points:
            # New implementation
            magnitude_new = dipole_field.get_field_magnitude(point)

            # Legacy implementation
            magnitude_legacy = get_geomagnetic_field_norm(point, b_field_type="dipole")

            np.testing.assert_allclose(
                magnitude_new,
                magnitude_legacy,
                rtol=1e-12,
                err_msg=f"Dipole field magnitude mismatch at point {point}"
            )

    def test_igrf_field_magnitude_matches_legacy(self) -> None:
        """Test that IGRF field magnitude matches legacy function."""
        test_date = datetime(2020, 1, 1)
        igrf_field = MagneticFieldFactory.create(MagneticFieldModel.IGRF, date=test_date)

        test_points = [
            Point(EARTH_RADIUS + 100.0, 0.0, 0.0, "lat/long geo"),
            Point(EARTH_RADIUS + 400.0, np.pi/4, np.pi/6, "lat/long geo"),
            Point(EARTH_RADIUS + 800.0, np.pi/3, -np.pi/4, "lat/long geo"),
        ]

        for point in test_points:
            # New implementation
            magnitude_new = igrf_field.get_field_magnitude(point)

            # Legacy implementation with mocked date
            with patch('emp.geometry_legacy.datetime') as mock_datetime:
                mock_datetime.today.return_value = test_date
                magnitude_legacy = get_geomagnetic_field_norm(point, b_field_type="igrf")

            np.testing.assert_allclose(
                magnitude_new,
                magnitude_legacy,
                rtol=1e-10,
                err_msg=f"IGRF field magnitude mismatch at point {point}"
            )

    def test_dipole_inclination_angle_matches_legacy(self) -> None:
        """Test that dipole inclination angle matches legacy function."""
        dipole_field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)

        test_points = [
            Point(EARTH_RADIUS + 100.0, 0.0, 0.0, "lat/long geo"),
            Point(EARTH_RADIUS + 400.0, np.pi/4, np.pi/6, "lat/long geo"),
            Point(EARTH_RADIUS + 800.0, np.pi/3, -np.pi/4, "lat/long geo"),
            Point(EARTH_RADIUS + 600.0, -np.pi/6, np.pi/3, "lat/long geo"),
            Point(EARTH_RADIUS + 300.0, np.pi/6, 0.0, "lat/long geo"),
        ]

        for point in test_points:
            # New implementation
            inclination_new = dipole_field.get_inclination_angle(point)

            # Legacy implementation
            inclination_legacy = get_inclination_angle(point, b_field_type="dipole")

            np.testing.assert_allclose(
                inclination_new,
                inclination_legacy,
                rtol=1e-12,
                err_msg=f"Dipole inclination angle mismatch at point {point}"
            )

    def test_dipole_theta_angle_matches_legacy(self) -> None:
        """Test that dipole theta angle matches legacy function."""
        dipole_field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)

        # Test various burst and line-of-sight point combinations
        test_cases = [
            (
                Point(EARTH_RADIUS + 300.0, np.pi/6, np.pi/8, "lat/long geo"),
                Point(EARTH_RADIUS + 500.0, np.pi/3, np.pi/4, "lat/long geo")
            ),
            (
                Point(EARTH_RADIUS + 200.0, 0.0, 0.0, "lat/long geo"),
                Point(EARTH_RADIUS + 400.0, np.pi/4, np.pi/6, "lat/long geo")
            ),
            (
                Point(EARTH_RADIUS + 600.0, -np.pi/4, -np.pi/3, "lat/long geo"),
                Point(EARTH_RADIUS + 100.0, np.pi/6, np.pi/4, "lat/long geo")
            ),
        ]

        for burst_point, los_point in test_cases:
            # New implementation
            theta_new = dipole_field.get_theta_angle(burst_point, los_point)

            # Legacy implementation
            theta_legacy = get_theta_angle(burst_point, los_point, b_field_type="dipole")

            np.testing.assert_allclose(
                theta_new,
                theta_legacy,
                rtol=1e-12,
                err_msg=f"Dipole theta angle mismatch for burst={burst_point}, los={los_point}"
            )

    def test_igrf_theta_angle_matches_legacy(self) -> None:
        """Test that IGRF theta angle matches legacy function."""
        test_date = datetime(2020, 1, 1)
        igrf_field = MagneticFieldFactory.create(MagneticFieldModel.IGRF, date=test_date)

        # Test various burst and line-of-sight point combinations
        test_cases = [
            (
                Point(EARTH_RADIUS + 300.0, np.pi/6, np.pi/8, "lat/long geo"),
                Point(EARTH_RADIUS + 500.0, np.pi/3, np.pi/4, "lat/long geo")
            ),
            (
                Point(EARTH_RADIUS + 200.0, 0.0, 0.0, "lat/long geo"),
                Point(EARTH_RADIUS + 400.0, np.pi/4, np.pi/6, "lat/long geo")
            ),
        ]

        for burst_point, los_point in test_cases:
            # New implementation
            theta_new = igrf_field.get_theta_angle(burst_point, los_point)

            # Legacy implementation with mocked date
            with patch('emp.geometry_legacy.datetime') as mock_datetime:
                mock_datetime.today.return_value = test_date
                theta_legacy = get_theta_angle(burst_point, los_point, b_field_type="igrf")

            np.testing.assert_allclose(
                theta_new,
                theta_legacy,
                rtol=1e-10,
                err_msg=f"IGRF theta angle mismatch for burst={burst_point}, los={los_point}"
            )

    def test_coordinate_system_consistency_with_legacy(self) -> None:
        """Test that coordinate system transformations are consistent with legacy approach."""
        dipole_field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)

        test_points = [
            Point(EARTH_RADIUS + 100.0, 0.0, 0.0, "lat/long geo"),
            Point(EARTH_RADIUS + 400.0, np.pi/4, np.pi/6, "lat/long geo"),
            Point(EARTH_RADIUS + 800.0, np.pi/3, -np.pi/4, "lat/long geo"),
        ]

        for point in test_points:
            # Get field in all coordinate systems from new implementation
            B_latlong_mag = dipole_field.get_field_vector(point, CoordinateSystem.LATLONG_MAG)
            B_cartesian_geo = dipole_field.get_field_vector(point, CoordinateSystem.CARTESIAN_GEO)

            # Compare with legacy functions
            B_latlong_legacy = get_geomagnetic_field_latlong(point)
            B_cartesian_legacy = get_geomagnetic_field_cartesian_dipole(point)

            # Test lat/long components
            np.testing.assert_allclose(
                B_latlong_mag,
                B_latlong_legacy,
                rtol=1e-12,
                err_msg=f"Lat/long components mismatch at {point}"
            )

            # Test Cartesian components
            np.testing.assert_allclose(
                B_cartesian_geo,
                B_cartesian_legacy,
                rtol=1e-12,
                err_msg=f"Cartesian geo components mismatch at {point}"
            )

            # Test that magnitudes are consistent
            magnitude_new = dipole_field.get_field_magnitude(point)
            magnitude_latlong = np.linalg.norm(B_latlong_mag)
            magnitude_cartesian = np.linalg.norm(B_cartesian_geo)
            magnitude_legacy = get_geomagnetic_field_norm(point, "dipole")

            # All magnitudes should be the same
            np.testing.assert_allclose(
                [magnitude_new, magnitude_latlong, magnitude_cartesian],
                [magnitude_legacy, magnitude_legacy, magnitude_legacy],
                rtol=1e-12,
                err_msg=f"Magnitude inconsistency at {point}"
            )

    def test_edge_cases_match_legacy(self) -> None:
        """Test edge cases like equator, poles, etc."""
        dipole_field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)

        # Test points at special locations
        edge_points = [
            Point(EARTH_RADIUS + 100.0, 0.0, 0.0, "lat/long geo"),          # Equator, 0° longitude
            Point(EARTH_RADIUS + 100.0, 0.0, np.pi/2, "lat/long geo"),     # Equator, 90° longitude
            Point(EARTH_RADIUS + 100.0, 0.0, np.pi - 1e-6, "lat/long geo"), # Equator, ~180° longitude (safely under π)
            Point(EARTH_RADIUS + 100.0, 0.0, -np.pi/2, "lat/long geo"),    # Equator, -90° longitude
            Point(EARTH_RADIUS + 100.0, np.pi/4, 0.0, "lat/long geo"),     # 45° N, 0° longitude
            Point(EARTH_RADIUS + 100.0, -np.pi/4, 0.0, "lat/long geo"),    # 45° S, 0° longitude
        ]

        for point in edge_points:
            # Test field components
            B_new = dipole_field.get_field_vector(point, CoordinateSystem.LATLONG_MAG)
            B_legacy = get_geomagnetic_field_latlong(point)

            np.testing.assert_allclose(
                B_new,
                B_legacy,
                rtol=1e-12,
                err_msg=f"Edge case mismatch at {point}"
            )

            # Test field magnitude
            magnitude_new = dipole_field.get_field_magnitude(point)
            magnitude_legacy = get_geomagnetic_field_norm(point, "dipole")

            np.testing.assert_allclose(
                magnitude_new,
                magnitude_legacy,
                rtol=1e-12,
                err_msg=f"Edge case magnitude mismatch at {point}"
            )

    def test_high_altitude_points_match_legacy(self) -> None:
        """Test that high altitude points match legacy functions."""
        dipole_field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)

        # Test points at various altitudes
        altitudes = [50.0, 200.0, 500.0, 1000.0, 2000.0]  # km above surface
        base_point_coords = [(0.0, 0.0), (np.pi/4, np.pi/6), (np.pi/3, -np.pi/4)]

        for altitude in altitudes:
            for phi, lambd in base_point_coords:
                point = Point(EARTH_RADIUS + altitude, phi, lambd, "lat/long geo")

                # Test components
                B_new = dipole_field.get_field_vector(point, CoordinateSystem.LATLONG_MAG)
                B_legacy = get_geomagnetic_field_latlong(point)

                np.testing.assert_allclose(
                    B_new,
                    B_legacy,
                    rtol=1e-12,
                    err_msg=f"High altitude mismatch at altitude={altitude}, phi={phi}, lambd={lambd}"
                )

                # Test magnitude
                magnitude_new = dipole_field.get_field_magnitude(point)
                magnitude_legacy = get_geomagnetic_field_norm(point, "dipole")

                np.testing.assert_allclose(
                    magnitude_new,
                    magnitude_legacy,
                    rtol=1e-12,
                    err_msg=f"High altitude magnitude mismatch at altitude={altitude}"
                )
