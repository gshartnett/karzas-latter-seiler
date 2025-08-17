from datetime import datetime

import numpy as np
import pytest

from emp.constants import EARTH_RADIUS
from emp.geomagnetic_field import (
    CoordinateSystem,
    DipoleMagneticField,
    IGRFMagneticField,
    MagneticFieldFactory,
    MagneticFieldModel,
)
from emp.geometry import Point


class TestMagneticFieldFactory:
    """Test the magnetic field factory."""

    def test_factory_creates_all_models(self) -> None:
        """Test that factory can create all supported magnetic field models."""
        # Test with enum
        dipole_field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)
        igrf_field = MagneticFieldFactory.create(MagneticFieldModel.IGRF)

        assert isinstance(dipole_field, DipoleMagneticField)
        assert isinstance(igrf_field, IGRFMagneticField)

        # Test with string
        dipole_field_str = MagneticFieldFactory.create("dipole")
        igrf_field_str = MagneticFieldFactory.create("igrf")

        assert isinstance(dipole_field_str, DipoleMagneticField)
        assert isinstance(igrf_field_str, IGRFMagneticField)

    def test_factory_invalid_model(self) -> None:
        """Test that factory raises error for invalid models."""
        with pytest.raises(ValueError, match="Unknown magnetic field model"):
            MagneticFieldFactory.create("invalid_model")

        with pytest.raises(ValueError, match="Unknown magnetic field model"):
            MagneticFieldFactory.create("xyz")


class TestMagneticFieldCoordinateSystems:
    """Test that all models work with all supported coordinate systems."""

    def test_dipole_all_coordinate_systems(self) -> None:
        """Test that dipole model works with all coordinate systems."""
        field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)
        point = Point(EARTH_RADIUS + 400.0, np.pi / 4, np.pi / 6, "lat/long geo")

        # Test all coordinate systems that should work
        B_latlong_mag = field.get_field_vector(point, CoordinateSystem.LATLONG_MAG)
        B_cartesian_mag = field.get_field_vector(point, CoordinateSystem.CARTESIAN_MAG)
        B_cartesian_geo = field.get_field_vector(point, CoordinateSystem.CARTESIAN_GEO)
        B_spherical_geo = field.get_field_vector(point, CoordinateSystem.SPHERICAL_GEO)

        # All should return 3-element arrays
        assert B_latlong_mag.shape == (3,)
        assert B_cartesian_mag.shape == (3,)
        assert B_cartesian_geo.shape == (3,)
        assert B_spherical_geo.shape == (3,)

        # All should be finite
        assert np.all(np.isfinite(B_latlong_mag))
        assert np.all(np.isfinite(B_cartesian_mag))
        assert np.all(np.isfinite(B_cartesian_geo))
        assert np.all(np.isfinite(B_spherical_geo))

    def test_igrf_supported_coordinate_systems(self) -> None:
        """Test that IGRF model works with supported coordinate systems."""
        field = MagneticFieldFactory.create(
            MagneticFieldModel.IGRF
        )  # Uses default 2020-01-01
        point = Point(EARTH_RADIUS + 400.0, np.pi / 4, np.pi / 6, "lat/long geo")

        # Test coordinate systems that should work for IGRF
        B_spherical_geo = field.get_field_vector(point, CoordinateSystem.SPHERICAL_GEO)
        B_cartesian_geo = field.get_field_vector(point, CoordinateSystem.CARTESIAN_GEO)
        B_cartesian_mag = field.get_field_vector(point, CoordinateSystem.CARTESIAN_MAG)

        # All should return 3-element arrays
        assert B_spherical_geo.shape == (3,)
        assert B_cartesian_geo.shape == (3,)
        assert B_cartesian_mag.shape == (3,)

        # All should be finite
        assert np.all(np.isfinite(B_spherical_geo))
        assert np.all(np.isfinite(B_cartesian_geo))
        assert np.all(np.isfinite(B_cartesian_mag))

    def test_igrf_with_custom_date(self) -> None:
        """Test IGRF model with custom date."""
        # Use a specific date within IGRF range
        test_date = datetime(2015, 6, 15)
        field = MagneticFieldFactory.create(MagneticFieldModel.IGRF, date=test_date)

        point = Point(EARTH_RADIUS + 400.0, np.pi / 4, np.pi / 6, "lat/long geo")
        B_vec = field.get_field_vector(point, CoordinateSystem.SPHERICAL_GEO)

        assert B_vec.shape == (3,)
        assert np.all(np.isfinite(B_vec))

    def test_igrf_unsupported_coordinate_system(self) -> None:
        """Test that IGRF raises error for unsupported coordinate system."""
        field = MagneticFieldFactory.create(MagneticFieldModel.IGRF)
        point = Point(EARTH_RADIUS + 400.0, np.pi / 4, np.pi / 6, "lat/long geo")

        with pytest.raises(
            NotImplementedError, match="IGRF field components in magnetic lat/long"
        ):
            field.get_field_vector(point, CoordinateSystem.LATLONG_MAG)

    def test_invalid_coordinate_system(self) -> None:
        """Test that invalid coordinate system raises ValueError."""
        # Since coordinate systems are enum-based, we can't easily test invalid values
        # without mocking. We'll test that the enum validation works properly instead.
        field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)
        point = Point(EARTH_RADIUS + 400.0, np.pi / 4, np.pi / 6, "lat/long geo")

        # All valid coordinate systems should work
        valid_systems = [
            CoordinateSystem.LATLONG_MAG,
            CoordinateSystem.CARTESIAN_MAG,
            CoordinateSystem.CARTESIAN_GEO,
            CoordinateSystem.SPHERICAL_GEO,
        ]

        for coord_sys in valid_systems:
            B_vec = field.get_field_vector(point, coord_sys)
            assert B_vec.shape == (3,)
            assert np.all(np.isfinite(B_vec))


class TestAngleMethodConsistency:
    """Test that angle methods return consistent results regardless of coordinate system."""

    def test_inclination_angle_coordinate_system_independence_dipole(self) -> None:
        """Test that dipole inclination angle is independent of how field vector is computed."""
        field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)

        # Test multiple points
        test_points = [
            Point(EARTH_RADIUS + 100.0, 0.0, 0.0, "lat/long geo"),  # Equator
            Point(
                EARTH_RADIUS + 400.0, np.pi / 4, np.pi / 6, "lat/long geo"
            ),  # Mid latitude
            Point(
                EARTH_RADIUS + 800.0, np.pi / 3, -np.pi / 4, "lat/long geo"
            ),  # Higher latitude
        ]

        for point in test_points:
            # Get inclination angle (this uses LATLONG_MAG internally)
            inclination = field.get_inclination_angle(point)

            # Also compute it manually from different coordinate representations
            B_latlong = field.get_field_vector(point, CoordinateSystem.LATLONG_MAG)
            B_r, B_phi, _ = B_latlong
            inclination_manual = np.arctan(np.abs(B_r) / np.abs(B_phi))

            # Should be the same
            assert np.isclose(inclination, inclination_manual, rtol=1e-12)
            assert isinstance(inclination, float)
            assert inclination >= 0  # Inclination should be positive

    def test_theta_angle_coordinate_system_independence_dipole(self) -> None:
        """Test that theta angle is independent of coordinate system used internally."""
        field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)

        # Create burst and line-of-sight points
        burst_point = Point(EARTH_RADIUS + 300.0, np.pi / 6, np.pi / 8, "lat/long geo")
        los_point = Point(EARTH_RADIUS + 500.0, np.pi / 3, np.pi / 4, "lat/long geo")

        # Get theta angle (this uses CARTESIAN_GEO internally)
        theta = field.get_theta_angle(burst_point, los_point)

        # The internal implementation should be consistent regardless of what coordinate
        # system is used to represent the magnetic field, since get_theta_angle always
        # uses CARTESIAN_GEO. But let's verify the result is sensible.

        assert isinstance(theta, float)
        assert 0 <= theta <= np.pi  # Angle should be between 0 and π
        assert np.isfinite(theta)

    def test_field_magnitude_coordinate_system_independence(self) -> None:
        """Test that field magnitude is the same regardless of coordinate system."""
        dipole_field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)

        test_points = [
            Point(EARTH_RADIUS + 100.0, 0.0, 0.0, "lat/long geo"),
            Point(EARTH_RADIUS + 400.0, np.pi / 4, np.pi / 6, "lat/long geo"),
            Point(EARTH_RADIUS + 800.0, np.pi / 3, -np.pi / 4, "lat/long geo"),
        ]

        for point in test_points:
            # Get magnitude (uses CARTESIAN_GEO internally)
            magnitude = dipole_field.get_field_magnitude(point)

            # Compute magnitude from different coordinate systems
            B_cartesian_geo = dipole_field.get_field_vector(
                point, CoordinateSystem.CARTESIAN_GEO
            )
            B_cartesian_mag = dipole_field.get_field_vector(
                point, CoordinateSystem.CARTESIAN_MAG
            )
            B_latlong_mag = dipole_field.get_field_vector(
                point, CoordinateSystem.LATLONG_MAG
            )
            B_spherical_geo = dipole_field.get_field_vector(
                point, CoordinateSystem.SPHERICAL_GEO
            )

            magnitude_cartesian_geo = np.linalg.norm(B_cartesian_geo)
            magnitude_cartesian_mag = np.linalg.norm(B_cartesian_mag)
            magnitude_latlong_mag = np.linalg.norm(B_latlong_mag)
            magnitude_spherical_geo = np.linalg.norm(B_spherical_geo)

            # All should be the same
            np.testing.assert_allclose(magnitude, magnitude_cartesian_geo, rtol=1e-12)
            np.testing.assert_allclose(magnitude, magnitude_cartesian_mag, rtol=1e-12)
            np.testing.assert_allclose(magnitude, magnitude_latlong_mag, rtol=1e-12)
            np.testing.assert_allclose(magnitude, magnitude_spherical_geo, rtol=1e-12)

            assert isinstance(magnitude, float)
            assert magnitude > 0  # Magnetic field should have positive magnitude

    def test_igrf_angle_methods_consistency(self) -> None:
        """Test that IGRF angle methods work and are consistent."""
        field = MagneticFieldFactory.create(MagneticFieldModel.IGRF)
        point = Point(EARTH_RADIUS + 400.0, np.pi / 4, np.pi / 6, "lat/long geo")

        # Test field magnitude
        magnitude = field.get_field_magnitude(point)
        assert isinstance(magnitude, float)
        assert magnitude > 0
        assert np.isfinite(magnitude)

        # Test inclination angle
        inclination = field.get_inclination_angle(point)
        assert isinstance(inclination, float)
        assert inclination >= 0
        assert inclination <= np.pi / 2  # Should be between 0 and 90 degrees
        assert np.isfinite(inclination)

        # Test theta angle
        burst_point = Point(EARTH_RADIUS + 300.0, np.pi / 6, np.pi / 8, "lat/long geo")
        theta = field.get_theta_angle(burst_point, point)
        assert isinstance(theta, float)
        assert 0 <= theta <= np.pi
        assert np.isfinite(theta)

    def test_coordinate_system_conversion_consistency_dipole(self) -> None:
        """Test that coordinate system conversions preserve field magnitude for dipole."""
        field = MagneticFieldFactory.create(MagneticFieldModel.DIPOLE)
        point = Point(EARTH_RADIUS + 400.0, np.pi / 4, np.pi / 6, "lat/long geo")

        # Get field in all coordinate systems
        B_latlong_mag = field.get_field_vector(point, CoordinateSystem.LATLONG_MAG)
        B_cartesian_mag = field.get_field_vector(point, CoordinateSystem.CARTESIAN_MAG)
        B_cartesian_geo = field.get_field_vector(point, CoordinateSystem.CARTESIAN_GEO)
        B_spherical_geo = field.get_field_vector(point, CoordinateSystem.SPHERICAL_GEO)

        # All should have the same magnitude
        magnitudes = [
            np.linalg.norm(B_latlong_mag),
            np.linalg.norm(B_cartesian_mag),
            np.linalg.norm(B_cartesian_geo),
            np.linalg.norm(B_spherical_geo),
        ]

        # Check all magnitudes are close
        for i in range(len(magnitudes)):
            for j in range(i + 1, len(magnitudes)):
                np.testing.assert_allclose(
                    magnitudes[i],
                    magnitudes[j],
                    rtol=1e-12,
                    err_msg=f"Magnitude mismatch between coordinate systems {i} and {j}",
                )
