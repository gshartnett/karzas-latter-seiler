"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""
from abc import (
    ABC,
    abstractmethod,
)
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    Type,
    Union,
)

import numpy as np
import ppigrf
from numpy.typing import NDArray

from emp.constants import (
    B0,
    EARTH_RADIUS,
)
from emp.geometry import (
    INV_ROTATION_MATRIX,
    ROTATION_MATRIX,
    Point,
    get_xvec_g_from_A_to_B,
)


class CoordinateSystem(Enum):
    """Coordinate systems for magnetic field output."""

    CARTESIAN_GEO = "cartesian_geo"
    CARTESIAN_MAG = "cartesian_mag"
    LATLONG_MAG = "latlong_mag"
    SPHERICAL_GEO = "spherical_geo"


class MagneticFieldModel(Enum):
    """Supported magnetic field models."""

    DIPOLE = "dipole"
    IGRF = "igrf"


class MagneticField(ABC):
    """Abstract base class for magnetic field models."""

    @abstractmethod
    def get_field_vector(
        self, point: Point, coordinate_system: CoordinateSystem
    ) -> NDArray[np.floating]:
        """Get magnetic field vector in the specified coordinate system."""
        pass

    def get_field_magnitude(self, point: Point) -> float:
        """Get the magnitude of the magnetic field."""
        # Always compute from Cartesian geographic for consistency
        B_vec = self.get_field_vector(point, CoordinateSystem.CARTESIAN_GEO)
        return float(np.linalg.norm(B_vec))

    def get_inclination_angle(self, point: Point) -> float:
        """Get the inclination angle of the magnetic field."""
        # Default implementation - can be overridden for model-specific calculations
        B_vec = self.get_field_vector(point, CoordinateSystem.LATLONG_MAG)
        B_r, B_phi, _ = B_vec
        return float(np.arctan(np.abs(B_r) / np.abs(B_phi)))

    def get_theta_angle(self, point_burst: Point, point_los: Point) -> float:
        """Get angle between line-of-sight vector and magnetic field."""
        B_vec = self.get_field_vector(point_los, CoordinateSystem.CARTESIAN_GEO)
        los_vec = get_xvec_g_from_A_to_B(point_burst, point_los)

        cos_theta = np.dot(los_vec, B_vec) / (
            np.linalg.norm(los_vec) * np.linalg.norm(B_vec)
        )
        return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


class DipoleMagneticField(MagneticField):
    """Dipole magnetic field model."""

    def get_field_vector(
        self, point: Point, coordinate_system: CoordinateSystem
    ) -> NDArray[np.floating]:
        """Get magnetic field vector in the specified coordinate system."""

        if coordinate_system == CoordinateSystem.LATLONG_MAG:
            # Native dipole field components in magnetic lat/long coordinates
            B_r = -2 * B0 * (EARTH_RADIUS / point.r_m) ** 3 * np.sin(point.phi_m)
            B_phi = B0 * (EARTH_RADIUS / point.r_m) ** 3 * np.cos(point.phi_m)
            B_lambd = 0.0
            return np.asarray([B_r, B_phi, B_lambd], dtype=np.floating)

        elif coordinate_system == CoordinateSystem.CARTESIAN_MAG:
            # Get magnetic lat/long components first
            B_latlong = self.get_field_vector(point, CoordinateSystem.LATLONG_MAG)
            B_r, B_phi, B_lambd = B_latlong

            # Convert to Cartesian using unit vectors
            x_hat = np.array([1, 0, 0])
            y_hat = np.array([0, 1, 0])
            z_hat = np.array([0, 0, 1])

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

            result = B_r * r_hat + B_phi * phi_hat + B_lambd * lambd_hat
            return np.asarray(result, dtype=np.floating)

        elif coordinate_system == CoordinateSystem.CARTESIAN_GEO:
            # Get magnetic Cartesian first, then rotate to geographic
            B_vec_m = self.get_field_vector(point, CoordinateSystem.CARTESIAN_MAG)
            B_vec_g = np.dot(ROTATION_MATRIX, B_vec_m)
            return np.asarray(B_vec_g, dtype=np.floating)

        elif coordinate_system == CoordinateSystem.SPHERICAL_GEO:
            # Get geographic Cartesian first, then convert to spherical
            B_vec_geo = self.get_field_vector(point, CoordinateSystem.CARTESIAN_GEO)

            # Convert to spherical coordinates
            theta = np.pi / 2 - point.phi_g  # Colatitude
            phi_azimuthal = point.lambd_g

            # Unit vectors for spherical coordinates
            x_hat = np.array([1, 0, 0])
            y_hat = np.array([0, 1, 0])
            z_hat = np.array([0, 0, 1])

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
            phi_hat = -np.sin(phi_azimuthal) * x_hat + np.cos(phi_azimuthal) * y_hat

            # Project onto spherical unit vectors
            B_r = np.dot(B_vec_geo, r_hat)
            B_theta = np.dot(B_vec_geo, theta_hat)
            B_phi_az = np.dot(B_vec_geo, phi_hat)

            return np.asarray([B_r, B_theta, B_phi_az], dtype=np.floating)

        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

    def get_inclination_angle(self, point: Point) -> float:
        """Get inclination angle with dipole-specific validation."""
        B_vec = self.get_field_vector(point, CoordinateSystem.LATLONG_MAG)
        B_r, B_phi, _ = B_vec

        # Two equivalent expressions for dipole field inclination
        expression_1 = np.arctan(np.abs(B_r) / np.abs(B_phi))
        expression_2 = np.arctan(2 * np.tan(np.abs(point.phi_m)))

        # Validate they agree (dipole-specific check)
        assert np.isclose(
            expression_1, expression_2, rtol=1e-10
        ), f"Dipole inclination expressions don't match: {expression_1} vs {expression_2}"

        return float(expression_1)


class IGRFMagneticField(MagneticField):
    """IGRF magnetic field model."""

    def __init__(self, date: datetime = None):
        """Initialize IGRF field with specific date.

        Parameters
        ----------
        date : datetime, optional
            Date for IGRF model evaluation. If None, uses January 1, 2020.
        """
        if date is None:
            self.date = datetime(2020, 1, 1)  # Fixed date within IGRF range
        else:
            self.date = date

    def get_field_vector(
        self, point: Point, coordinate_system: CoordinateSystem
    ) -> NDArray[np.floating]:
        """Get magnetic field vector in the specified coordinate system."""

        if coordinate_system == CoordinateSystem.SPHERICAL_GEO:
            # Native IGRF output in geographic spherical coordinates
            theta = np.pi / 2 - point.phi_g  # Colatitude
            phi_azimuthal = point.lambd_g  # Azimuthal angle

            # Get IGRF field (returns nT) with fixed date
            B_r, B_theta, B_phi_az = ppigrf.igrf_gc(
                point.r_g, np.degrees(theta), np.degrees(phi_azimuthal), self.date
            )

            # Convert to Tesla and flatten to 1D array
            B_r = float(B_r.flatten()[0]) * 1e-9
            B_theta = float(B_theta.flatten()[0]) * 1e-9
            B_phi_az = float(B_phi_az.flatten()[0]) * 1e-9

            return np.asarray([B_r, B_theta, B_phi_az], dtype=np.floating)

        elif coordinate_system == CoordinateSystem.CARTESIAN_GEO:
            # Get spherical first, then convert to Cartesian
            B_sph = self.get_field_vector(point, CoordinateSystem.SPHERICAL_GEO)
            B_r, B_theta, B_phi_az = B_sph

            theta = np.pi / 2 - point.phi_g
            phi_azimuthal = point.lambd_g

            # Unit vectors for spherical coordinates
            x_hat = np.array([1, 0, 0])
            y_hat = np.array([0, 1, 0])
            z_hat = np.array([0, 0, 1])

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
            phi_hat = -np.sin(phi_azimuthal) * x_hat + np.cos(phi_azimuthal) * y_hat

            result = B_r * r_hat + B_theta * theta_hat + B_phi_az * phi_hat
            return np.asarray(result, dtype=np.floating)

        elif coordinate_system == CoordinateSystem.CARTESIAN_MAG:
            # Get geographic Cartesian first, then rotate to magnetic
            B_vec_geo = self.get_field_vector(point, CoordinateSystem.CARTESIAN_GEO)
            B_vec_mag = np.dot(INV_ROTATION_MATRIX, B_vec_geo)
            return np.asarray(B_vec_mag, dtype=np.floating)

        elif coordinate_system == CoordinateSystem.LATLONG_MAG:
            # Not directly available - would need complex conversion
            raise NotImplementedError(
                "IGRF field components in magnetic lat/long coordinates not directly available"
            )

        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

    def get_inclination_angle(self, point: Point) -> float:
        """Get inclination angle from IGRF field."""
        # Get field in spherical geographic coordinates
        B_sph = self.get_field_vector(point, CoordinateSystem.SPHERICAL_GEO)
        B_r, B_theta, _ = B_sph

        # Inclination: angle between field and horizontal plane
        B_horizontal = np.abs(B_theta)
        return float(np.arctan(np.abs(B_r) / B_horizontal))


class MagneticFieldFactory:
    """Factory for creating magnetic field model instances."""

    _models: Dict[MagneticFieldModel, Type[MagneticField]] = {
        MagneticFieldModel.DIPOLE: DipoleMagneticField,
        MagneticFieldModel.IGRF: IGRFMagneticField,
    }

    @classmethod
    def create(
        cls, model: Union[MagneticFieldModel, str], **kwargs: Any
    ) -> MagneticField:
        """Create a magnetic field model instance.

        Parameters
        ----------
        model : MagneticFieldModel or str
            The magnetic field model type.
        **kwargs : Any
            Additional arguments passed to model constructor.
            For IGRF: date (datetime) - date for field evaluation

        Returns
        -------
        MagneticField
            Instance of the requested magnetic field model.
        """
        if isinstance(model, str):
            try:
                model = MagneticFieldModel(model)
            except ValueError:
                raise ValueError(f"Unknown magnetic field model: {model}")

        model_class = cls._models[model]
        instance: MagneticField = model_class(**kwargs)
        return instance
