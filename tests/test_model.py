"""
Unit tests for the EMPMODEL class.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate._ivp.ivp import OdeResult

from emp.constants import (
    EARTH_RADIUS,
    DEFAULT_Compton_KE,
    DEFAULT_gamma_yield_fraction,
    DEFAULT_total_yield_kt,
)
from emp.geometry import Point
from emp.model import (
    EmpLosResult,
    EmpModel,
)


class TestEMPMODEL:
    """Test the EMPMODEL class functionality."""

    @pytest.fixture  # type: ignore[misc]
    def default_points(self) -> tuple[Point, Point]:
        """Create default burst and target points for testing."""
        burst_point = Point(
            EARTH_RADIUS + 100, 0.0, 0.0, "lat/long geo"
        )  # 100km altitude
        target_point = Point(
            EARTH_RADIUS, 0.1, 0.0, "lat/long geo"
        )  # Ground level, slight offset
        return burst_point, target_point

    def test_initialization_with_defaults(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that the model initializes correctly with default parameters."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        # Check that primary parameters are set
        assert model.total_yield_kt == DEFAULT_total_yield_kt
        assert model.gamma_yield_fraction == DEFAULT_gamma_yield_fraction
        assert model.Compton_KE == DEFAULT_Compton_KE
        assert model.HOB == 100.0  # Should be calculated from burst_point

        # Check that points are stored
        assert model.burst_point == burst_point
        assert model.target_point == target_point

        # Check that derived parameters are computed
        assert hasattr(model, "Amax")
        assert hasattr(model, "R0")
        assert hasattr(model, "V0")
        assert hasattr(model, "beta")
        assert hasattr(model, "gamma")
        assert hasattr(model, "omega")
        assert hasattr(model, "q")
        assert hasattr(model, "rmin")
        assert hasattr(model, "rmax")
        assert hasattr(model, "rtarget")
        assert hasattr(model, "A")  # Now calculated automatically
        assert hasattr(model, "theta")  # Now calculated automatically
        assert hasattr(model, "Bnorm")  # Now calculated automatically

    def test_initialization_with_custom_parameters(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test initialization with custom parameters."""
        burst_point, target_point = default_points
        model = EmpModel(
            burst_point,
            target_point,
            total_yield_kt=10.0,
            Compton_KE=2.0,
            magnetic_field_model="dipole",
        )

        assert model.total_yield_kt == 10.0
        assert model.Compton_KE == 2.0

    def test_different_burst_heights(self) -> None:
        """Test model with different burst heights."""
        target_point = Point(EARTH_RADIUS, 0.1, 0.0, "lat/long geo")

        for hob in [55.0, 100.0, 200.0, 400.0]:
            burst_point = Point(EARTH_RADIUS + hob, 0.0, 0.0, "lat/long geo")
            model = EmpModel(burst_point, target_point)

            assert model.HOB == hob
            assert model.Amax > 0
            assert model.rmin >= 0
            assert model.rmax > model.rmin
            assert model.rtarget > model.rmax

    def test_RCompton_returns_positive_float(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that RCompton returns positive float values."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        test_radii = [55.0, 75.0, 100.0, 125.0]
        for r in test_radii:
            result = model.RCompton(r)
            assert isinstance(result, float)
            assert result >= 0

    def test_TCompton_returns_positive_float(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that TCompton returns positive float values."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        test_radii = [50.0, 75.0, 100.0, 125.0]
        for r in test_radii:
            result = model.TCompton(r)
            assert isinstance(result, float)
            assert result > 0
            assert result <= 1e3  # Should be capped at 1 microsecond

    def test_f_pulse_scalar_input(self, default_points: tuple[Point, Point]) -> None:
        """Test f_pulse with scalar input."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        # Test positive time
        result = model.f_pulse(1.0)
        assert isinstance(result, (float, np.number))
        assert result >= 0

        # Test negative time (should be zero due to mask)
        result = model.f_pulse(-1.0)
        assert isinstance(result, (float, np.number))
        assert result == 0.0

        # Test zero time
        result = model.f_pulse(0.0)
        assert isinstance(result, (float, np.number))
        assert result >= 0

    def test_f_pulse_array_input(self, default_points: tuple[Point, Point]) -> None:
        """Test f_pulse with array input."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        times = np.array([-1.0, 0.0, 1.0, 5.0, 10.0])
        result = model.f_pulse(times)

        assert isinstance(result, np.ndarray)
        assert result.shape == times.shape
        assert result[0] == 0.0  # Negative time should be zero
        assert all(result[1:] >= 0)  # Non-negative times should be non-negative

    def test_rho_divided_by_rho0_returns_positive(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that density ratio returns positive values."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        test_radii = [50.0, 75.0, 100.0, 125.0]
        for r in test_radii:
            result = model.rho_divided_by_rho0(r)
            assert isinstance(result, float)
            assert result > 0

    def test_mean_free_path_returns_positive(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that mean free path returns positive values."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        test_radii = [50.0, 75.0, 100.0, 125.0]
        for r in test_radii:
            result = model.mean_free_path(r)
            assert isinstance(result, float)
            assert result > 0

    def test_gCompton_returns_non_negative(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that gCompton returns non-negative values."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        test_radii = [50.0, 75.0, 100.0, 125.0]
        for r in test_radii:
            result = model.gCompton(r)
            assert isinstance(result, float)
            assert result >= 0  # Can be zero in some regions, so use >= instead of >

    def test_electron_collision_freq_at_sea_level(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test electron collision frequency calculation."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        # Test with different E field values and times
        test_cases = [
            (1e3, 1.0),  # Low E field
            (1e5, 5.0),  # High E field
            (5e4, 10.0),  # Boundary E field
        ]

        for E, t in test_cases:
            result = model.electron_collision_freq_at_sea_level(E, t)
            assert isinstance(result, float)
            assert result > 0

    def test_conductivity_returns_float(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that conductivity returns float values."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        r, t, nuC_0 = 75.0, 5.0, 1e-3
        result = model.conductivity(r, t, nuC_0)
        assert isinstance(result, float)

    def test_JCompton_components_return_float(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that Compton current components return floats."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        r, t = 75.0, 5.0

        J_theta = model.JCompton_theta(r, t)
        assert isinstance(J_theta, float)

        J_phi = model.JCompton_phi(r, t)
        assert isinstance(J_phi, float)

    def test_JCompton_KL_components_return_float(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that KL Compton current components return floats."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        r, t = 75.0, 5.0

        J_theta_KL = model.JCompton_theta_KL(r, t)
        assert isinstance(J_theta_KL, float)

        J_phi_KL = model.JCompton_phi_KL(r, t)
        assert isinstance(J_phi_KL, float)

    def test_conductivity_KL_returns_float(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that KL conductivity returns float."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        r, t, nuC_0 = 75.0, 5.0, 1e-3
        result = model.conductivity_KL(r, t, nuC_0)
        assert isinstance(result, float)

    def test_F_Seiler_components_return_float(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that Seiler F functions return floats."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        E, r, t = 1e3, 75.0, 5.0
        nuC_0 = lambda x: 1e-3  # Simple function for testing

        F_theta = model.F_theta_Seiler(E, r, t, nuC_0)
        assert isinstance(F_theta, float)

        F_phi = model.F_phi_Seiler(E, r, t, nuC_0)
        assert isinstance(F_phi, float)

    def test_solve_KL_ODEs_returns_solutions(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that solve_KL_ODEs returns proper solution objects."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        t = 5.0
        nuC_0 = lambda x: 1e-3

        sol_theta, sol_phi = model.solve_KL_ODEs(t, nuC_0)

        # Check that solutions are OdeResult objects
        assert isinstance(sol_theta, OdeResult)
        assert isinstance(sol_phi, OdeResult)

        # Check that solutions have the expected structure
        assert hasattr(sol_theta, "t")
        assert hasattr(sol_theta, "y")
        assert hasattr(sol_phi, "t")
        assert hasattr(sol_phi, "y")

        # Check that y has the right shape (1 equation)
        assert sol_theta.y.shape[0] == 1
        assert sol_phi.y.shape[0] == 1

    def test_run_returns_proper_result(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that run returns properly structured EmpLosResult."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        # Use a short time list for faster testing
        time_points = np.linspace(0, 10, 5)

        result = model.run(time_points)

        # Check that it returns EmpLosResult object
        assert isinstance(result, EmpLosResult)

        # Check that arrays have correct length
        assert len(result.E_theta_at_ground) == len(time_points)
        assert len(result.E_phi_at_ground) == len(time_points)
        assert len(result.E_norm_at_ground) == len(time_points)

        # Check that values are numeric
        for value in result.E_theta_at_ground:
            assert isinstance(value, (float, np.number))
        for value in result.E_phi_at_ground:
            assert isinstance(value, (float, np.number))
        for value in result.E_norm_at_ground:
            assert isinstance(value, (float, np.number))

        # Check that model parameters and points are stored
        assert result.model_params is not None
        assert result.burst_point_dict is not None
        assert result.target_point_dict is not None

    def test_result_save_load_functionality(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that EmpLosResult can be saved and loaded."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        # Run a quick calculation
        time_points = np.linspace(0, 5, 3)
        result = model.run(time_points)

        # Test save/load
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_result.json"
            result.save(filepath)

            assert filepath.exists()

            loaded_result = EmpLosResult.load(filepath)

            # Check key fields match
            np.testing.assert_array_equal(loaded_result.time_points, result.time_points)
            assert loaded_result.E_theta_at_ground == result.E_theta_at_ground
            assert loaded_result.burst_point_dict == result.burst_point_dict
            assert loaded_result.target_point_dict == result.target_point_dict

    def test_derived_parameters_consistency(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that derived parameters have expected relationships."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        # Beta should be less than 1 (non-relativistic or relativistic but < c)
        assert 0 < model.beta < 1

        # Gamma should be >= 1
        assert model.gamma >= 1

        # Check beta-gamma relationship
        expected_gamma = 1 / np.sqrt(1 - model.beta**2)
        assert abs(model.gamma - expected_gamma) < 1e-10

        # rmax should be greater than rmin
        assert model.rmax > model.rmin

        # rtarget should be greater than rmax
        assert model.rtarget > model.rmax

        # A should be within valid range
        assert 0 <= model.A <= model.Amax

    def test_physical_units_and_scales(
        self, default_points: tuple[Point, Point]
    ) -> None:
        """Test that computed values have reasonable physical scales."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point)

        # V0 should be a reasonable velocity (m/s)
        assert 1e6 < model.V0 < 3e8  # Between 1000 km/s and c

        # Omega should be positive (cyclotron frequency)
        assert model.omega > 0

        # q should be positive (number of electrons)
        assert model.q > 0

        # R0 should be reasonable range in meters
        assert 0.1 < model.R0 < 1000  # Between cm and km

        # Bnorm should be reasonable (Earth's magnetic field scale)
        assert 1e-6 < model.Bnorm < 1e-3  # Between 1 µT and 1 mT

    @pytest.mark.parametrize("yield_kt", [0.1, 1.0, 10.0, 100.0])  # type: ignore[misc]
    def test_different_yields(
        self, default_points: tuple[Point, Point], yield_kt: float
    ) -> None:
        """Test model behavior with different yield values."""
        burst_point, target_point = default_points
        model = EmpModel(burst_point, target_point, total_yield_kt=yield_kt)

        # gCompton should be non-negative
        g_val = model.gCompton(75.0)
        assert g_val >= 0

        # Higher yields should give higher gCompton values (roughly)
        if yield_kt > 1.0:
            model_small = EmpModel(burst_point, target_point, total_yield_kt=0.1)
            g_small = model_small.gCompton(75.0)
            # Only compare if both values are positive
            if g_val > 0 and g_small > 0:
                assert g_val > g_small

    def test_magnetic_field_models(self, default_points: tuple[Point, Point]) -> None:
        """Test different magnetic field models."""
        burst_point, target_point = default_points

        # Test dipole model
        model_dipole = EmpModel(
            burst_point, target_point, magnetic_field_model="dipole"
        )
        assert model_dipole.Bnorm > 0
        assert 0 <= model_dipole.theta <= np.pi

        # Test IGRF model
        model_igrf = EmpModel(burst_point, target_point, magnetic_field_model="igrf")
        assert model_igrf.Bnorm > 0
        assert 0 <= model_igrf.theta <= np.pi


def test_rho_divided_by_rho0_at_burst_point() -> None:
    """Test that density ratio has expected value at the burst point."""
    burst_point = Point(EARTH_RADIUS + 60, 0.0, 0.0, "lat/long geo")  # 60 km
    target_point = Point(EARTH_RADIUS, 0.0, 0.0, "lat/long geo")
    model = EmpModel(burst_point, target_point)

    # At r=0 (burst point), we're at the burst height
    # So rho/rho0 should equal exp(-HOB/SCALE_HEIGHT)
    expected = np.exp(-60.0 / 7.0)  # assuming SCALE_HEIGHT = 7 km
    result = model.rho_divided_by_rho0(r=0)

    assert np.isclose(result, expected, rtol=1e-6)


def test_rho_divided_by_rho0_at_sea_level() -> None:
    """Test that density ratio equals 1 when the ray reaches sea level."""
    burst_point = Point(EARTH_RADIUS + 55, 0.0, 0.0, "lat/long geo")  # 55 km
    target_point = Point(EARTH_RADIUS, 0.0, 0.0, "lat/long geo")  # Directly below
    model = EmpModel(burst_point, target_point)

    # When r*cos(A) = HOB, we're at sea level (altitude = 0)
    r_sea_level = model.HOB / np.cos(model.A)
    result = model.rho_divided_by_rho0(r_sea_level)

    # At sea level, rho/rho0 should equal 1
    assert np.isclose(result, 1.0, rtol=1e-6)


def test_current_components_same_order_of_magnitude() -> None:
    """Test that KL and Seiler methods return similar orders of magnitude."""
    burst_point = Point(EARTH_RADIUS + 55, 0.0, 0.0, "lat/long geo")
    target_point = Point(EARTH_RADIUS, 0.02, 0.0, "lat/long geo")
    model = EmpModel(burst_point, target_point)

    r = 20.0
    t = 10.0

    j_theta_seiler = model.JCompton_theta(r, t)
    j_theta_kl = model.JCompton_theta_KL(r, t)
    j_phi_seiler = model.JCompton_phi(r, t)
    j_phi_kl = model.JCompton_phi_KL(r, t)

    # Should be within 3 orders of magnitude of each other
    if abs(j_theta_seiler) > 1e-20 and abs(j_theta_kl) > 1e-20:
        assert abs(np.log10(abs(j_theta_seiler)) - np.log10(abs(j_theta_kl))) < 3
    if abs(j_phi_seiler) > 1e-20 and abs(j_phi_kl) > 1e-20:
        assert abs(np.log10(abs(j_phi_seiler)) - np.log10(abs(j_phi_kl))) < 3


def test_conductivity_methods_comparable() -> None:
    """Test that KL and Seiler conductivity are comparable."""
    burst_point = Point(EARTH_RADIUS + 55, 0.0, 0.0, "lat/long geo")
    target_point = Point(EARTH_RADIUS, 0.05, 0.0, "lat/long geo")
    model = EmpModel(burst_point, target_point)

    r = 20.0
    t = 10.0
    nuC_0 = 1e-3

    sigma_seiler = model.conductivity(r, t, nuC_0)
    sigma_kl = model.conductivity_KL(r, t, nuC_0)

    # Should be same order of magnitude
    if sigma_seiler > 1e-20 and sigma_kl > 1e-20:
        assert abs(np.log10(sigma_seiler) - np.log10(sigma_kl)) < 2


## ADD a test when target lies b/w the absoprtion layers
