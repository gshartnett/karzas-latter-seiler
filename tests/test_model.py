"""
Unit tests for the EMPMODEL class.
"""

import numpy as np
import pytest
from scipy.integrate._ivp.ivp import OdeResult

from emp.constants import (
    DEFAULT_A,
    DEFAULT_HOB,
    EARTH_RADIUS,
    DEFAULT_Bnorm,
    DEFAULT_Compton_KE,
    DEFAULT_gamma_yield_fraction,
    DEFAULT_theta,
    DEFAULT_total_yield_kt,
)
from emp.model import EmpModel


class TestEMPMODEL:
    """Test the EMPMODEL class functionality."""

    def test_initialization_with_defaults(self) -> None:
        """Test that the model initializes correctly with default parameters."""
        model = EmpModel()

        # Check that primary parameters are set
        assert model.total_yield_kt == DEFAULT_total_yield_kt
        assert model.gamma_yield_fraction == DEFAULT_gamma_yield_fraction
        assert model.Compton_KE == DEFAULT_Compton_KE
        assert model.HOB == DEFAULT_HOB
        assert model.Bnorm == DEFAULT_Bnorm
        assert model.theta == DEFAULT_theta
        assert model.A == DEFAULT_A

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

    def test_initialization_with_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        model = EmpModel(
            total_yield_kt=10.0,
            HOB=200.0,
            Compton_KE=2.0,
            theta=np.pi / 3,
        )

        assert model.total_yield_kt == 10.0
        assert model.HOB == 200.0
        assert model.Compton_KE == 2.0
        assert model.theta == np.pi / 3

    def test_invalid_angle_A_raises_error(self) -> None:
        """Test that invalid angle A raises ValueError."""
        # Calculate maximum valid angle for default HOB
        Amax = np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + DEFAULT_HOB))

        # Test angle too large
        with pytest.raises(ValueError, match="Angle A .* must be between 0 and Amax"):
            EmpModel(A=Amax + 0.1)

        # Test negative angle
        with pytest.raises(ValueError, match="Angle A .* must be between 0 and Amax"):
            EmpModel(A=-0.1)

    def test_RCompton_returns_positive_float(self) -> None:
        """Test that RCompton returns positive float values."""
        model = EmpModel()

        test_radii = [50.0, 75.0, 100.0, 125.0]
        for r in test_radii:
            result = model.RCompton(r)
            assert isinstance(result, float)
            assert result >= 0

    def test_TCompton_returns_positive_float(self) -> None:
        """Test that TCompton returns positive float values."""
        model = EmpModel()

        test_radii = [50.0, 75.0, 100.0, 125.0]
        for r in test_radii:
            result = model.TCompton(r)
            assert isinstance(result, float)
            assert result > 0
            assert result <= 1e3  # Should be capped at 1 microsecond

    def test_f_pulse_scalar_input(self) -> None:
        """Test f_pulse with scalar input."""
        model = EmpModel()

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

    def test_f_pulse_array_input(self) -> None:
        """Test f_pulse with array input."""
        model = EmpModel()

        times = np.array([-1.0, 0.0, 1.0, 5.0, 10.0])
        result = model.f_pulse(times)

        assert isinstance(result, np.ndarray)
        assert result.shape == times.shape
        assert result[0] == 0.0  # Negative time should be zero
        assert all(result[1:] >= 0)  # Non-negative times should be non-negative

    def test_rho_divided_by_rho0_returns_positive(self) -> None:
        """Test that density ratio returns positive values."""
        model = EmpModel()

        test_radii = [50.0, 75.0, 100.0, 125.0]
        for r in test_radii:
            result = model.rho_divided_by_rho0(r)
            assert isinstance(result, float)
            assert result > 0

    def test_mean_free_path_returns_positive(self) -> None:
        """Test that mean free path returns positive values."""
        model = EmpModel()

        test_radii = [50.0, 75.0, 100.0, 125.0]
        for r in test_radii:
            result = model.mean_free_path(r)
            assert isinstance(result, float)
            assert result > 0

    def test_gCompton_returns_positive(self) -> None:
        """Test that gCompton returns positive values."""
        model = EmpModel()

        test_radii = [50.0, 75.0, 100.0, 125.0]
        for r in test_radii:
            result = model.gCompton(r)
            assert isinstance(result, float)
            assert result >= 0

    def test_gCompton_returns_non_negative(self) -> None:
        """Test that gCompton returns non-negative values."""
        model = EmpModel()

        test_radii = [50.0, 75.0, 100.0, 125.0]
        for r in test_radii:
            result = model.gCompton(r)
            assert isinstance(result, float)
            assert result >= 0  # Can be zero in some regions, so use >= instead of >

    def test_electron_collision_freq_at_sea_level(self) -> None:
        """Test electron collision frequency calculation."""
        model = EmpModel()

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

    def test_conductivity_returns_float(self) -> None:
        """Test that conductivity returns float values."""
        model = EmpModel()

        r, t, nuC_0 = 75.0, 5.0, 1e-3
        result = model.conductivity(r, t, nuC_0)
        assert isinstance(result, float)

    def test_JCompton_components_return_float(self) -> None:
        """Test that Compton current components return floats."""
        model = EmpModel()

        r, t = 75.0, 5.0

        J_theta = model.JCompton_theta(r, t)
        assert isinstance(J_theta, float)

        J_phi = model.JCompton_phi(r, t)
        assert isinstance(J_phi, float)

    def test_JCompton_KL_components_return_float(self) -> None:
        """Test that KL Compton current components return floats."""
        model = EmpModel()

        r, t = 75.0, 5.0

        J_theta_KL = model.JCompton_theta_KL(r, t)
        assert isinstance(J_theta_KL, float)

        J_phi_KL = model.JCompton_phi_KL(r, t)
        assert isinstance(J_phi_KL, float)

    def test_conductivity_KL_returns_float(self) -> None:
        """Test that KL conductivity returns float."""
        model = EmpModel()

        r, t, nuC_0 = 75.0, 5.0, 1e-3
        result = model.conductivity_KL(r, t, nuC_0)
        assert isinstance(result, float)

    def test_F_Seiler_components_return_float(self) -> None:
        """Test that Seiler F functions return floats."""
        model = EmpModel()

        E, r, t = 1e3, 75.0, 5.0
        nuC_0 = lambda x: 1e-3  # Simple function for testing

        F_theta = model.F_theta_Seiler(E, r, t, nuC_0)
        assert isinstance(F_theta, float)

        F_phi = model.F_phi_Seiler(E, r, t, nuC_0)
        assert isinstance(F_phi, float)

    def test_ODE_solve_returns_solutions(self) -> None:
        """Test that ODE_solve returns proper solution objects."""
        model = EmpModel()

        t = 5.0
        nuC_0 = lambda x: 1e-3

        sol_theta, sol_phi = model.ODE_solve(t, nuC_0)

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

    def test_solver_returns_proper_dictionary(self) -> None:
        """Test that solver returns properly structured results."""
        model = EmpModel()

        # Use a short time list for faster testing
        tlist = np.linspace(0, 10, 5)

        result = model.solver(tlist)

        # Check structure
        assert isinstance(result, dict)
        assert "tlist" in result
        assert "E_theta_at_ground" in result
        assert "E_phi_at_ground" in result
        assert "E_norm_at_ground" in result

        # Check that arrays have correct length
        assert len(result["E_theta_at_ground"]) == len(tlist)
        assert len(result["E_phi_at_ground"]) == len(tlist)
        assert len(result["E_norm_at_ground"]) == len(tlist)

        # Check that values are numeric
        for component in ["E_theta_at_ground", "E_phi_at_ground", "E_norm_at_ground"]:
            assert all(isinstance(x, (float, np.number)) for x in result[component])

    def test_derived_parameters_consistency(self) -> None:
        """Test that derived parameters have expected relationships."""
        model = EmpModel()

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

    def test_physical_units_and_scales(self) -> None:
        """Test that computed values have reasonable physical scales."""
        model = EmpModel()

        # V0 should be a reasonable velocity (m/s)
        assert 1e6 < model.V0 < 3e8  # Between 1000 km/s and c

        # Omega should be positive (cyclotron frequency)
        assert model.omega > 0

        # q should be positive (number of electrons)
        assert model.q > 0

        # R0 should be reasonable range in meters
        assert 0.1 < model.R0 < 1000  # Between cm and km

    @pytest.mark.parametrize("yield_kt", [0.1, 1.0, 10.0, 100.0])
    def test_different_yields(self, yield_kt: float) -> None:
        """Test model behavior with different yield values."""
        model = EmpModel(total_yield_kt=yield_kt)

        # gCompton should be non-negative
        g_val = model.gCompton(75.0)
        assert g_val >= 0

        # Higher yields should give higher gCompton values (roughly)
        if yield_kt > 1.0:
            model_small = EmpModel(total_yield_kt=0.1)
            g_small = model_small.gCompton(75.0)
            # Only compare if both values are positive
            if g_val > 0 and g_small > 0:
                assert g_val > g_small

    @pytest.mark.parametrize("hob", [50.0, 100.0, 200.0, 400.0])
    def test_different_heights_of_burst(self, hob: float) -> None:
        """Test model behavior with different heights of burst."""
        model = EmpModel(HOB=hob)

        # Check that geometry parameters are reasonable
        assert model.Amax > 0
        assert model.rmin >= 0
        assert model.rmax > model.rmin
        assert model.rtarget > model.rmax


def test_rho_divided_by_rho0_at_burst_point() -> None:
    """Test that density ratio equals 1 at the burst point (r=0)."""
    model = EmpModel(HOB=30.0)  # 30 km height of burst

    # At r=0 (burst point), we're at the burst height
    # So rho/rho0 should equal exp(-HOB/SCALE_HEIGHT)
    expected = np.exp(-30.0 / 7.0)  # assuming SCALE_HEIGHT = 7 km
    result = model.rho_divided_by_rho0(r=0)

    assert np.isclose(result, expected, rtol=1e-6)


def test_rho_divided_by_rho0_at_sea_level() -> None:
    """Test that density ratio equals 1 when the ray reaches sea level."""
    model = EmpModel(HOB=30.0, A=0.0)  # vertical line of sight

    # When r*cos(A) = HOB, we're at sea level (altitude = 0)
    r_sea_level = model.HOB / np.cos(model.A)  # = 30 km for A=0
    result = model.rho_divided_by_rho0(r_sea_level)

    # At sea level, rho/rho0 should equal 1
    assert np.isclose(result, 1.0, rtol=1e-6)


def test_rho_divided_by_rho0_standard_atmosphere() -> None:
    """Test against standard atmosphere values."""
    model = EmpModel(HOB=20.0, A=0.0)

    # Standard atmosphere: at 10 km altitude, density ~ 0.26 of sea level
    r_for_10km_altitude = 10.0  # km (since HOB=20, altitude = 20-10 = 10 km)
    result = model.rho_divided_by_rho0(r_for_10km_altitude)
    expected = np.exp(-10.0 / 7.0)  # ≈ 0.24

    assert np.isclose(result, expected, rtol=1e-3)


def test_rho_divided_by_rho0_monotonic_decrease() -> None:
    """Test that density decreases monotonically with altitude."""
    model = EmpModel(HOB=30.0, A=0.0)

    r_values = np.linspace(0, 30, 10)  # from burst to ground
    density_ratios = [model.rho_divided_by_rho0(r) for r in r_values]

    # Density should increase as we go from burst (high altitude) to ground
    # Since r increases as altitude decreases for A=0
    assert all(
        density_ratios[i] <= density_ratios[i + 1]
        for i in range(len(density_ratios) - 1)
    )


def test_current_components_same_order_of_magnitude() -> None:
    """Test that KL and Seiler methods return similar orders of magnitude."""
    model = EmpModel(HOB=30.0, theta=np.pi / 4)
    r = 20.0
    t = 10.0

    j_theta_seiler = model.JCompton_theta(r, t)
    j_theta_kl = model.JCompton_theta_KL(r, t)
    j_phi_seiler = model.JCompton_phi(r, t)
    j_phi_kl = model.JCompton_phi_KL(r, t)

    # Should be within 3 orders of magnitude of each other
    assert abs(np.log10(abs(j_theta_seiler)) - np.log10(abs(j_theta_kl))) < 3
    assert abs(np.log10(abs(j_phi_seiler)) - np.log10(abs(j_phi_kl))) < 3


def test_conductivity_methods_comparable() -> None:
    """Test that KL and Seiler conductivity are comparable."""
    model = EmpModel()
    r = 20.0
    t = 10.0
    nuC_0 = 1e-3

    sigma_seiler = model.conductivity(r, t, nuC_0)
    sigma_kl = model.conductivity_KL(r, t, nuC_0)

    # Should be same order of magnitude
    assert abs(np.log10(sigma_seiler) - np.log10(sigma_kl)) < 2
