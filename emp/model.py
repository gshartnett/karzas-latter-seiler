"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing

Contains the EmpModel class for simulating EMP effects, as well as the
EmpLosResult dataclass for storing results.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Union,
    cast,
)

import numpy as np
import yaml  # type: ignore
from numpy.typing import NDArray
from scipy.integrate import (
    quad,
    solve_ivp,
)
from scipy.integrate._ivp.ivp import OdeResult

from emp import geometry
from emp.constants import (
    ABSORPTION_LAYER_LOWER,
    ABSORPTION_LAYER_UPPER,
    AIR_DENSITY_AT_SEA_LEVEL,
    DEFAULT_HOB,
    EARTH_RADIUS,
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    KT_TO_MEV,
    MEAN_FREE_PATH_AT_SEA_LEVEL,
    MEV_TO_KG,
    SCALE_HEIGHT,
    SPEED_OF_LIGHT,
    VACUUM_PERMEABILITY,
    DEFAULT_Bnorm,
    DEFAULT_Compton_KE,
    DEFAULT_gamma_yield_fraction,
    DEFAULT_pulse_param_a,
    DEFAULT_pulse_param_b,
    DEFAULT_rtol,
    DEFAULT_total_yield_kt,
)
from emp.geomagnetic_field import (
    MagneticFieldFactory,
    MagneticFieldModel,
)


@dataclass
class EmpLosResult:
    """
    Result dataclass for EMP line-of-sight calculations.

    Contains time series data for electric field components and magnitude
    at ground level, along with model parameters used for the calculation.
    """

    time_points: NDArray[np.float64]
    E_theta_at_ground: List[float]
    E_phi_at_ground: List[float]
    E_norm_at_ground: List[float]

    # Store model parameters for reproducibility
    model_params: Dict[str, Any]

    # Store burst and target point coordinates in geographic lat/long system
    burst_point_dict: Dict[str, float]
    target_point_dict: Dict[str, float]

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the EmpLosResult to a JSON file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path where to save the result file.
        """
        filepath = Path(filepath)

        # Convert numpy arrays to lists for JSON serialization
        data = {
            "time_points": self.time_points.tolist(),
            "E_theta_at_ground": self.E_theta_at_ground,
            "E_phi_at_ground": self.E_phi_at_ground,
            "E_norm_at_ground": self.E_norm_at_ground,
            "model_params": self.model_params,
            "burst_point_dict": self.burst_point_dict,
            "target_point_dict": self.target_point_dict,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EmpLosResult":
        """
        Load an EmpLosResult from a JSON file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the result file to load.

        Returns
        -------
        EmpLosResult
            Loaded result object.
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)

        return cls(
            time_points=np.array(data["time_points"], dtype=np.float64),
            E_theta_at_ground=data["E_theta_at_ground"],
            E_phi_at_ground=data["E_phi_at_ground"],
            E_norm_at_ground=data["E_norm_at_ground"],
            model_params=data["model_params"],
            burst_point_dict=data["burst_point_dict"],
            target_point_dict=data["target_point_dict"],
        )

    def get_max_field_magnitude(self) -> float:
        """Get the maximum field magnitude from the result."""
        return max(self.E_norm_at_ground)

    def get_max_field_time(self) -> float:
        """Get the time at which maximum field occurs."""
        max_idx = np.argmax(self.E_norm_at_ground)
        return float(self.time_points[max_idx])


class EmpModel:
    """
    The EMP model class.
    """

    def __init__(
        self,
        burst_point: "geometry.Point",
        target_point: "geometry.Point",
        total_yield_kt: float = DEFAULT_total_yield_kt,
        gamma_yield_fraction: float = DEFAULT_gamma_yield_fraction,
        Compton_KE: float = DEFAULT_Compton_KE,
        pulse_param_a: float = DEFAULT_pulse_param_a,
        pulse_param_b: float = DEFAULT_pulse_param_b,
        rtol: float = DEFAULT_rtol,
        numerical_integration_method: str = "Radau",
        magnetic_field_model: Union[str, MagneticFieldModel] = "dipole",
        time_max: float = 100.0,
        num_time_points: int = 300,
    ) -> None:
        """
        Init method.

        Parameters
        ----------
        burst_point : geometry.Point
            Nuclear burst location.
        target_point : geometry.Point
            Target point where EMP effects are calculated.
        total_yield_kt : float, optional
            Total yield in kilotons.
            By default DEFAULT_total_yield_kt.
        gamma_yield_fraction : float, optional
            Fraction of yield deposited in prompt Gamma radiation.
            By default DEFAULT_gamma_yield_fraction.
        Compton_KE : float, optional
            Kinetic energy of Compton electron in MeV.
            By default DEFAULT_Compton_KE.
        pulse_param_a : float, optional
            Pulse parameter in 1/ns.
            By default, DEFAULT_pulse_param_a.
        pulse_param_b : float, optional
            Pulse parameter in 1/ns.
            By default, DEFAULT_pulse_param_b.
        rtol : float, optional
            Relative tolerance for ODE integration.
            By default, DEFAULT_rtol.
        numerical_integration_method : str, optional
            Integration method used for solve_ivp.
            By default, 'Radau', which is well-suited
            for stiff problems.
        magnetic_field_model : Union[str, MagneticFieldModel], optional
            Magnetic field model to use ('dipole' or 'igrf').
            By default 'dipole'.
        time_max : float, optional
            Max time to integrate to in ns.
            By default 100.0 ns.
        num_time_points : int, optional
            Number of time points to compute.
            By default 300.

        Yields
        ------
        None
            No returns.
        """

        # Store input points
        self.burst_point = burst_point
        self.target_point = target_point

        # Variable input parameters
        self.total_yield_kt = total_yield_kt
        self.gamma_yield_fraction = gamma_yield_fraction
        self.Compton_KE = Compton_KE
        self.pulse_param_a = pulse_param_a
        self.pulse_param_b = pulse_param_b
        self.rtol = rtol
        self.numerical_integration_method = numerical_integration_method
        self.time_max = time_max
        self.num_time_points = num_time_points

        # List of time points to evaluate
        self.time_points = np.linspace(0, self.time_max, self.num_time_points)

        # Height of burst from the radius of burst point
        self.HOB = burst_point.r_g - EARTH_RADIUS

        # Check line of sight
        geometry.line_of_sight_check(burst_point, target_point)

        # Calculate midway point for magnetic field calculations
        self.midway_point = geometry.get_line_of_sight_midway_point(
            burst_point, target_point
        )

        # Calculate geometric angle A
        self.A = geometry.get_A_angle(burst_point, self.midway_point)

        # Initialize magnetic field model and calculate theta, Bnorm
        self.magnetic_field = MagneticFieldFactory.create(magnetic_field_model)
        self.theta = self.magnetic_field.get_theta_angle(
            point_burst=burst_point, point_los=self.midway_point
        )
        self.Bnorm = self.magnetic_field.get_field_magnitude(self.midway_point)

        # Secondary/derivative parameters
        # max A angle (line of sight is tangent to horizon)
        self.Amax = np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + self.HOB))

        # range of Compton electrons at sea level in m
        self.R0 = (
            412
            * self.Compton_KE ** (1.265 - 0.0954 * np.log(self.Compton_KE))
            / AIR_DENSITY_AT_SEA_LEVEL
            * 1e-2
        )

        # yield in MeV
        self.total_yield_MeV = self.total_yield_kt * KT_TO_MEV

        # initial Compton velocity in m/s
        self.V0 = SPEED_OF_LIGHT * np.sqrt(
            (self.Compton_KE**2 + 2 * ELECTRON_MASS * self.Compton_KE)
            / (self.Compton_KE + ELECTRON_MASS) ** 2
        )

        # velocity in units of c
        self.beta = self.V0 / SPEED_OF_LIGHT

        # Lorentz factor
        self.gamma = np.sqrt(1 / (1 - self.beta**2))

        # Larmor frequency in units of 1/s
        self.omega = (
            ELECTRON_CHARGE * self.Bnorm / (self.gamma * ELECTRON_MASS * MEV_TO_KG)
        )

        # number of secondary electrons
        self.q = self.Compton_KE / (33 * 1e-6)

        # distance from burst point (r=0) to top of absorption layer in km
        self.rmin = (self.HOB - ABSORPTION_LAYER_UPPER) / np.cos(self.A)

        # distance from burst point (r=0) to bottom of absorption layer in km
        self.rmax = (self.HOB - ABSORPTION_LAYER_LOWER) / np.cos(self.A)

        # distance from burst point (r=0) to bottom of absorption layer in km
        self.rtarget = self.HOB / np.cos(self.A)

        # check that the angle A lies in the correct range
        if not (0 <= self.A <= self.Amax):
            raise ValueError(
                f"Angle A ({self.A:.4f}) must be between 0 and Amax ({self.Amax:.4f}) "
                f"for height of burst {self.HOB} km"
            )

    def RCompton(self, r: float) -> float:
        """
        The Compton electron stopping distance.

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.

        Returns
        -------
        float
            Stopping distance.
        """
        R = self.R0 / self.rho_divided_by_rho0(r)
        return float(R)

    def TCompton(self, r: float) -> float:
        """
        The (scaled) Compton electron lifetime.

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.

        Returns
        -------
        float
            Compton electron lifetime, in ns.
        """
        T = 1e9 * self.RCompton(r) / self.V0
        T = min(T, 1e3)  # don't let T exceed 1 micro second
        T = (1 - self.beta) * T
        return float(T)

    def f_pulse(self, t: Union[float, np.ndarray]) -> Union[float, NDArray[np.float64]]:
        """
        Normalized gamma pulse, for the difference of exponential form
        used by Seiler. This parameterization of the pulse profile has
        the benefit that it allows many integrals to be analytically
        solved. In general, other parameterizations are possible but
        they will change the equations derived by Seiler.

        Parameters
        ----------
        t : Union[float, ndarray]
            Time, in ns.

        Returns
        -------
        Union[float, ndarray]
            Gamma flux pulse profile, in 1/ns.
        """
        mask = 1.0 * (t >= 0)  # if t<0, define the pulse to be zero
        prefactor = (self.pulse_param_a * self.pulse_param_b) / (
            self.pulse_param_b - self.pulse_param_a
        )
        out = (
            mask
            * prefactor
            * (np.exp(-self.pulse_param_a * t) - np.exp(-self.pulse_param_b * t))
        )
        return out

    def rho_divided_by_rho0(
        self, r: Union[float, np.ndarray]
    ) -> Union[float, NDArray[np.float64]]:
        """
        Ratio of air density at radius r to air density at sea level.

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.

        Returns
        -------
        float
            Air density ratio, dimensionless.
        """
        result = np.exp(-(self.HOB - r * np.cos(self.A)) / SCALE_HEIGHT)
        return result

    def mean_free_path(
        self, r: Union[float, np.ndarray]
    ) -> Union[float, NDArray[np.float64]]:
        """
        Compton electron mean free path.

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.

        Returns
        -------
        float
           Mean free path, in km.
        """
        result = MEAN_FREE_PATH_AT_SEA_LEVEL / self.rho_divided_by_rho0(r)
        return result

    def gCompton(
        self, r: Union[float, np.ndarray]
    ) -> Union[float, NDArray[np.float64]]:
        """
        The g function for the creation of primary electrons, as introduced
        in Karzas, Latter Eq (4).
        The radius is measured from the burst.

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.

        Returns
        -------
        float
            Compton g function, in km^(-3)
        """
        out = (
            self.gamma_yield_fraction
            * self.total_yield_MeV
            / self.Compton_KE
            / (4 * np.pi * r**2 * self.mean_free_path(r))
        )
        out *= np.exp(
            -np.exp(-self.HOB / SCALE_HEIGHT)
            / MEAN_FREE_PATH_AT_SEA_LEVEL
            * SCALE_HEIGHT
            / np.cos(self.A)
            * (np.exp(r * np.cos(self.A) / SCALE_HEIGHT) - 1)
        )
        return out

    def gCompton_numerical_integration(
        self, r: Union[float, np.ndarray]
    ) -> Union[float, NDArray[np.float64]]:
        """
        The g function for the creation of primary electrons
        The radius is measured from the burst.
        Here g is computed using a numerical integration.
        TO DO:
        - can we vectorize the numerical integral w/ scipy?
        - should we increase the precision?

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Compton g function, in km^(-3)
        """
        r_array = np.asarray(r)
        integral = np.asarray(
            [
                quad(lambda x: 1 / self.mean_free_path(x), 0, ri)[0]
                for ri in r_array.flat
            ]
        )
        if r_array.ndim == 0:
            integral_val = integral.item()
            r_for_calc = r_array.item()
        else:
            integral_val = integral.reshape(r_array.shape)
            r_for_calc = r_array

        out = (
            self.gamma_yield_fraction
            * self.total_yield_MeV
            / self.Compton_KE
            * np.exp(-integral_val)
            / (4 * np.pi * r_for_calc**2 * self.mean_free_path(r_for_calc))
        )
        return out

    def electron_collision_freq_at_sea_level(self, E: float, t: float) -> float:
        """
        Electron collision frequency at sea level.
        Defined in Seiler eq 55.
        I modified his expression so t is in units of ns.

        Parameters
        ----------
        E : float
            Norm of the electric field, in V/m.
        t : float
            Evaluation retarded time, in ns.

        Returns
        -------
        float
            Electron collision frequency, in 1/ns.
        """
        nu_1 = (-2.5 * 1e20 * 1e-9 * t) + (4.5 * 1e12)
        if E < 5e4:
            nu_2 = (0.43 * 1e8) * E + (1.6 * 1e12)
        else:
            nu_2 = (6 * 1e7) * E + (0.8 * 1e12)
        nu_3 = 2.8 * 1e12
        nu = min(4.4 * 1e12, max(nu_1, nu_2, nu_3))
        return float(1e-9 * nu)

    def conductivity(self, r: float, t: float, nuC_0: float) -> float:
        """
        The conductivity computed using Seiler's approximation.
        Defined in Seiler eq. 67 and 70. The expression in eq. 70
        has been simlified.

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.
        t : float
            Evaluation retarded time, in ns.
        nuC_0 : float
            Ground-level electron collision frequency, in units of 1/ns.

        Returns
        -------
        float
            Conductivity, in units of C^2 s m^{-3} kg^{-1}.
        """
        T = self.TCompton(r)  # integration time, as a function of r, in units of ns
        nuC = nuC_0 * self.rho_divided_by_rho0(r)
        prefactor = (
            (ELECTRON_CHARGE**2 * self.q / ELECTRON_MASS)
            * self.gCompton(r)
            / nuC
            * 1
            / ((self.pulse_param_b - self.pulse_param_a) * T)
        )
        if t < T:
            main_term = (
                self.pulse_param_a * t - 1 + np.exp(-self.pulse_param_a * t)
            ) * (self.pulse_param_b / self.pulse_param_a) - (
                self.pulse_param_b * t - 1 + np.exp(-self.pulse_param_b * t)
            ) * (
                self.pulse_param_a / self.pulse_param_b
            )
        else:
            main_term = (self.pulse_param_b / self.pulse_param_a) * (
                self.pulse_param_a * T
                + np.exp(-self.pulse_param_a * t)
                - np.exp(self.pulse_param_a * (T - t))
            )
            main_term -= (self.pulse_param_a / self.pulse_param_b) * (
                self.pulse_param_b * T
                + np.exp(-self.pulse_param_b * t)
                - np.exp(self.pulse_param_b * (T - t))
            )
        units_conversion_factor = 1 / MEV_TO_KG * (1 / 1000) ** 3 * (1e-9)
        result = units_conversion_factor * prefactor * main_term
        return float(result)

    def JCompton_theta(self, r: float, t: float) -> float:
        """
        The theta component of the Compton current, computed using
        Seiler's approximation. Defined in Seiler eq. 65 and 68.

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.
        t : float
            Evaluation retarded time, in ns.

        Returns
        -------
        float
            J_theta, in units of C s^{-1} m^{-2}.
        """
        T = self.TCompton(r)  # integration time, as a function of z, in units of ns
        prefactor = (
            ELECTRON_CHARGE
            * self.gCompton(r)
            * np.sin(2 * self.theta)
            * (self.omega**2 / 4)
        )
        prefactor *= (
            self.V0
            / (self.pulse_param_b - self.pulse_param_a)
            * (1 - self.beta) ** (-3)
        )
        if t < T:
            main_term = (
                (self.pulse_param_a * t) ** 2
                - 2 * self.pulse_param_a * t
                + 2
                - 2 * np.exp(-self.pulse_param_a * t)
            ) * (self.pulse_param_b / self.pulse_param_a**2)
            main_term -= (
                (self.pulse_param_b * t) ** 2
                - 2 * self.pulse_param_b * t
                + 2
                - 2 * np.exp(-self.pulse_param_b * t)
            ) * (self.pulse_param_a / self.pulse_param_b**2)
        else:
            main_term = (
                np.exp(-self.pulse_param_a * t)
                * (
                    np.exp(self.pulse_param_a * T)
                    * ((self.pulse_param_a * T) ** 2 - 2 * self.pulse_param_a * T + 2)
                    - 2
                )
                * (self.pulse_param_b / self.pulse_param_a**2)
            )
            main_term -= (
                np.exp(-self.pulse_param_b * t)
                * (
                    np.exp(self.pulse_param_b * T)
                    * ((self.pulse_param_b * T) ** 2 - 2 * self.pulse_param_b * T + 2)
                    - 2
                )
                * (self.pulse_param_a / self.pulse_param_b**2)
            )
        units_conversion_factor = 1e-27
        result = units_conversion_factor * prefactor * main_term
        return float(result)

    def JCompton_phi(self, r: float, t: float) -> float:
        """
        The azimuthal component of the Compton current, computed using
        Seiler's approximation. Defined in Seiler eq. 66 and 69.

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.
        t : float
            Evaluation retarded time, in ns.

        Returns
        -------
        float
            J_phi, in units of C s^{-1} m^{-2}.
        """
        T = self.TCompton(r)  # integration time, as a function of z, in units of ns
        prefactor = (
            -ELECTRON_CHARGE
            * self.gCompton(r)
            * np.sin(self.theta)
            * self.omega
            * self.V0
            / (self.pulse_param_b - self.pulse_param_a)
            * (1 - self.beta) ** (-2)
        )
        if t < T:
            main_term = (
                self.pulse_param_a * t - 1 + np.exp(-self.pulse_param_a * t)
            ) * (self.pulse_param_b / self.pulse_param_a) - (
                self.pulse_param_b * t - 1 + np.exp(-self.pulse_param_b * t)
            ) * (
                self.pulse_param_a / self.pulse_param_b
            )
        else:
            main_term = np.exp(-self.pulse_param_a * t) * (
                np.exp(self.pulse_param_a * T) * (self.pulse_param_a * T - 1) + 1
            ) * (self.pulse_param_b / self.pulse_param_a) - np.exp(
                -self.pulse_param_b * t
            ) * (
                np.exp(self.pulse_param_b * T) * (self.pulse_param_b * T - 1) + 1
            ) * (
                self.pulse_param_a / self.pulse_param_b
            )
        units_conversion_factor = 1e-18
        result = units_conversion_factor * prefactor * main_term
        return float(result)

    def JCompton_theta_KL(self, r: float, t: float) -> float:
        """
        The theta component of the Compton current, computed using
        the KL approximation. Defined in KL eq. 15.

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.
        t : float
            Evaluation retarded time, in ns.

        Returns
        -------
        float
            J_theta, in units of C s^{-1} m^{-2}.
        """
        int_upper_limit = (
            self.RCompton(r) / self.V0 * 1e9
        )  # integration upper limit (in ns)
        int_upper_limit = min(int_upper_limit, 1e3)
        omega_ns = self.omega * 1e-9  # cyclotron frequency in 1/ns
        prefactor = (
            -ELECTRON_CHARGE
            * self.V0
            * self.gCompton(r)
            * np.sin(self.theta)
            * np.cos(self.theta)
        )
        units_conversion_factor = 1e-9

        def integrand(tau: float, tau_p: float) -> float:
            tau_tilde = (
                tau
                - (1 - self.beta * np.cos(self.theta) ** 2) * tau_p
                + self.beta
                * np.sin(self.theta) ** 2
                * np.sin(omega_ns * tau_p)
                / omega_ns
            )
            out = self.f_pulse(tau_tilde) * (np.cos(omega_ns * tau_p) - 1)
            return cast(float, out)

        main_term, _ = quad(lambda tau_p: integrand(t, tau_p), 0, int_upper_limit)
        result = cast(float, units_conversion_factor * prefactor * main_term)
        return result

    def JCompton_phi_KL(self, r: float, t: float) -> float:
        """
        The phi component of the Compton current, computed using
        the KL approximation. Defined in KL eq. 16.

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.
        t : float
            Evaluation retarded time, in ns.

        Returns
        -------
        float
            J_phi, in units of C s^{-1} m^{-2}.
        """
        int_upper_limit = (
            self.RCompton(r) / self.V0 * 1e9
        )  # integration upper limit (in ns)
        int_upper_limit = min(int_upper_limit, 1e3)
        omega_ns = self.omega * 1e-9  # cyclotron frequency in 1/ns
        prefactor = -ELECTRON_CHARGE * self.V0 * self.gCompton(r) * np.sin(self.theta)
        units_conversion_factor = 1e-9

        def integrand(tau: float, tau_p: float) -> float:
            tau_tilde = (
                tau
                - (1 - self.beta * np.cos(self.theta) ** 2) * tau_p
                + self.beta
                * np.sin(self.theta) ** 2
                * np.sin(omega_ns * tau_p)
                / omega_ns
            )
            out = self.f_pulse(tau_tilde) * np.sin(omega_ns * tau_p)
            return cast(float, out)

        main_term, _ = quad(lambda tau_p: integrand(t, tau_p), 0, int_upper_limit)
        result = units_conversion_factor * prefactor * main_term
        return cast(float, result)

    def conductivity_KL(self, r: float, t: float, nuC_0: float) -> float:
        """
        The conductivity computed using the KL approximation.
        Defined in KL eq. 53 and 13.

        Parameters
        ----------
        r : float
            Radius from burst to point of interest, in km.
        t : float
            Evaluation retarded time, in ns.
        nuC_0 : float
            Ground-level electron collision frequency, in units of 1/ns.

        Returns
        -------
        float
            Conductivity, in units of C^2 s m^{-3} kg^{-1}.
        """
        int_upper_limit = (
            self.RCompton(r) / self.V0 * 1e9
        )  # integration upper limit (in ns)
        int_upper_limit = min(int_upper_limit, 1e3)

        omega_ns = self.omega * 1e-9  # cyclotron frequency in 1/ns
        nuC = nuC_0 * self.rho_divided_by_rho0(r)
        prefactor = (
            ELECTRON_CHARGE**2
            * self.q
            / ELECTRON_MASS
            * self.gCompton(r)
            / nuC
            / (int_upper_limit * 1e-9)
        )
        units_conversion_factor = 1 / MEV_TO_KG * (1 / 1000) ** 3 * 1e-18

        def integrand(tau: float, tau_p: float) -> Union[float, np.ndarray]:
            tau_tilde = (
                tau
                - (1 - self.beta * np.cos(self.theta) ** 2) * tau_p
                + self.beta
                * np.sin(self.theta) ** 2
                * np.sin(omega_ns * tau_p)
                / omega_ns
            )
            out = self.f_pulse(tau_tilde)
            return out

        inner_integral: Callable[[float], float] = lambda tau: quad(
            lambda tau_p: integrand(tau, tau_p), 0, int_upper_limit
        )[0]
        outer_integral = quad(lambda tau: inner_integral(tau), 0.0, t)[0]
        result = units_conversion_factor * prefactor * outer_integral
        return cast(float, result)

    def F_theta_Seiler(
        self, E: float, r: float, t: float, nuC_0: Callable[[float], float]
    ) -> float:
        """
        The theta-component of the Maxwell equations,
        expressed as dE/dr = F(E).

        Parameters
        ----------
        E : float
            Electric field, in V/m.
        r : float
            Radius from burst to point of interest, in km.
        t : float
            Evaluation retarded time, in ns.
        nuC_0 : Callable[[float], float]
            Ground-level electron collision frequency function, in units of 1/ns.

        Returns
        -------
        float
            Returns the RHS of the ODE. Units are in (V/m/km).
            The factor of 1e3 is to account for the fact that r
            is measured in km.
        """
        result: float = -E / r - (1e3 * VACUUM_PERMEABILITY * SPEED_OF_LIGHT / 2) * (
            self.JCompton_theta(r, t) + self.conductivity(r, t, nuC_0(r)) * E
        )
        return result

    def F_phi_Seiler(
        self, E: float, r: float, t: float, nuC_0: Callable[[float], float]
    ) -> float:
        """
        The phi-component of the Maxwell equations,
        expressed as dE/dr = F(E).

        Parameters
        ----------
        E : float
            Electric field, in V/m.
        r : float
            Radius from burst to point of interest, in km.
        t : float
            Evaluation retarded time, in ns.
        nuC_0 : Callable[[float], float]
            Ground-level electron collision frequency function, in units of 1/ns.

        Returns
        -------
        float
            Returns the RHS of the ODE. Units are in (V/m/km).
            The factor of 1e3 is to account for the fact that r
            is measured in km.
        """
        result: float = -E / r - (1e3 * VACUUM_PERMEABILITY * SPEED_OF_LIGHT / 2) * (
            self.JCompton_phi(r, t) + self.conductivity(r, t, nuC_0(r)) * E
        )
        return result

    def solve_KL_ODEs(
        self, t: float, nuC_0: Callable[[float], float]
    ) -> Tuple[OdeResult, OdeResult]:
        """
        Solve the angular components of the KL ODEs.

        TO DO: consider adding radial component

        Parameters
        ----------
        t : float
            Evaluation retarded time, in ns.
        nuC_0 : Callable[[float], float]
            Ground-level electron collision frequency function, in units of 1/ns.

        Returns
        -------
        Tuple[OdeResult, OdeResult]
            A result tuple for each component (theta, phi).
        """
        sol_theta = solve_ivp(
            lambda r, e: self.F_theta_Seiler(e, r, t, nuC_0),
            [self.rmin, self.rmax],
            [0],
            method=self.numerical_integration_method,
            rtol=self.rtol,
        )
        sol_phi = solve_ivp(
            lambda r, e: self.F_phi_Seiler(e, r, t, nuC_0),
            [self.rmin, self.rmax],
            [0],
            method=self.numerical_integration_method,
            rtol=self.rtol,
        )
        return sol_theta, sol_phi

    def run(self) -> EmpLosResult:
        """
        Solve the KL equations using the Seiler approximations
        for the source terms for a range of retarded times and
        return the angular components of the electric field for
        r = r_target (at the Earth's surface).

        Returns
        -------
        EmpLosResult
            Result object containing time series of the components
            and norm of the E-field evaluated at a target point
            on the Earth's surface.
        """
        E_theta_at_ground: List[float] = []
        E_phi_at_ground: List[float] = []
        E_norm_at_ground: List[float] = []

        rlist = np.linspace(self.rmin, self.rmax, 200)
        # E_norm_at_rmax = 0.0
        E_norm_interp: Callable[[float], float] = lambda x: 0.0

        for t in self.time_points:
            # compute the electron collision freq.
            # nuC_0 = self.electron_collision_freq_at_sea_level(E_norm_at_rmax * self.rmax/self.rtarget, t)
            nuC_0_points = np.asarray(
                [
                    self.electron_collision_freq_at_sea_level(E_norm_interp(r), t)
                    for r in rlist
                ]
            )
            nuC_0: Callable[[float], float] = lambda x: np.interp(
                x, rlist, nuC_0_points
            )

            # solve the KL equations
            sol_theta, sol_phi = self.solve_KL_ODEs(t, nuC_0)

            # build an interpolation of E_norm(r)
            E_theta_interp: Callable[[float], float] = lambda x: np.interp(
                x, sol_theta.t, sol_theta.y[0]
            )
            E_phi_interp: Callable[[float], float] = lambda x: np.interp(
                x, sol_phi.t, sol_phi.y[0]
            )
            E_norm_interp = lambda x: np.sqrt(
                E_theta_interp(x) ** 2 + E_phi_interp(x) ** 2
            )

            # record the value at rmax
            E_theta_at_ground.append(sol_theta.y[0, -1] * self.rmax / self.rtarget)
            E_phi_at_ground.append(sol_phi.y[0, -1] * self.rmax / self.rtarget)
            E_norm_at_ground.append(
                np.sqrt(sol_theta.y[0, -1] ** 2 + sol_phi.y[0, -1] ** 2)
                * self.rmax
                / self.rtarget
            )

        # check that the time of max EMP intensity is not the last time considered
        i_max = max(
            np.argmax(np.abs(E_theta_at_ground)),
            np.argmax(np.abs(E_phi_at_ground)),
        )
        if i_max == len(self.time_points) - 1:
            import warnings

            warnings.warn(
                "Warning, evolution terminated before max EMP intensity has been reached."
            )

        # Store model parameters for reproducibility
        model_params = {
            "total_yield_kt": self.total_yield_kt,
            "gamma_yield_fraction": self.gamma_yield_fraction,
            "Compton_KE": self.Compton_KE,
            "HOB": self.HOB,
            "Bnorm": self.Bnorm,
            "theta": self.theta,
            "A": self.A,
            "pulse_param_a": self.pulse_param_a,
            "pulse_param_b": self.pulse_param_b,
            "rtol": self.rtol,
            "numerical_integration_method": self.numerical_integration_method,
            "magnetic_field_model": str(type(self.magnetic_field).__name__),
            "time_max": self.time_max,
            "num_time_points": self.num_time_points,
        }

        return EmpLosResult(
            time_points=self.time_points,
            E_theta_at_ground=E_theta_at_ground,
            E_phi_at_ground=E_phi_at_ground,
            E_norm_at_ground=E_norm_at_ground,
            model_params=model_params,
            burst_point_dict={
                "radius_km": self.burst_point.r_g,
                "latitude_rad": self.burst_point.phi_g,
                "longitude_rad": self.burst_point.lambd_g,
            },
            target_point_dict={
                "radius_km": self.target_point.r_g,
                "latitude_rad": self.target_point.phi_g,
                "longitude_rad": self.target_point.lambd_g,
            },
        )

    def to_yaml(
        self,
        filepath: Union[str, Path],
    ) -> None:
        """
        Export the EmpModel configuration to a YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path where to save the YAML configuration file.
        """
        filepath = Path(filepath)

        config = {
            "model_parameters": {
                "total_yield_kt": float(self.total_yield_kt),
                "gamma_yield_fraction": float(self.gamma_yield_fraction),
                "Compton_KE": float(self.Compton_KE),
                "pulse_param_a": float(self.pulse_param_a),
                "pulse_param_b": float(self.pulse_param_b),
                "rtol": float(self.rtol),
                "numerical_integration_method": self.numerical_integration_method,
                "magnetic_field_model": str(type(self.magnetic_field).__name__)
                .replace("MagneticField", "")
                .lower(),
                "time_max": float(self.time_max),
                "num_time_points": int(self.num_time_points),
            },
            "geometry": {
                "burst_point": {
                    "latitude_deg": float(np.degrees(self.burst_point.phi_g)),
                    "longitude_deg": float(np.degrees(self.burst_point.lambd_g)),
                    "altitude_km": float(self.HOB),
                },
                "target_point": {
                    "latitude_deg": float(np.degrees(self.target_point.phi_g)),
                    "longitude_deg": float(np.degrees(self.target_point.lambd_g)),
                    "altitude_km": float(self.target_point.r_g - EARTH_RADIUS),
                },
            },
        }

        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "EmpModel":
        """
        Create an EmpModel instance from a YAML configuration file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the YAML configuration file.

        Returns
        -------
        EmpModel
            New EmpModel instance configured from the YAML file.

        Raises
        ------
        FileNotFoundError
            If the configuration file doesn't exist.
        KeyError
            If required configuration parameters are missing.
        ValueError
            If configuration parameters have invalid values.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r") as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ["model_parameters", "geometry"]
        for section in required_sections:
            if section not in config:
                raise KeyError(f"Required section '{section}' missing from config")

        # Extract model parameters
        model_params = config["model_parameters"]
        geometry_config = config["geometry"]

        # Create burst point
        burst_config = geometry_config["burst_point"]
        burst_point = geometry.Point.from_gps_coordinates(
            latitude=burst_config["latitude_deg"],
            longitude=burst_config["longitude_deg"],
            altitude_km=burst_config["altitude_km"],
        )

        # Create target point
        target_config = geometry_config["target_point"]
        target_point = geometry.Point.from_gps_coordinates(
            latitude=target_config["latitude_deg"],
            longitude=target_config["longitude_deg"],
            altitude_km=target_config.get("altitude_km", 0.0),
        )

        # Create model instance
        model = cls(
            burst_point=burst_point,
            target_point=target_point,
            total_yield_kt=model_params["total_yield_kt"],
            gamma_yield_fraction=model_params["gamma_yield_fraction"],
            Compton_KE=model_params["Compton_KE"],
            pulse_param_a=model_params["pulse_param_a"],
            pulse_param_b=model_params["pulse_param_b"],
            rtol=model_params["rtol"],
            numerical_integration_method=model_params["numerical_integration_method"],
            magnetic_field_model=model_params["magnetic_field_model"],
            time_max=model_params["time_max"],
            num_time_points=model_params["num_time_points"],
        )

        return model
