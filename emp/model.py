"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing
"""

# imports
import argparse
import os
import pickle
import warnings

# plotting settings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from scipy.integrate import quad, solve_ivp

from emp.constants import (
    ABSORPTION_LAYER_LOWER,
    ABSORPTION_LAYER_UPPER,
    AIR_DENSITY_AT_SEA_LEVEL,
    DEFAULT_A,
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
    DEFAULT_theta,
    DEFAULT_total_yield_kt,
)

# plotting settings
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 5.0
plt.rcParams["xtick.minor.size"] = 3.0
plt.rcParams["ytick.major.size"] = 5.0
plt.rcParams["ytick.minor.size"] = 3.0
plt.rc("font", family="serif", size=14)
matplotlib.rc("text", usetex=True)
matplotlib.rc("legend", fontsize=14)
matplotlib.rcParams["axes.prop_cycle"] = cycler(
    color=["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42", "#FFB5B8"]
)


class EMPMODEL:
    """
    The EMP model class.
    """

    def __init__(
        self,
        total_yield_kt: float = DEFAULT_total_yield_kt,
        gamma_yield_fraction: float = DEFAULT_gamma_yield_fraction,
        Compton_KE: float = DEFAULT_Compton_KE,
        HOB: float = DEFAULT_HOB,
        Bnorm: float = DEFAULT_Bnorm,
        theta: float = DEFAULT_theta,
        A: float = DEFAULT_A,
        pulse_param_a: float = DEFAULT_pulse_param_a,
        pulse_param_b: float = DEFAULT_pulse_param_b,
        rtol: float = DEFAULT_rtol,
        method: str = "Radau",
    ) -> None:
        """
        Init method.

        Parameters
        ----------
        total_yield_kt : float, optional
            Total yield in kilotons.
            By default DEFAULT_total_yield_kt.
        gamma_yield_fraction : float, optional
            Fraction of yield deposited in prompt Gamma radiation.
            By default DEFAULT_gamma_yield_fraction.
        Compton_KE : float, optional
            Kinetic energy of Compton electron in MeV.
            By default DEFAULT_Compton_KE.
        HOB : float, optional
            Height of burst in km.
            By default DEFAULT_HOB.
        Bnorm : float, optional
            Geomagnetic field strength in Teslas.
            By default, DEFAULT_Bnorm.
        theta : float, optional
            Angle between line of sight vector and magnetic field in radians.
            By default, DEFAULT_theta.
        A : float, optional
            Angle between radial ray from burst point to target and normal
            in radians.
            By default, DEFAULT_A.
        pulse_param_a : float, optional
            Pulse parameter in 1/ns.
            By default, DEFAULT_pulse_param_a.
        pulse_param_b : float, optional
            Pulse parameter in 1/ns.
            By default, DEFAULT_pulse_param_b.
        rtol : float, optional
            Relative tolerance for ODE integration.
            By default, DEFAULT_rtol.
        method : str, optional
            Integration method used for solve_ivp.
            By default, 'Radau', which is well-suited
            for stiff problems.

        Yields
        ------
        None
            No returns.
        """

        # variable input parameters
        self.total_yield_kt = total_yield_kt
        self.gamma_yield_fraction = gamma_yield_fraction
        self.Compton_KE = Compton_KE
        self.HOB = HOB
        self.Bnorm = Bnorm
        self.theta = theta
        self.A = A
        self.pulse_param_a = pulse_param_a
        self.pulse_param_b = pulse_param_b
        self.rtol = rtol
        self.method = method

        # secondary/derivative parameters
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

    def RCompton(self, r):
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
        return R

    def TCompton(self, r):
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
        return T

    def f_pulse(self, t):
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

    def rho_divided_by_rho0(self, r):
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
        return np.exp(-(self.HOB - r * np.cos(self.A)) / SCALE_HEIGHT)

    def mean_free_path(self, r):
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
        return MEAN_FREE_PATH_AT_SEA_LEVEL / self.rho_divided_by_rho0(r)

    def gCompton(self, r):
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

    def gCompton_numerical_integration(self, r):
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
        float
            Compton g function, in km^(-3)
        """
        integral = np.asarray(
            [quad(lambda x: 1 / self.mean_free_path(x), 0, ri)[0] for ri in r]
        )
        out = (
            self.gamma_yield_fraction
            * self.total_yield_MeV
            / self.Compton_KE
            * np.exp(-integral)
            / (4 * np.pi * r**2 * self.mean_free_path(r))
        )
        return out

    def electron_collision_freq_at_sea_level(self, E, t):
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
        return 1e-9 * nu

    def conductivity(self, r, t, nuC_0):
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
            #            main_term = (self.pulse_param_b/self.pulse_param_a)*(self.pulse_param_a*T - 1 + np.exp(-self.pulse_param_a*T) - (np.exp(self.pulse_param_a*T) - 1)*(np.exp(-self.pulse_param_a*t) - np.exp(-self.pulse_param_a*T)))
            #            main_term -= (self.pulse_param_a/self.pulse_param_b)*(self.pulse_param_b*T - 1 + np.exp(-self.pulse_param_b*T) - (np.exp(self.pulse_param_b*T) - 1)*(np.exp(-self.pulse_param_b*t) - np.exp(-self.pulse_param_b*T)))
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
        return units_conversion_factor * prefactor * main_term

    def JCompton_theta(self, r, t):
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
        return units_conversion_factor * prefactor * main_term

    def JCompton_phi(self, r, t):
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
        return units_conversion_factor * prefactor * main_term

    def JCompton_theta_KL(self, r, t):
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

        def integrand(tau, tau_p):
            tau_tilde = (
                tau
                - (1 - self.beta * np.cos(self.theta) ** 2) * tau_p
                + self.beta
                * np.sin(self.theta) ** 2
                * np.sin(omega_ns * tau_p)
                / omega_ns
            )
            out = self.f_pulse(tau_tilde) * (np.cos(omega_ns * tau_p) - 1)
            return out

        main_term = quad(lambda tau_p: integrand(t, tau_p), 0, int_upper_limit)[0]
        return units_conversion_factor * prefactor * main_term

    def JCompton_phi_KL(self, r, t):
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

        def integrand(tau, tau_p):
            tau_tilde = (
                tau
                - (1 - self.beta * np.cos(self.theta) ** 2) * tau_p
                + self.beta
                * np.sin(self.theta) ** 2
                * np.sin(omega_ns * tau_p)
                / omega_ns
            )
            out = self.f_pulse(tau_tilde) * np.sin(omega_ns * tau_p)
            return out

        main_term = quad(lambda tau_p: integrand(t, tau_p), 0, int_upper_limit)[0]
        return units_conversion_factor * prefactor * main_term

    def conductivity_KL(self, r, t, nuC_0):
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

        def integrand(tau, tau_p):
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

        inner_integral = lambda tau: quad(
            lambda tau_p: integrand(tau, tau_p), 0, int_upper_limit
        )[0]
        outer_integral = quad(lambda tau: inner_integral(tau), 0.0, t)[0]
        return units_conversion_factor * prefactor * outer_integral

    def F_theta_Seiler(self, E, r, t, nuC_0):
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
        nuC_0 : float
            Ground-level electron collision frequency, in units of 1/ns.

        Returns
        -------
        float
            Returns the RHS of the ODE. Units are in (V/m/km).
            The factor of 1e3 is to account for the fact that r
            is measured in km.
        """
        return -E / r - (1e3 * VACUUM_PERMEABILITY * SPEED_OF_LIGHT / 2) * (
            self.JCompton_theta(r, t) + self.conductivity(r, t, nuC_0(r)) * E
        )

    def F_phi_Seiler(self, E, r, t, nuC_0):
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
        nuC_0 : float
            Ground-level electron collision frequency, in units of 1/ns.

        Returns
        -------
        float
            Returns the RHS of the ODE. Units are in (V/m/km).
            The factor of 1e3 is to account for the fact that r
            is measured in km.
        """
        return -E / r - (1e3 * VACUUM_PERMEABILITY * SPEED_OF_LIGHT / 2) * (
            self.JCompton_phi(r, t) + self.conductivity(r, t, nuC_0(r)) * E
        )

    def ODE_solve(self, t, nuC_0):
        """
        Solve the angular components of the KL ODEs.

        TO DO: consider adding radial component

        Parameters
        ----------
        t : float
            Evaluation retarded time, in ns.
        nuC_0 : float
            Ground-level electron collision frequency, in units of 1/ns.

        Returns
        -------
        Dict
            A result dictionary for each component (theta, phi).
        """
        sol_theta = solve_ivp(
            lambda r, e: self.F_theta_Seiler(e, r, t, nuC_0),
            [self.rmin, self.rmax],
            [0],
            method=self.method,
            rtol=self.rtol,
        )
        sol_phi = solve_ivp(
            lambda r, e: self.F_phi_Seiler(e, r, t, nuC_0),
            [self.rmin, self.rmax],
            [0],
            method=self.method,
            rtol=self.rtol,
        )
        return sol_theta, sol_phi

    def solver(self, tlist):
        """
        Solve the KL equations using the Seiler approximations
        for the source terms for a range of retarded times and
        return the angular components of the electric field for
        r = r_target (at the Earth's surface).

        Parameters
        ----------
        tlist : ndarray
            A numpy array of the list of evaluation times.

        Returns
        -------
        Dict
            A dictionary containing time series of the components
            and norm of the E-field evaluated at a target point
            on the Earth's surface.
        """
        out = {
            "tlist": tlist,
            "E_theta_at_ground": [],
            "E_phi_at_ground": [],
            "E_norm_at_ground": [],
        }
        rlist = np.linspace(self.rmin, self.rmax, 200)
        # E_norm_at_rmax = 0.0
        E_norm_interp = lambda x: 0.0

        for t in tlist:
            # compute the electron collision freq.
            # nuC_0 = self.electron_collision_freq_at_sea_level(E_norm_at_rmax * self.rmax/self.rtarget, t)
            nuC_0_points = np.asarray(
                [
                    self.electron_collision_freq_at_sea_level(E_norm_interp(r), t)
                    for r in rlist
                ]
            )
            nuC_0 = lambda x: np.interp(x, rlist, nuC_0_points)

            # solve the KL equations
            sol_theta, sol_phi = self.ODE_solve(t, nuC_0)

            # build an interpolation of E_norm(r)
            E_theta_interp = lambda x: np.interp(x, sol_theta.t, sol_theta.y[0])
            E_phi_interp = lambda x: np.interp(x, sol_phi.t, sol_phi.y[0])
            E_norm_interp = lambda x: np.sqrt(
                E_theta_interp(x) ** 2 + E_phi_interp(x) ** 2
            )

            # record the value at rmax
            out["E_theta_at_ground"].append(
                sol_theta.y[0, -1] * self.rmax / self.rtarget
            )
            out["E_phi_at_ground"].append(sol_phi.y[0, -1] * self.rmax / self.rtarget)
            out["E_norm_at_ground"].append(
                np.sqrt(sol_theta.y[0, -1] ** 2 + sol_phi.y[0, -1] ** 2)
                * self.rmax
                / self.rtarget
            )

        # check that the time of max EMP intensity is not the last time considered
        i_max = max(
            np.argmax(np.abs(out["E_theta_at_ground"])),
            np.argmax(np.abs(out["E_phi_at_ground"])),
        )
        if i_max == len(tlist) - 1:
            warnings.warn(
                "Warning, evolution terminated before max EMP intensity has been reached."
            )

        return out
