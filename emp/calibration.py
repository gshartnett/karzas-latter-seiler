import logging
import os
import tempfile
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np
import yaml  # type: ignore
from scipy.interpolate import (
    CubicSpline,
    interp1d,
)
from scipy.optimize import (
    minimize,
    minimize_scalar,
)

from emp.config import load_and_update_config
from emp.model import EmpModel


def calibrate_pulse_params(
    ref_time_peak: float,
    ref_amplitude_peak: float,
    parameters_to_optimize: List[str],
    initial_guess: List[float],
    reference_filepath: str,
    bounds: Optional[List[Tuple]] = None,
    max_evaluations: int = 5,
) -> np.ndarray:
    """
    Calibrate model parameters by minimizing difference between simulated and reference peak characteristics.

    Parameters
    ----------
    ref_time_peak : float
        Reference time to peak
    ref_amplitude_peak : float
        Reference amplitude peak value
    parameters_to_optimize : list of str
        List of parameter paths to optimize (dot notation)
        Example: ['model_parameters.pulse_param_a', 'model_parameters.pulse_param_b', 'joint_latitude']
        Special parameter 'joint_latitude' sets both burst_point and target_point latitudes
    initial_guess : list of float
        Initial guess values for each parameter (same order as parameters_to_optimize)
    reference_filepath : str
        Path to reference YAML configuration file
    bounds : list of tuples, optional
        Bounds for each parameter as (min, max) tuples. Same order as parameters_to_optimize.
        Example: [(0.001, 0.1), (0.1, 1.0), (30.0, 50.0)]
        If None, no bounds are applied.
    max_evaluations : int
        Maximum function evaluations for optimizer

    Returns
    -------
    np.ndarray
        Optimized parameter values (same order as parameters_to_optimize)
    """

    # Check that the lengths of parameters_to_optimize and initial_guess match
    if len(parameters_to_optimize) != len(initial_guess):
        raise ValueError(
            "Length of parameters_to_optimize must match length of initial_guess"
        )
    if bounds is not None and len(bounds) != len(parameters_to_optimize):
        raise ValueError("Length of bounds must match length of parameters_to_optimize")

    def objective_function(params: List[float]) -> float:
        # Convert parameter lists to dictionary for new interface
        parameters_dict = dict(zip(parameters_to_optimize, params))

        config_dict = load_and_update_config(reference_filepath, parameters_dict)

        # Write updated config to temporary YAML file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            yaml.dump(config_dict, temp_file, default_flow_style=False)
            temp_yaml_path = temp_file.name

        # Simulate model from temporary YAML
        model = EmpModel.from_yaml(filepath=temp_yaml_path)
        result = model.run()

        # Clean up temporary file
        os.unlink(temp_yaml_path)

        # Extract simulated peak characteristics
        sim_time_peak, sim_amplitude_peak = find_peak_characteristics(
            result.time_points, result.E_norm_at_ground
        )

        # Compute relative errors (as percentages)
        time_rel_error = abs(sim_time_peak - ref_time_peak) / ref_time_peak
        amplitude_rel_error = (
            abs(sim_amplitude_peak - ref_amplitude_peak) / ref_amplitude_peak
        )

        # Combined objective (sum of squared relative errors)
        objective = time_rel_error**2 + amplitude_rel_error**2

        return objective

    # Optimize with appropriate options for each method
    if bounds is not None:
        # Use L-BFGS-B for bounded optimization
        result = minimize(
            objective_function,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxfun": max_evaluations},  # L-BFGS-B uses 'maxfun'
        )
    else:
        # Use Nelder-Mead for unbounded optimization
        result = minimize(
            objective_function,
            initial_guess,
            method="Nelder-Mead",
            options={"maxfev": max_evaluations},  # Nelder-Mead uses 'maxfev'
        )

    if result.success:
        logging.info(f"Optimization successful - Final objective: {result.fun:.6e}")
    else:
        logging.warning(f"Optimization failed: {result.message}")

    logging.info("Optimized parameters:")
    for param_name, value in zip(parameters_to_optimize, result.x):
        logging.info(f"  {param_name}: {value:.6f}")

    return result.x


def find_peak_characteristics(time_points, Efield_points, method="cubic"):
    """
    Find peak time and peak value using interpolation.

    Parameters
    ----------
    time_points : array-like
        Time points
    Efield_points : array-like
        E-field values
    method : str
        'cubic', 'quadratic', or 'linear'

    Returns
    -------
    tuple
        (time_to_peak, peak_value)
    """

    time_points = np.array(time_points)
    Efield_points = np.array(Efield_points)

    # Sort by time
    sort_idx = np.argsort(time_points)
    t = time_points[sort_idx]
    E = Efield_points[sort_idx]

    # Create interpolation function
    if method == "cubic":
        interp_func = CubicSpline(t, E)
    else:
        interp_func = interp1d(
            t, E, kind=method, bounds_error=False, fill_value="extrapolate"
        )

    # Find peak by minimizing negative E-field
    result = minimize_scalar(
        lambda x: -interp_func(x), bounds=(t[0], t[-1]), method="bounded"
    )

    peak_time = result.x
    peak_value = interp_func(peak_time)

    return peak_time, peak_value
