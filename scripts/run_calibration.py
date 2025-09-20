import logging

logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml  # type: ignore

from emp.calibration import (
    calibrate_pulse_params,
    find_peak_characteristics,
)
from emp.config import load_and_update_config
from emp.model import EmpModel

# =============================================================================
# Data Loading and Initial Setup
# =============================================================================
print("Loading reference data and running initial simulation...")

# Load the Longmire data
df = pd.read_csv("Digitized Data/Longmire/StarfishPrime_CHAP_curve.csv", header=None, names=["x", "y"])
time_values = df["x"]
Efield_values = df["y"]
print(f"Loaded Longmire reference data: {len(time_values)} data points")

# =============================================================================
# Initial Model Run (Before Calibration)
# =============================================================================
print("\nRunning initial model simulation...")

# Instantiate the model and run
model = EmpModel.from_yaml("configs/StarfishPrime_benchmark.yaml")
initial_result = model.run()

# Find peak characteristics for both simulated and reference data
t_peak_sim, E_peak_sim = find_peak_characteristics(
    time_points=initial_result.time_points,
    Efield_points=initial_result.E_norm_at_ground,
)
t_peak_ref, E_peak_ref = find_peak_characteristics(
    time_points=time_values, Efield_points=Efield_values
)

print(f"Initial simulation completed:")
print(f"  Maximum field strength: {initial_result.get_max_field_magnitude():.2e} V/m")
print(f"  Time of maximum field: {initial_result.get_max_field_time():.2f} ns")
print(f"  Peak time (simulation): {t_peak_sim:.2f} ns")
print(f"  Peak amplitude (simulation): {E_peak_sim:.2e} V/m")
print(f"  Peak time (reference): {t_peak_ref:.2f} ns")
print(f"  Peak amplitude (reference): {E_peak_ref:.2e} V/m")

# Calculate initial errors
time_rel_error = abs(t_peak_sim - t_peak_ref) / t_peak_ref
amplitude_rel_error = abs(E_peak_sim - E_peak_ref) / E_peak_ref
print(f"  Initial time relative error: {time_rel_error:.4f}")
print(f"  Initial amplitude relative error: {amplitude_rel_error:.4f}")

# Create initial comparison plot
fig, ax = plt.subplots(1, figsize=(7, 5))
ax.plot(
    initial_result.time_points,
    initial_result.E_norm_at_ground,
    "-",
    color="k",
    linewidth=1.5,
    label="Initial simulation",
)
ax.plot(
    time_values,
    Efield_values,
    "-",
    linewidth=1.5,
    label="Longmire reference",
)
ax.scatter(t_peak_sim, E_peak_sim, color="k", s=50, label="Sim peak")
ax.scatter(t_peak_ref, E_peak_ref, color="r", s=50, label="Ref peak")

ax.set_xlabel(r"$\tau$ [ns]")
ax.set_ylabel(r"E [V/m]")
plt.minorticks_on()
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.grid(alpha=0.5)
plt.title("Surface EMP Intensity (Before Calibration)")
plt.legend()
plt.show()

# =============================================================================
# Parameter Calibration
# =============================================================================
print("\n" + "=" * 60)
print("Starting parameter calibration...")

# Define parameters to optimize
parameters_to_optimize = [
    "model_parameters.pulse_param_a",
    "model_parameters.pulse_param_b",
    # "joint_latitude_deg",
]
print(f"Parameters to optimize: {parameters_to_optimize}")

# Perform the calibration
optimal_params = calibrate_pulse_params(
    ref_time_peak=t_peak_ref,
    ref_amplitude_peak=E_peak_ref,
    parameters_to_optimize=parameters_to_optimize,
    initial_guess=[model.pulse_param_a, model.pulse_param_b],
    reference_filepath="configs/StarfishPrime_benchmark.yaml",
    max_evaluations=20,
)

print(f"Calibration completed. Optimal parameters:")
for param, value in zip(parameters_to_optimize, optimal_params):
    print(f"  {param}: {value:.6f}")

# =============================================================================
# Generate Optimized Configuration
# =============================================================================
print("\nGenerating optimized configuration...")

# Create parameter dictionary for new interface
parameters_dict = dict(zip(parameters_to_optimize, optimal_params))

# Generate optimized config using new interface
optimized_config_dict = load_and_update_config(
    reference_filepath="configs/StarfishPrime_benchmark.yaml",
    parameters=parameters_dict,
)

# Save optimized configuration
output_config_path = "configs/StarfishPrime_benchmark_optimized.yaml"
with open(output_config_path, "w") as f:
    yaml.dump(
        optimized_config_dict, f, default_flow_style=False, sort_keys=False, indent=2
    )
print(f"Optimized configuration saved to: {output_config_path}")

# =============================================================================
# Run Optimized Model and Compare Results
# =============================================================================
print("\nRunning optimized model simulation...")

# Run optimized model
model = EmpModel.from_yaml(output_config_path)
result = model.run()

# Find peak characteristics for optimized results
t_peak_opt, E_peak_opt = find_peak_characteristics(
    time_points=result.time_points, Efield_points=result.E_norm_at_ground
)

print(f"Optimized simulation completed:")
print(f"  Maximum field strength: {result.get_max_field_magnitude():.2e} V/m")
print(f"  Time of maximum field: {result.get_max_field_time():.2f} ns")
print(f"  Peak time (optimized): {t_peak_opt:.2f} ns")
print(f"  Peak amplitude (optimized): {E_peak_opt:.2e} V/m")

# Calculate final errors
time_rel_error_final = abs(t_peak_opt - t_peak_ref) / t_peak_ref
amplitude_rel_error_final = abs(E_peak_opt - E_peak_ref) / E_peak_ref
print(f"  Final time relative error: {time_rel_error_final:.4f}")
print(f"  Final amplitude relative error: {amplitude_rel_error_final:.4f}")

# Show improvement
print(f"\nCalibration improvement:")
print(f"  Time error: {time_rel_error:.4f} → {time_rel_error_final:.4f}")
print(f"  Amplitude error: {amplitude_rel_error:.4f} → {amplitude_rel_error_final:.4f}")

# Create final comparison plot
fig, ax = plt.subplots(1, figsize=(7, 5))
ax.plot(
    initial_result.time_points,
    initial_result.E_norm_at_ground,
    "--",
    color="gray",
    linewidth=1.5,
    label="Initial simulation",
)
ax.plot(
    result.time_points,
    result.E_norm_at_ground,
    "-",
    color="k",
    linewidth=1.5,
    label="Optimized simulation",
)
ax.plot(
    time_values,
    Efield_values,
    "-",
    linewidth=1.5,
    label="Longmire reference",
)
ax.scatter(t_peak_opt, E_peak_opt, color="k", s=50, label="Opt peak")
ax.scatter(t_peak_ref, E_peak_ref, color="r", s=50, label="Ref peak")

ax.set_xlabel(r"$\tau$ [ns]")
ax.set_ylabel(r"E [V/m]")
plt.minorticks_on()
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.grid(alpha=0.5)
plt.title("Surface EMP Intensity (After Calibration)")
plt.legend()
plt.show()

print("\nCalibration process completed!")
