"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing

This script runs an EMP line-of-sight calculation using a YAML
configuration file.

Usage:
    python run_line_of_sight.py                    # Uses default config: configs/example/basic_line_of_sight.yaml
    python run_line_of_sight.py my_config.yaml     # Uses specified config file
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from emp.model import EmpModel

parser = argparse.ArgumentParser(
    description="Run EMP line-of-sight simulation from YAML config"
)
parser.add_argument(
    "config",
    nargs="?",
    default="configs/example/basic_line_of_sight.yaml",
    help="Path to YAML configuration file (default: configs/example/basic_line_of_sight.yaml)",
)
args = parser.parse_args()

config_path = Path(args.config)
if not config_path.exists():
    raise ValueError(f"Config file not found: {config_path}")

print(f"Loading configuration from: {config_path}")

# Load model from YAML config
model = EmpModel.from_yaml(config_path)

# Run the simulation
result = model.run()

# Save results
result_path = config_path.parent / f"{config_path.stem}_result.json"
result.save(result_path)
print(f"Results saved to: {result_path}")

# Print summary
print(f"\nSimulation completed:")
print(f"Maximum field strength: {result.get_max_field_magnitude():.2e} V/m")
print(f"Time of maximum field: {result.get_max_field_time():.2f} ns")

# Create plot
fig, ax = plt.subplots(1, figsize=(7, 5))
ax.plot(
    result.time_points,
    result.E_norm_at_ground,
    "-",
    color="k",
    linewidth=1.5,
)
ax.set_xlabel(r"$\tau$ [ns]")
ax.set_ylabel(r"E [V/m]")
plt.minorticks_on()
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.grid(alpha=0.5)
plt.title("Surface EMP Intensity")

# Save figure
figure_path = config_path.parent / "emp_intensity.png"
plt.savefig(figure_path, bbox_inches="tight", dpi=600)
print(f"Figure saved to: {figure_path}")
plt.show()
