"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing

This module provides utilities to generate multiple configuration files from
a base template and run them in parallel.

Minimal usage example:

    from emp.config import generate_configs, run_configs

    # Generate configs for a parameter sweep
    generate_configs(
        base_config_path="configs/base.yaml",
        output_dir="configs",
        scan_name="yield_study",
        parameters={
            "model_parameters.total_yield_kt": [1, 5, 10, 20],
            "geometry.burst_point.altitude_km": [100, 150, 200]
        }
    )

    # Run all configs in parallel
    summary = run_configs("runs/yield_study", num_cores=4)
"""

import itertools
import json
import logging
import multiprocessing as mp
import time
import traceback
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import yaml  # type: ignore

from emp.model import EmpModel

# Set up logger for this module
logger = logging.getLogger(__name__)


def _flatten_parameters(
    params: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, List[Any]]:
    """
    Flatten a nested dictionary into dot-notation keys.

    Example
    -------
    {
        "target_point": {
            "latitude": [1, 2],
            "longitude": [3, 4],
            "altitude_km": 0.0,
        }
    }
    ->
    {
        "target_point.latitude": [1, 2],
        "target_point.longitude": [3, 4],
        "target_point.altitude_km": [0.0],
    }
    """
    items: Dict[str, List[Any]] = {}
    for k, v in params.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_parameters(v, new_key, sep=sep))
        else:
            # always wrap scalars into a list for consistency
            items[new_key] = v if isinstance(v, list) else [v]
    return items


def generate_configs(
    base_config_path: Union[str, Path],
    output_dir: Union[str, Path],
    scan_name: str,
    parameters: Dict[str, Any],
) -> Path:
    """
    Generate multiple configuration files from a base config by varying specified parameters.

    Parameters
    ----------
    base_config_path : Union[str, Path]
        Path to the base YAML configuration file
    output_dir : str
        Directory where generated config files will be saved.
    scan_name : str
        Name for this scan (used in directory and file names)
    parameters : Dict[str, Any]
        Dictionary mapping parameters (can be nested) to lists or scalars

    Returns
    -------
    Path
        Path to the directory where configs were generated.
    """
    base_config_path = Path(base_config_path)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")

    # Ensure output directory exists
    output_dir_with_scan_name = Path(output_dir) / scan_name
    output_dir_with_scan_name.mkdir(parents=True, exist_ok=True)

    # Flatten parameters (nested dict -> dot notation)
    flat_parameters = _flatten_parameters(parameters)

    # Generate all parameter combinations
    param_names = list(flat_parameters.keys())
    param_values = list(flat_parameters.values())
    combinations = list(itertools.product(*param_values))

    # Generate configuration files
    for i, combination in enumerate(combinations):
        config = yaml.safe_load(base_config_path.read_text())  # fresh copy each time

        # Apply parameter values
        for param_name, value in zip(param_names, combination):
            keys = param_name.split(".")
            current = config
            for key in keys[:-1]:
                if key not in current or not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value

        # Save configuration with sequential filename
        output_filename = f"config_{i:04d}.yaml"
        with open(output_dir_with_scan_name / output_filename, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

    logger.info(
        f"Generated {len(combinations)} configurations in {output_dir_with_scan_name}"
    )

    return output_dir_with_scan_name


def _run_single_config_parallel(args: Tuple[Path, Path]) -> Dict[str, Any]:
    """
    Run a single configuration file. Used by run_configs for parallel execution.

    Parameters
    ----------
    args : Tuple[Path, Path]
        Tuple of (config_path, results_dir)

    Returns
    -------
    Dict[str, Any]
        Results summary
    """
    config_path, results_dir = args
    start_time = time.time()

    result_summary = {
        "config_file": config_path.name,
        "config_path": str(config_path),
        "success": False,
        "duration": 0.0,
        "error": None,
        "max_field": None,
        "max_field_time": None,
        "result_file": None,
    }

    try:
        # Load and run model
        model = EmpModel.from_yaml(config_path)
        result = model.run()

        # Save results
        result_file = results_dir / f"{config_path.stem}_result.json"
        result.save(result_file)

        # Update summary
        result_summary.update(
            {
                "success": True,
                "max_field": result.get_max_field_magnitude(),
                "max_field_time": result.get_max_field_time(),
                "result_file": str(result_file),
            }
        )

    except Exception as e:
        result_summary.update(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )

    finally:
        result_summary["duration"] = time.time() - start_time

    return result_summary


def run_configs(
    config_dir: Union[str, Path],
    results_dir: Union[str, Path],
    num_cores: Optional[int] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run all YAML configuration files in a directory in parallel.

    Parameters
    ----------
    config_dir : Union[str, Path]
        Directory containing YAML configuration files
    results_dir : Union[str, Path]
        Directory to save results.
    num_cores : Optional[int]
        Number of parallel processes. If None, uses half of available CPU cores
    timeout : Optional[float]
        Timeout per simulation in seconds. If None, no timeout

    Returns
    -------
    Dict[str, Any]
        Summary of execution results
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise ValueError(f"Configuration directory does not exist: {config_dir}")

    # Find configuration files
    config_files: List[Path] = []
    for pattern in ["*.yaml", "*.yml"]:
        config_files.extend(config_dir.glob(pattern))

    # Filter out metadata files
    config_files = [f for f in config_files if not f.name.startswith("scan_")]
    config_files.sort()

    if not config_files:
        raise ValueError(f"No YAML configuration files found in {config_dir}")

    # Set default number of cores
    if num_cores is None:
        num_cores = max(1, mp.cpu_count() // 2)

    logger.info(f"Found {len(config_files)} configuration files")
    logger.info(f"Running with {num_cores} cores")
    if timeout:
        logger.info(f"Timeout per simulation: {timeout}s")

    # Create results directory
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run configurations in parallel
    start_time = time.time()
    results = []
    failed_configs = []

    # Prepare arguments for parallel execution
    args_list = [(config_file, results_dir) for config_file in config_files]

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(_run_single_config_parallel, args): args[0]
            for args in args_list
        }

        # Collect results with progress tracking
        completed = 0
        for future in as_completed(future_to_config, timeout=timeout):
            config_file = future_to_config[future]
            completed += 1

            try:
                result = future.result()
                results.append(result)

                if result["success"]:
                    logger.info(
                        f"[{completed:3d}/{len(config_files)}] ✓ {config_file.name} "
                        f"({result['duration']:.1f}s, E_max={result['max_field']:.2e} V/m)"
                    )
                else:
                    failed_configs.append(config_file.name)
                    logger.warning(
                        f"[{completed:3d}/{len(config_files)}] ✗ {config_file.name} "
                        f"({result['duration']:.1f}s) FAILED: {result['error']}"
                    )

            except Exception as e:
                failed_configs.append(config_file.name)
                logger.error(
                    f"[{completed:3d}/{len(config_files)}] ✗ {config_file.name} ERROR: {e}"
                )
                results.append(
                    {
                        "config_file": config_file.name,
                        "success": False,
                        "error": str(e),
                        "duration": 0,
                    }
                )

    # Calculate summary statistics
    total_time = time.time() - start_time
    successful_results = [r for r in results if r["success"]]

    summary = {
        "config_directory": str(config_dir),
        "execution_time": total_time,
        "num_cores": num_cores,
        "timeout": timeout,
        "total_configs": len(config_files),
        "successful_configs": len(successful_results),
        "failed_configs": len(failed_configs),
        "failed_config_names": failed_configs,
        "results": results,
    }

    # Add statistics if we have successful results
    if successful_results:
        max_fields = [
            r["max_field"] for r in successful_results if r["max_field"] is not None
        ]
        if max_fields:
            summary["field_statistics"] = {
                "min": float(np.min(max_fields)),
                "max": float(np.max(max_fields)),
                "mean": float(np.mean(max_fields)),
                "std": float(np.std(max_fields)),
            }

    # Save summary
    summary_file = results_dir / "execution_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    success_rate = len(successful_results) / len(config_files) * 100
    logger.info(f"\nExecution completed in {total_time:.1f}s")
    logger.info(
        f"Success rate: {len(successful_results)}/{len(config_files)} ({success_rate:.1f}%)"
    )

    if failed_configs:
        logger.warning(f"Failed configurations: {', '.join(failed_configs)}")

    if "field_statistics" in summary:
        stats = cast(Dict[str, float], summary["field_statistics"])
        logger.info(f"\nField strength statistics:")
        logger.info(f"  Range: {stats['min']:.2e} - {stats['max']:.2e} V/m")
        logger.info(f"  Mean:  {stats['mean']:.2e} ± {stats['std']:.2e} V/m")

    logger.info(f"\nResults saved to: {results_dir}")

    return summary
