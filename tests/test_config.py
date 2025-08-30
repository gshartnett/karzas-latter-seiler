"""
Copyright (C) 2023 by The RAND Corporation
See LICENSE and README.md for information on usage and licensing

Test config.py functions.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import (
    MagicMock,
    patch,
)

import yaml  # type: ignore

from emp.config import (
    generate_configs,
    run_configs,
)


def test_generate_configs() -> None:
    """Test that generate_configs creates the expected configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a simple base config
        base_config = {
            "model_parameters": {"total_yield_kt": 5.0},
            "geometry": {"burst_point": {"altitude_km": 100}},
        }
        base_config_path = temp_path / "base.yaml"
        with open(base_config_path, "w") as f:
            yaml.dump(base_config, f)

        # Generate configs with parameter variations
        parameters = {
            "model_parameters.total_yield_kt": [1, 5, 10],
            "geometry.burst_point.altitude_km": [100, 200],
        }

        output_dir = generate_configs(
            base_config_path=base_config_path,
            scan_name="test_scan",
            parameters=parameters,
            output_dir=temp_path / "test_output",
        )

        # Verify output directory was created
        assert output_dir.exists()

        # Verify scan_info.json was created
        scan_info_file = output_dir / "scan_info.json"
        assert scan_info_file.exists()

        with open(scan_info_file) as f:
            scan_info = json.load(f)
        assert scan_info["scan_name"] == "test_scan"
        assert scan_info["num_configs"] == 6  # 3 * 2 = 6 combinations

        # Verify config files were created
        config_files = list(output_dir.glob("*.yaml"))
        assert len(config_files) == 6

        # Verify one config file has correct parameter values
        sample_config_path = config_files[0]
        with open(sample_config_path) as f:
            sample_config = yaml.safe_load(f)

        # Should have the varied parameters
        assert "total_yield_kt" in sample_config["model_parameters"]
        assert "altitude_km" in sample_config["geometry"]["burst_point"]


def test_run_configs() -> None:
    """Test that run_configs processes configuration files and returns a summary."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock config files
        config1_path = temp_path / "config1.yaml"
        config2_path = temp_path / "config2.yaml"

        mock_config = {
            "model_parameters": {
                "total_yield_kt": 5.0,
                "gamma_yield_fraction": 0.05,
                "Compton_KE": 1.28,
                "pulse_param_a": 0.01,
                "pulse_param_b": 0.37,
                "rtol": 1e-4,
                "numerical_integration_method": "Radau",
                "magnetic_field_model": "dipole",
                "time_max": 10.0,
                "num_time_points": 10,
            },
            "geometry": {
                "burst_point": {
                    "latitude_deg": 40.0,
                    "longitude_deg": -100.0,
                    "altitude_km": 100.0,
                },
                "target_point": {
                    "latitude_deg": 41.0,
                    "longitude_deg": -101.0,
                    "altitude_km": 0.0,
                },
            },
        }
        for config_path in [config1_path, config2_path]:
            with open(config_path, "w") as f:
                yaml.dump(mock_config, f)

        # Mock the EmpModel and its results
        mock_result = MagicMock()
        mock_result.get_max_field_magnitude.return_value = 1.23e5
        mock_result.get_max_field_time.return_value = 45.6
        mock_result.save = MagicMock()

        mock_model = MagicMock()
        mock_model.run.return_value = mock_result

        with patch("emp.config.EmpModel") as mock_emp_model:
            mock_emp_model.from_yaml.return_value = mock_model

            # Run the configs
            summary = run_configs(temp_path, num_cores=1)

        # Verify summary structure
        assert isinstance(summary, dict)
        assert summary["total_configs"] == 2
        assert summary["successful_configs"] == 2
        assert summary["failed_configs"] == 0
        assert "results" in summary
        assert len(summary["results"]) == 2

        # Verify results directory was created
        results_dir = temp_path / "results"
        assert results_dir.exists()

        # Verify execution summary was saved
        execution_summary_path = results_dir / "execution_summary.json"
        assert execution_summary_path.exists()

        # Verify field statistics were calculated
        assert "field_statistics" in summary
        stats = summary["field_statistics"]
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
