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
    """Test that generate_configs creates the expected number of config files."""
    scan_name = "unit_test_tmp_dir"

    # Use a temporary directory for this test
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Generate configs
        base_config_path = "configs/example/basic_line_of_sight.yaml"

        generate_configs(
            base_config_path=base_config_path,
            scan_name=scan_name,
            output_dir=tmp_path,  # <-- temporary directory
            parameters={
                "geometry": {
                    "target_point": {
                        "latitude_deg": [40.0, 41.0, 42.0, 43.0, 44.0],
                        "longitude_deg": [-100.0, -101.0, -102.0, -103.0, -104.0],
                        "altitude_km": 0.0,
                    }
                }
            },
        )

        # Verify configs were written under tmp_dir / scan_name
        config_dir = tmp_path / scan_name
        assert config_dir.exists() and config_dir.is_dir()

        config_files = list(config_dir.glob("config_*.yaml"))
        assert len(config_files) == 25  # 5 lat × 5 lon = 25 configs

        # Verify contents of one config file
        with open(config_files[0], "r") as f:
            config = yaml.safe_load(f)

        assert "model_parameters" in config
        assert "geometry" in config
        assert "burst_point" in config["geometry"]
        assert "target_point" in config["geometry"]


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
            summary = run_configs(temp_path, temp_path, num_cores=1)

        # Verify summary structure
        assert isinstance(summary, dict)
        assert summary["total_configs"] == 2
        assert summary["successful_configs"] == 2
        assert summary["failed_configs"] == 0
        assert "results" in summary
        assert len(summary["results"]) == 2

        # Verify field statistics were calculated
        assert "field_statistics" in summary
        stats = summary["field_statistics"]
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
