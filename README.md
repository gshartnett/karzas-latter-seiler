[![codecov](https://codecov.io/gh/gshartnett/karzas-latter-seiler/branch/main/graph/badge.svg)](https://codecov.io/gh/gshartnett/karzas-latter-seiler)

# The Karzas-Latter-Seiler Model of a High-Altitude Electromagnetic Pulse

## Introduction
This repository contains Python code for modeling the early (E1) phase of a high-altitude electromagnetic pulse (EMP). The EMP is modeled using a variant of the well-known [Karzas-Latter model](https://journals.aps.org/pr/abstract/10.1103/PhysRev.137.B1369) introduced by [Seiler](https://apps.dtic.mil/sti/citations/ADA009208). The code may be used to produce the characteristic "smile diagrams", which depict the peak intensity of the electric field over the surface of the Earth. The diagram below corresponds to a blast detonated 100 km directly overhead Topeka, Kansas.

<img src="Topeka_smile.png" alt="Topeka" width="1000"/>

## Contents
The repository is organized as follows:
- `configs/` contains configuration yaml files:
    - `example/basic_line_of_sight.yaml` simple example for a single line of sight integration.
    - `historical_detonations/` contains config files for historical high-altitude tests, such as the Soviet K-series tests or the US StarfishPrime test.
- `emp/` contains the core code for the package:
    - `emp/model.py`: contains the EMP model class and other useful functions.
    - `emp/geometry.py`: contains code for geometrical calculations.
    - `emp/geomagnetic_field.py`: contains code for handling Earth's magnetic field.
    - `emp/constants.py`: contains constants of nature and default model parameters.
    - `emp/region_scan.py`: contains code for scanning over a range of target points and creating the "smile" diagrams.
    - `emp/HOB_yield_scan.py`: contains code for scanning over a range of height of burst (HOB) values and yields.
- `scripts/run_line_of_sight.py`: performs a single line of sight integration.
- `Seiler Digitized Data` a directory containing digitized data from select figures in the original Seiler report. The data was digitized using [this online tool](https://apps.automeris.io/wpd/).

For more information, see the Sphinx docs available [here](https://gshartnett.github.io/karzas-latter-seiler/).

## Installation

### Prerequisites
- Python 3.11 or 3.12
- [Poetry](https://python-poetry.org/) (recommended) or pip

### Using Poetry
It is recommended to perform the installation within a conda environment:

```bash
# Create and activate conda environment
conda create -n emp python=3.11
conda activate emp

# Clone repository
git clone git@github.com:gshartnett/karzas-latter-seiler.git
cd karzas-latter-seiler

# Install Poetry if not already installed
pip install poetry

# Install dependencies and package
poetry install

# Activate the poetry shell
poetry shell
```

## License
This code is Copyright (C) 2023 RAND Corporation, and provided under the MIT license. See `LICENSE` for more information.

## Contact
Developed by [Gavin Hartnett](https://gshartnett.github.io/) (email: hartnett@rand.org).
