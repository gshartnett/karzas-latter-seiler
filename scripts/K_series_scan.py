import ppigrf
from datetime import datetime
import numpy as np
import pandas as pd
from emp.geometry import Point
from emp.constants import EARTH_RADIUS
from emp.region_scan_parallel import (
    setup_region_scan_config,
    run_scan,
)
import warnings
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
import pickle

# plot settings
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['xtick.minor.size'] = 3.0
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['lines.linewidth'] = 2
plt.rc('font', family='serif',size=16)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)
matplotlib.rcParams['axes.prop_cycle'] = cycler(
    color=['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']
    )
matplotlib.rcParams.update(
    {"axes.grid":True,
    "grid.alpha":0.75,
    "grid.linewidth":0.5}
    )
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

warnings.filterwarnings("ignore", category=FutureWarning, message=".*'unit' keyword in TimedeltaIndex construction.*")

df = pd.read_csv("K_series_tests.csv")

# loop over each run
for i in [3, 4, 5]:

    # load the run parameters
    run_id = df.iloc[i]['Name']
    date = datetime.strptime(df.iloc[i]['Date'], "%m/%d/%Y")
    yield_kt = df.iloc[i]['Yield (kt)']
    Latitude = df.iloc[i]['Latitude'] * np.pi / 180
    Longitude = df.iloc[i]['Longitude'] * np.pi / 180
    HOB = df.iloc[i]['Altitude (km)']

    # burst point
    Burst_Point = Point(
        EARTH_RADIUS + HOB,
        Latitude,
        Longitude,
        coordsys="lat/long geo",
    )

    # set-up the scan
    filepath =  f"data/{run_id}"
    setup_region_scan_config(
        filepath = filepath,
        Burst_Point=Burst_Point,
        HOB=HOB,
        total_yield_kt=yield_kt,
        N_pts_phi=100,
        N_pts_lambd=100,
        N_pts_time=120,
        b_field_type='igrf',
    )

    # perform the scan
    run_scan(filepath, parallel=True)