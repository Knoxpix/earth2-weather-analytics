# %% [markdown]
# # Earth2Studio Disaster Replay Demo: Hat Yai Flood — November 2025
#
# **Research/demo only. Not an official operational flood forecast.**
#
# ภาษาไทย: โน้ตบุ๊กนี้เป็นเดโมเชิงเทคนิคสำหรับผู้บริหาร หน่วยงานรัฐ และทีมวิศวกรรม
# เพื่อสาธิตว่า SIAM.AI + NVIDIA Earth2Studio + AI weather models สามารถประกอบเป็นระบบ
# "Disaster Replay" และ early warning decision support สำหรับเหตุการณ์น้ำท่วมหาดใหญ่
# เดือนพฤศจิกายน 2568 (November 2025) ได้อย่างไร
#
# English: This notebook is a technical demonstration of an AI weather + flood-risk
# early warning workflow. It replays the November 2025 Hat Yai flood scenario using
# Earth2Studio-style deterministic and ensemble forecast workflows, then converts
# forecast signals into local flood-risk indicators, alert levels, maps, and executive
# warning messages.
#
# **Demo questions**
#
# 1. What happened in November 2025?
# 2. What weather signals were visible before the disaster?
# 3. What forecast variables should be monitored?
# 4. When could warning levels have been escalated?
# 5. What would an AI early warning dashboard show?
#
# **Important safety note**
#
# This Notebook is for research/demo only. Do not present it as an official life-safety
# warning system unless validated by authorized meteorological, hydrological, and
# disaster management agencies.

# %% [markdown]
# ## Section 1 — Install and Environment Check
#
# This section checks the local Python, CUDA, Earth2Studio, and plotting environment.
# Missing optional dependencies should not crash the demo. If real model weights or
# remote data are unavailable, the notebook switches to **DEMO_MODE** and uses clearly
# labeled synthetic data.

# %%
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", message=r".*Glyph.*missing from font.*")
warnings.filterwarnings("ignore", message=r".*IProgress.*")
warnings.filterwarnings("ignore", message=r".*Warp DeprecationWarning.*")

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception as exc:  # pragma: no cover - notebook guard
    HAS_MATPLOTLIB = False
    print(f"Matplotlib unavailable: {exc}")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

try:
    import torch

    HAS_TORCH = True
    HAS_GPU = bool(torch.cuda.is_available())
except Exception:
    torch = None
    HAS_TORCH = False
    HAS_GPU = False

try:
    import earth2studio  # noqa: F401

    HAS_EARTH2STUDIO = True
except Exception:
    HAS_EARTH2STUDIO = False

HAS_REMOTE_DATA = False
RUN_REAL_EARTH2 = os.environ.get("RUN_REAL_EARTH2", "0").strip().lower() in {"1", "true", "yes"}
DEMO_MODE = not (HAS_EARTH2STUDIO and RUN_REAL_EARTH2)

PROJECT_ROOT = Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hatyai_flood_nov2025"
FIGURE_DIR = OUTPUT_DIR / "figures"
DET_DIR = OUTPUT_DIR / "deterministic"
ENS_DIR = OUTPUT_DIR / "ensemble"
for folder in [OUTPUT_DIR, FIGURE_DIR, DET_DIR, ENS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

if HAS_MATPLOTLIB:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 5),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "savefig.dpi": 160,
        }
    )

print("Environment summary")
print("-------------------")
print(f"Python: {sys.version.split()[0]}")
print(f"Torch available: {HAS_TORCH}")
if HAS_TORCH:
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {HAS_GPU}")
    if HAS_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Earth2Studio import status: {HAS_EARTH2STUDIO}")
print(f"Cartopy available: {HAS_CARTOPY}")
print(f"Remote weather data enabled: {HAS_REMOTE_DATA}")
print(f"RUN_REAL_EARTH2: {RUN_REAL_EARTH2}")
print(f"DEMO_MODE: {DEMO_MODE}")

# %% [markdown]
# ## Section 2 — Case Configuration
#
# พฤศจิกายน 2568 = **November 2025**. The historical window covers all of November,
# while the operational replay focuses on the escalation period around 22–26 November.
#
# Geographic focus: Hat Yai, Songkhla, Thailand, including Khlong U-Taphao, Khlong R.1,
# Songkhla Lake drainage path, lowland urban floodplain, urban drainage bottlenecks,
# and rainfall accumulation over upstream catchments.

# %%
CASE_NAME = "hatyai_flood_nov2025"
CENTER_LAT = 7.008
CENTER_LON = 100.474
BBOX = dict(lat_min=5.5, lat_max=8.5, lon_min=99.0, lon_max=102.0)

HIST_START = "2025-11-01"
HIST_END = "2025-11-30"
CRITICAL_START = "2025-11-17"
CRITICAL_END = "2025-11-28"
MAIN_ESCALATION_START = "2025-11-22"
MAIN_ESCALATION_END = "2025-11-26"
INIT_TIMES = [
    "2025-11-17 00:00",
    "2025-11-19 00:00",
    "2025-11-21 00:00",
    "2025-11-23 00:00",
    "2025-11-24 00:00",
]
FORECAST_HOURS = 240
STEP_HOURS = 6
N_STEPS = FORECAST_HOURS // STEP_HOURS
ENSEMBLE_MEMBERS = 16 if HAS_GPU else 4

KEY_WEATHER_VARIABLES = [
    "total_precipitation",
    "precip_6h",
    "precip_24h",
    "precip_72h",
    "t2m",
    "msl",
    "wind10m",
    "moisture_flux_850_proxy",
    "tcwv_or_integrated_water_vapor",
    "soil_moisture_optional",
    "runoff_proxy_optional",
]

DRAINAGE_BOTTLENECK_FACTOR = 1.18
URBAN_VULNERABILITY_FACTOR = 1.25
UPSTREAM_CATCHMENT_FACTOR = 1.15

print(f"Case: {CASE_NAME}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Forecast initializations: {INIT_TIMES}")
print(f"Ensemble members: {ENSEMBLE_MEMBERS}")

# %% [markdown]
# ## Section 3 — Data Sources and Fallback Strategy
#
# **Data hierarchy / ลำดับแหล่งข้อมูล**
#
# 1. Earth2Studio GFS / ERA5 / operational initial conditions
# 2. Public precipitation observations such as GPM IMERG / GSMaP / CHIRPS where available
# 3. Thai station rainfall / ThaiWater / TMD data if manually provided later
# 4. Synthetic demo data that mimics the November 2025 event shape
#
# The synthetic mode is clearly labeled in plots and tables. It creates a plausible event
# shape: antecedent rainfall from 17 November, an extreme peak around 22–25 November,
# river/water-level lag by 6–24 hours, and flood severity rising sharply around 24–26 November.

# %%
def daily_dates(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Return daily timestamps from start to end inclusive."""
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="1D")


def gaussian(x: np.ndarray | pd.Index, center: float, sigma: float) -> np.ndarray:
    """Simple Gaussian pulse used to build the synthetic event."""
    arr = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((arr - center) / sigma) ** 2)


def load_forecast_initial_conditions(init_time: str | pd.Timestamp) -> dict[str, Any]:
    """Load or describe forecast initial conditions.

    In demo mode this returns metadata only. In real mode this is the place to attach
    Earth2Studio GFS/ERA5-like data sources.
    """
    return {
        "init_time": pd.Timestamp(init_time),
        "source": "synthetic_demo" if DEMO_MODE else "earth2studio_gfs_or_era5",
        "available": not DEMO_MODE,
    }


def load_observed_precipitation(
    start: str | pd.Timestamp, end: str | pd.Timestamp, bbox: dict[str, float]
) -> pd.DataFrame:
    """Load observed precipitation or fall back to synthetic event observations."""
    return create_synthetic_hatyai_event(start, end, bbox)["daily"]


def load_water_level_or_proxy(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Load water level if available, otherwise return synthetic water-level proxy."""
    return create_synthetic_hatyai_event(start, end, BBOX)["daily"][["date", "water_level_proxy"]]


def create_synthetic_hatyai_event(
    start: str | pd.Timestamp, end: str | pd.Timestamp, bbox: dict[str, float]
) -> dict[str, Any]:
    """Create a synthetic Hat Yai November 2025 flood event.

    The synthetic series is designed for a presentation demo:
    - daily data from 2025-11-01 to 2025-11-30
    - antecedent rainfall buildup from 2025-11-17 onward
    - extreme single-day rainfall peak around 300-350 mm
    - water-level proxy lagging rainfall by roughly 1 day
    - flood severity rising sharply around 2025-11-24 to 2025-11-26
    """
    dates = daily_dates(start, end)
    day_index = np.arange(len(dates))
    nov_day = dates.day.values
    rng = np.random.default_rng(202511)

    base = 8 + 5 * np.sin(day_index / 3.5)
    antecedent = 32 * gaussian(nov_day, 18.5, 2.0)
    pre_peak = 82 * gaussian(nov_day, 22.2, 1.2)
    main_peak = 285 * gaussian(nov_day, 24.0, 0.85)
    tail = 55 * gaussian(nov_day, 26.0, 1.2)
    rainfall = np.clip(base + antecedent + pre_peak + main_peak + tail + rng.normal(0, 3, len(dates)), 0, None)
    rainfall = np.round(rainfall, 1)

    # Force key dates to make the event story explicit and reproducible.
    overrides = {
        "2025-11-17": 42.0,
        "2025-11-18": 65.0,
        "2025-11-19": 88.0,
        "2025-11-20": 72.0,
        "2025-11-21": 118.0,
        "2025-11-22": 172.0,
        "2025-11-23": 238.0,
        "2025-11-24": 335.0,
        "2025-11-25": 285.0,
        "2025-11-26": 142.0,
        "2025-11-27": 60.0,
        "2025-11-28": 36.0,
    }
    for key, value in overrides.items():
        idx = np.where(dates == pd.Timestamp(key))[0]
        if len(idx):
            rainfall[idx[0]] = value

    rain_series = pd.Series(rainfall, index=dates)
    rain_3d = rain_series.rolling(3, min_periods=1).sum()
    antecedent_7d = rain_series.rolling(7, min_periods=1).sum()
    lagged_rain = 0.55 * rain_series.shift(1).fillna(0) + 0.30 * rain_series.shift(2).fillna(0) + 0.15 * rain_series
    water_proxy = np.clip((lagged_rain / 280.0) + (antecedent_7d / 950.0), 0, 1.25)
    severity = np.clip(
        0.40 * (rain_series / 335.0)
        + 0.34 * (rain_3d / 750.0)
        + 0.26 * water_proxy
        + np.where((dates >= "2025-11-24") & (dates <= "2025-11-26"), 0.18, 0.0),
        0,
        1,
    )

    daily = pd.DataFrame(
        {
            "date": dates,
            "observed_or_synthetic_rainfall_mm": rain_series.values,
            "rainfall_72h_mm": rain_3d.values,
            "antecedent_7d_mm": antecedent_7d.values,
            "water_level_proxy": water_proxy.values,
            "flood_severity_index": severity,
            "data_source": "SYNTHETIC DEMO - not observed",
        }
    )
    return {"daily": daily, "bbox": bbox}


event = create_synthetic_hatyai_event(HIST_START, HIST_END, BBOX)
observed_daily = event["daily"]
display_cols = [
    "date",
    "observed_or_synthetic_rainfall_mm",
    "rainfall_72h_mm",
    "water_level_proxy",
    "flood_severity_index",
    "data_source",
]
observed_daily.loc[(observed_daily["date"] >= CRITICAL_START) & (observed_daily["date"] <= CRITICAL_END), display_cols]

# %% [markdown]
# ## Section 4 — Earth2Studio Deterministic Forecast Workflow
#
# This function follows the Earth2Studio pattern used in the examples:
#
# ```python
# from earth2studio.data import GFS
# from earth2studio.io import ZarrBackend
# from earth2studio.models.px import FCN3
# from earth2studio.run import deterministic
# ```
#
# If FCN3 is unavailable, it tries FCN, DLWP, and SFNO. If weights, GFS access, CDS
# credentials, or optional packages are not available, it falls back to synthetic forecast
# fields concentrated near Hat Yai and upstream catchments.

# %%
def make_lat_lon_grid(
    bbox: dict[str, float], resolution: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create lat/lon 1D and 2D grids."""
    lats = np.arange(bbox["lat_min"], bbox["lat_max"] + resolution / 2, resolution)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"] + resolution / 2, resolution)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lats, lons, lat2d, lon2d


def spatial_rainfall_pattern(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    center_lat: float = CENTER_LAT,
    center_lon: float = CENTER_LON,
    offset_lat: float = 0.0,
    offset_lon: float = 0.0,
) -> np.ndarray:
    """Create a rainfall pattern emphasizing Hat Yai and upstream catchments."""
    urban_core = np.exp(
        -(((lat2d - (center_lat + offset_lat)) ** 2) / (2 * 0.22**2) + ((lon2d - (center_lon + offset_lon)) ** 2) / (2 * 0.26**2))
    )
    upstream = np.exp(
        -(((lat2d - (7.45 + offset_lat)) ** 2) / (2 * 0.38**2) + ((lon2d - (100.15 + offset_lon)) ** 2) / (2 * 0.35**2))
    )
    coastal_plume = np.exp(
        -(((lat2d - (6.75 + offset_lat)) ** 2) / (2 * 0.42**2) + ((lon2d - (100.75 + offset_lon)) ** 2) / (2 * 0.55**2))
    )
    pattern = 0.52 * urban_core + 0.34 * upstream + 0.20 * coastal_plume
    return pattern / np.nanmax(pattern)


def daily_event_rainfall_at_time(valid_time: pd.Timestamp, timing_shift_hours: int = 0) -> float:
    """Map a 6-hour forecast valid time to the synthetic daily event rainfall."""
    shifted = pd.Timestamp(valid_time) - pd.Timedelta(hours=timing_shift_hours)
    row = observed_daily.loc[observed_daily["date"] == shifted.normalize()]
    if row.empty:
        return 8.0
    return float(row["observed_or_synthetic_rainfall_mm"].iloc[0])


def synthetic_forecast_dataset(
    init_time: str | pd.Timestamp,
    member: int | None = None,
    multiplier: float = 1.0,
    timing_shift_hours: int = 0,
    offset_lat: float = 0.0,
    offset_lon: float = 0.0,
    resolution: float = 0.05,
) -> xr.Dataset:
    """Create synthetic deterministic forecast fields with Earth2Studio-like dimensions."""
    init = pd.Timestamp(init_time)
    valid_times = pd.date_range(init, periods=N_STEPS + 1, freq=f"{STEP_HOURS}h")
    lead_hours = np.arange(0, FORECAST_HOURS + STEP_HOURS, STEP_HOURS)
    lats, lons, lat2d, lon2d = make_lat_lon_grid(BBOX, resolution)
    pattern = spatial_rainfall_pattern(lat2d, lon2d, offset_lat=offset_lat, offset_lon=offset_lon)

    precip = []
    wind = []
    pressure = []
    moisture = []
    for vt, lead in zip(valid_times, lead_hours):
        daily_total = daily_event_rainfall_at_time(vt, timing_shift_hours=timing_shift_hours)
        diurnal = {0: 0.22, 6: 0.20, 12: 0.25, 18: 0.33}.get(pd.Timestamp(vt).hour, 0.25)
        lead_factor = 1.0 + 0.08 * np.tanh((72 - lead) / 96)
        rain_6h = daily_total * diurnal * multiplier * lead_factor
        broad = 0.18 * rain_6h
        field = broad + rain_6h * (0.82 * pattern)
        precip.append(field.astype("float32"))

        intensity = min(daily_total / 335.0, 1.2)
        wind.append((4.0 + 10.0 * intensity * pattern + 1.5 * np.sin(np.deg2rad(lon2d * 3))).astype("float32"))
        pressure.append((1010.0 - 10.0 * intensity * pattern - 1.3 * np.cos(np.deg2rad(lat2d * 4))).astype("float32"))
        moisture.append((120.0 + 650.0 * intensity * pattern + 45.0 * (lat2d - BBOX["lat_min"])).astype("float32"))

    ds = xr.Dataset(
        {
            "precip_6h": (("time", "lat", "lon"), np.stack(precip)),
            "wind10m": (("time", "lat", "lon"), np.stack(wind)),
            "pressure": (("time", "lat", "lon"), np.stack(pressure)),
            "moisture_flux_proxy": (("time", "lat", "lon"), np.stack(moisture)),
        },
        coords={"time": valid_times, "lead_time_hours": ("time", lead_hours), "lat": lats, "lon": lons},
        attrs={
            "case": CASE_NAME,
            "init_time": str(init),
            "data_source": "SYNTHETIC DEMO - Earth2Studio fallback field",
            "member": -1 if member is None else int(member),
        },
    )
    ds["precip_24h"] = ds["precip_6h"].rolling(time=4, min_periods=1).sum()
    ds["precip_72h"] = ds["precip_6h"].rolling(time=12, min_periods=1).sum()
    ds["flood_risk_index"] = compute_gridded_flood_risk(ds)
    return ds


def compute_gridded_flood_risk(ds: xr.Dataset) -> xr.DataArray:
    """Compute a simple gridded flood-risk index from rainfall and hydrometeorological proxies."""
    p24 = ds["precip_24h"] / 220.0
    p72 = ds["precip_72h"] / 430.0
    moisture = (ds["moisture_flux_proxy"] - 100.0) / 800.0
    wind = ds["wind10m"] / 22.0
    risk = 0.42 * p24 + 0.34 * p72 + 0.16 * moisture + 0.08 * wind
    return risk.clip(0, 1).astype("float32")


def save_dataset(ds: xr.Dataset, path: Path) -> Path:
    """Save a dataset to Zarr, with NetCDF fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds.to_zarr(path, mode="w")
        return path
    except Exception as exc:
        nc_path = path.with_suffix(".nc")
        print(f"Zarr save failed for {path.name}: {exc}. Saving NetCDF fallback: {nc_path.name}")
        ds.to_netcdf(nc_path)
        return nc_path


def try_load_earth2studio_model() -> tuple[Any | None, str]:
    """Try loading a global prognostic model through Earth2Studio."""
    if not HAS_EARTH2STUDIO or not RUN_REAL_EARTH2:
        return None, "demo_mode"
    model_candidates = [
        ("FCN3", "earth2studio.models.px"),
        ("FCN", "earth2studio.models.px"),
        ("DLWP", "earth2studio.models.px"),
        ("SFNO", "earth2studio.models.px"),
    ]
    for model_name, module_name in model_candidates:
        try:
            module = __import__(module_name, fromlist=[model_name])
            model_cls = getattr(module, model_name)
            package = model_cls.load_default_package()
            model = model_cls.load_model(package)
            return model, model_name
        except Exception as exc:
            print(f"Could not load {model_name}: {exc}")
    return None, "unavailable"


def run_deterministic_forecast(init_time: str | pd.Timestamp) -> xr.Dataset:
    """Run Earth2Studio deterministic forecast or synthetic fallback."""
    model, model_name = try_load_earth2studio_model()
    output_path = DET_DIR / f"{pd.Timestamp(init_time):%Y%m%d%H}.zarr"
    if model is None:
        ds = synthetic_forecast_dataset(init_time)
        ds.attrs["model_name"] = "synthetic_earth2studio_fallback"
        save_dataset(ds, output_path)
        return ds

    try:
        import earth2studio.run as run
        from earth2studio.data import GFS
        from earth2studio.io import ZarrBackend

        data = GFS()
        io = ZarrBackend(file_name=str(output_path), backend_kwargs={"overwrite": True})
        run.deterministic([pd.Timestamp(init_time).strftime("%Y-%m-%d %H:%M")], N_STEPS, model, data, io)
        ds = xr.open_zarr(output_path)
        ds.attrs["model_name"] = model_name
        return ds
    except Exception as exc:
        print(f"Real Earth2Studio deterministic run failed for {init_time}: {exc}")
        ds = synthetic_forecast_dataset(init_time)
        ds.attrs["model_name"] = f"synthetic_fallback_after_{model_name}_failure"
        save_dataset(ds, output_path)
        return ds


deterministic_runs = {init: run_deterministic_forecast(init) for init in INIT_TIMES}
list(deterministic_runs.keys())

# %% [markdown]
# ## Section 5 — Ensemble Forecast Workflow
#
# If Earth2Studio perturbation methods are available and real execution is enabled, use
# `SphericalGaussian` or related perturbations. In demo mode, create synthetic ensemble
# members by perturbing rainfall intensity, timing, and spatial center.
#
# Ensemble dimensions: `member, init_time, time, lat, lon`

# %%
def synthetic_ensemble_for_init(init_time: str | pd.Timestamp, n_members: int = ENSEMBLE_MEMBERS) -> xr.Dataset:
    """Create a synthetic ensemble for one initialization time."""
    rng = np.random.default_rng(int(pd.Timestamp(init_time).strftime("%Y%m%d%H")))
    members = []
    for member in range(n_members):
        multiplier = rng.uniform(0.7, 1.4)
        timing_shift = int(rng.choice([-12, -6, 0, 6, 12, 18]))
        offset_lat = rng.uniform(-0.2, 0.2)
        offset_lon = rng.uniform(-0.2, 0.2)
        ds = synthetic_forecast_dataset(
            init_time,
            member=member,
            multiplier=multiplier,
            timing_shift_hours=timing_shift,
            offset_lat=offset_lat,
            offset_lon=offset_lon,
        )
        river_lag = int(rng.choice([6, 12, 18, 24]))
        ds = ds.assign_coords(member=member).expand_dims("member")
        ds["river_response_lag_hours"] = xr.DataArray([river_lag], dims=["member"])
        ds.attrs["rainfall_multiplier"] = float(multiplier)
        ds.attrs["timing_shift_hours"] = int(timing_shift)
        members.append(ds)
    ens = xr.concat(members, dim="member")
    ens = ens.assign_coords(init_time=pd.Timestamp(init_time)).expand_dims("init_time")
    ens.attrs["data_source"] = "SYNTHETIC DEMO ENSEMBLE - not operational"
    return ens


def run_ensemble_forecast(init_time: str | pd.Timestamp, n_members: int = ENSEMBLE_MEMBERS) -> xr.Dataset:
    """Run Earth2Studio ensemble workflow or synthetic fallback."""
    output_path = ENS_DIR / f"{pd.Timestamp(init_time):%Y%m%d%H}_ensemble.zarr"
    if not (HAS_EARTH2STUDIO and RUN_REAL_EARTH2):
        ds = synthetic_ensemble_for_init(init_time, n_members)
        save_dataset(ds, output_path)
        return ds

    try:
        import earth2studio.run as run
        from earth2studio.data import GFS
        from earth2studio.io import ZarrBackend
        from earth2studio.perturbation import SphericalGaussian

        model, model_name = try_load_earth2studio_model()
        if model is None:
            raise RuntimeError("No Earth2Studio model available")
        perturbation = SphericalGaussian(noise_amplitude=0.15)
        io = ZarrBackend(
            file_name=str(output_path),
            chunks={"ensemble": 1, "time": 1, "lead_time": 1},
            backend_kwargs={"overwrite": True},
        )
        run.ensemble([pd.Timestamp(init_time).strftime("%Y-%m-%d %H:%M")], N_STEPS, n_members, model, GFS(), io, perturbation)
        ds = xr.open_zarr(output_path)
        ds.attrs["model_name"] = model_name
        return ds
    except Exception as exc:
        print(f"Real Earth2Studio ensemble run failed for {init_time}: {exc}")
        ds = synthetic_ensemble_for_init(init_time, n_members)
        save_dataset(ds, output_path)
        return ds


ensemble_runs = [run_ensemble_forecast(init) for init in INIT_TIMES]
ensemble_ds = xr.concat(ensemble_runs, dim="init_time")
ensemble_ds

# %% [markdown]
# ## Section 6 — Downscaling / Local Risk Approximation
#
# Global AI weather models can be too coarse for urban flooding. Local flood hazard in
# Hat Yai depends on downscaling, terrain, drainage, river capacity, land use, and exposure.
#
# In production, this stage should use CorrDiff / regional downscaling / radar nowcasting
# / hydrological routing. Here we implement a simple local proxy:
#
# - interpolate coarse rainfall to a finer local grid
# - apply orographic / basin weighting
# - apply an urban drainage vulnerability mask
# - apply upstream catchment accumulation proxy

# %%
def local_vulnerability_layers(resolution: float = 0.02) -> xr.Dataset:
    """Create static local vulnerability proxy layers around Hat Yai."""
    lats, lons, lat2d, lon2d = make_lat_lon_grid(BBOX, resolution)
    urban_lowland = np.exp(-(((lat2d - CENTER_LAT) ** 2) / (2 * 0.18**2) + ((lon2d - CENTER_LON) ** 2) / (2 * 0.22**2)))
    upstream_catchment = np.exp(-(((lat2d - 7.45) ** 2) / (2 * 0.42**2) + ((lon2d - 100.15) ** 2) / (2 * 0.38**2)))
    drainage_bottleneck = np.exp(-(((lat2d - 7.02) ** 2) / (2 * 0.12**2) + ((lon2d - 100.42) ** 2) / (2 * 0.16**2)))
    lake_backwater_proxy = np.exp(-(((lat2d - 7.25) ** 2) / (2 * 0.55**2) + ((lon2d - 100.55) ** 2) / (2 * 0.35**2)))
    vulnerability = np.clip(
        0.38 * urban_lowland + 0.27 * upstream_catchment + 0.22 * drainage_bottleneck + 0.13 * lake_backwater_proxy,
        0,
        1,
    )
    return xr.Dataset(
        {
            "urban_lowland": (("lat", "lon"), urban_lowland.astype("float32")),
            "upstream_catchment": (("lat", "lon"), upstream_catchment.astype("float32")),
            "drainage_bottleneck": (("lat", "lon"), drainage_bottleneck.astype("float32")),
            "lake_backwater_proxy": (("lat", "lon"), lake_backwater_proxy.astype("float32")),
            "vulnerability": (("lat", "lon"), vulnerability.astype("float32")),
        },
        coords={"lat": lats, "lon": lons},
    )


def demo_downscale_forecast(ds: xr.Dataset, resolution: float = 0.02) -> xr.Dataset:
    """Interpolate rainfall fields to a finer grid and apply local risk weighting."""
    layers = local_vulnerability_layers(resolution)
    fine = ds[["precip_6h", "precip_24h", "precip_72h", "flood_risk_index"]].interp(lat=layers.lat, lon=layers.lon)
    fine["local_precip_24h_weighted"] = fine["precip_24h"] * (1 + 0.45 * layers["upstream_catchment"] + 0.25 * layers["urban_lowland"])
    fine["local_flood_risk_index"] = (
        0.55 * fine["flood_risk_index"]
        + 0.30 * (fine["local_precip_24h_weighted"] / 260.0)
        + 0.15 * layers["vulnerability"]
    ).clip(0, 1)
    return xr.merge([fine, layers])


primary_init = "2025-11-21 00:00"
downscaled_ds = demo_downscale_forecast(deterministic_runs[primary_init])
save_dataset(downscaled_ds, OUTPUT_DIR / "downscaled_local_risk_demo.zarr")
downscaled_ds

# Optional CorrDiff hook; intentionally not executed in demo mode.
def optional_corrdiff_downscaling_hook() -> None:
    """Placeholder for CorrDiff-based downscaling in a production notebook."""
    # from earth2studio.models.dx import CorrDiff
    # corrdiff = CorrDiff.load_model(CorrDiff.load_default_package())
    # high_resolution = corrdiff(...)
    return None

# %% [markdown]
# ## Section 7 — Retrospective Event Timeline
#
# This section creates an operational-style daily timeline:
#
# - observed/synthetic rainfall
# - forecast 24h and 72h rainfall
# - ensemble exceedance probabilities
# - water-level proxy
# - flood-risk index
# - recommended alert level and suggested action

# %%
THRESHOLDS = {
    "yellow_24h": 100.0,
    "orange_24h": 150.0,
    "orange_72h": 250.0,
    "red_24h": 200.0,
    "red_72h": 350.0,
    "purple_24h": 300.0,
    "purple_risk": 0.85,
}

ALERT_ORDER = ["GREEN", "YELLOW", "ORANGE", "RED", "PURPLE"]
ALERT_SCORE = {level: i for i, level in enumerate(ALERT_ORDER)}
ALERT_COLOR = {"GREEN": "#2ca02c", "YELLOW": "#f1c40f", "ORANGE": "#ff7f0e", "RED": "#d62728", "PURPLE": "#7b3294"}


def basin_max_series(ds: xr.Dataset, variable: str) -> pd.Series:
    """Extract a basin-domain maximum time series from a gridded dataset."""
    da = ds[variable]
    if "member" in da.dims:
        da = da.max(dim=["lat", "lon"])
    else:
        da = da.max(dim=["lat", "lon"])
    return da.to_series()


def ensemble_probability_for_day(
    ens: xr.Dataset, target_date: pd.Timestamp, variable: str, threshold: float
) -> float:
    """Probability that ensemble members exceed a threshold on a given valid date."""
    valid = ens.sel(time=slice(target_date, target_date + pd.Timedelta(hours=23)))
    if valid.time.size == 0:
        return np.nan
    member_max = valid[variable].max(dim=["time", "lat", "lon"])
    return float((member_max > threshold).mean().item())


def latest_forecast_value_for_date(target_date: pd.Timestamp, variable: str) -> float:
    """Use the latest initialization at or before target date and return domain max."""
    eligible = [pd.Timestamp(x) for x in INIT_TIMES if pd.Timestamp(x) <= target_date]
    if not eligible:
        eligible = [pd.Timestamp(INIT_TIMES[0])]
    init = max(eligible)
    ds = deterministic_runs[init.strftime("%Y-%m-%d %H:%M")]
    valid = ds.sel(time=slice(target_date, target_date + pd.Timedelta(hours=23)))
    if valid.time.size == 0:
        return np.nan
    return float(valid[variable].max().item())


def decide_alert_level(
    rain24: float,
    rain72: float,
    antecedent: float,
    water_proxy: float,
    risk_index: float,
    prob100: float,
    prob200: float,
    prob300: float,
    drainage_factor: float = DRAINAGE_BOTTLENECK_FACTOR,
) -> tuple[str, str]:
    """Convert rainfall, ensemble probabilities, and local proxies into an alert level."""
    level = "GREEN"
    reasons = []
    adjusted_risk = min(1.0, risk_index * drainage_factor + 0.08 * (antecedent > 250) + 0.08 * (water_proxy > 0.70))
    if rain24 > THRESHOLDS["yellow_24h"] or prob100 >= 0.35:
        level = "YELLOW"
        reasons.append("heavy rain watch threshold exceeded")
    if rain24 > THRESHOLDS["orange_24h"] or rain72 > THRESHOLDS["orange_72h"] or prob100 >= 0.55:
        level = "ORANGE"
        reasons.append("preparedness threshold exceeded")
    if rain24 > THRESHOLDS["red_24h"] or rain72 > THRESHOLDS["red_72h"] or prob200 >= 0.45 or adjusted_risk > 0.72:
        level = "RED"
        reasons.append("severe flood warning threshold exceeded")
    if rain24 > THRESHOLDS["purple_24h"] or adjusted_risk > THRESHOLDS["purple_risk"] or prob300 >= 0.35:
        level = "PURPLE"
        reasons.append("extreme emergency threshold exceeded")
    if not reasons:
        reasons.append("normal monitoring")
    return level, "; ".join(reasons)


def suggested_action_for_level(level: str) -> str:
    """Return a recommended action for each alert level."""
    return {
        "GREEN": "Normal monitoring",
        "YELLOW": "Heavy rain watch; verify gauges and drainage readiness",
        "ORANGE": "Flood preparedness; pre-position pumps, shelters, and rescue assets",
        "RED": "Severe flood warning; evacuation advisory for low-lying zones",
        "PURPLE": "Extreme emergency; life-safety rescue priority and evacuation operations",
    }[level]


timeline_rows = []
for _, obs in observed_daily.iterrows():
    date = pd.Timestamp(obs["date"])
    f24 = latest_forecast_value_for_date(date, "precip_24h")
    f72 = latest_forecast_value_for_date(date, "precip_72h")
    prob100 = ensemble_probability_for_day(ensemble_ds, date, "precip_24h", 100.0)
    prob200 = ensemble_probability_for_day(ensemble_ds, date, "precip_24h", 200.0)
    prob300 = ensemble_probability_for_day(ensemble_ds, date, "precip_24h", 300.0)
    risk = max(float(obs["flood_severity_index"]), min(1.0, (f24 / 320.0) * 0.52 + (f72 / 700.0) * 0.32 + float(obs["water_level_proxy"]) * 0.16))
    level, reason = decide_alert_level(
        rain24=float(obs["observed_or_synthetic_rainfall_mm"]),
        rain72=float(obs["rainfall_72h_mm"]),
        antecedent=float(obs["antecedent_7d_mm"]),
        water_proxy=float(obs["water_level_proxy"]),
        risk_index=risk,
        prob100=prob100,
        prob200=prob200,
        prob300=prob300,
    )
    timeline_rows.append(
        {
            "date": date,
            "observed_or_synthetic_rainfall_mm": float(obs["observed_or_synthetic_rainfall_mm"]),
            "forecast_24h_rainfall_mm": f24,
            "forecast_72h_rainfall_mm": f72,
            "ensemble_probability_rain_gt_100mm_24h": prob100,
            "ensemble_probability_rain_gt_200mm_24h": prob200,
            "ensemble_probability_rain_gt_300mm_24h": prob300,
            "water_level_proxy": float(obs["water_level_proxy"]),
            "flood_risk_index": risk,
            "recommended_alert_level": level,
            "alert_reason": reason,
            "suggested_action": suggested_action_for_level(level),
            "data_source_note": "SYNTHETIC DEMO - replace with GPM/TMD/ThaiWater observations for production",
        }
    )

event_timeline = pd.DataFrame(timeline_rows)
alert_timeline = event_timeline[["date", "recommended_alert_level", "flood_risk_index", "suggested_action", "alert_reason"]]
event_timeline.to_csv(OUTPUT_DIR / "event_timeline.csv", index=False)
alert_timeline.to_csv(OUTPUT_DIR / "alert_timeline.csv", index=False)
event_timeline.loc[(event_timeline["date"] >= CRITICAL_START) & (event_timeline["date"] <= CRITICAL_END)]

# %% [markdown]
# ## Section 8 — Forecast Verification and Skill Metrics
#
# Demo metrics are computed against the synthetic observation series. In production,
# replace this with station rainfall, radar, satellite precipitation, and hydrological
# observations.
#
# Metrics:
# - MAE / RMSE for rainfall time series
# - Brier score for threshold exceedance
# - CRPS-like approximation for ensemble rainfall
# - lead-time analysis for alert escalation

# %%
def brier_score(prob: pd.Series, observed_event: pd.Series) -> float:
    """Compute Brier score for a probabilistic exceedance forecast."""
    valid = ~(prob.isna() | observed_event.isna())
    if valid.sum() == 0:
        return np.nan
    return float(np.mean((prob[valid] - observed_event[valid].astype(float)) ** 2))


def crps_like_ensemble_score(ensemble_values: np.ndarray, obs: float) -> float:
    """Approximate CRPS with mean absolute ensemble error minus ensemble spread correction."""
    ensemble_values = np.asarray(ensemble_values, dtype=float)
    if ensemble_values.size == 0:
        return np.nan
    mae_term = np.mean(np.abs(ensemble_values - obs))
    spread_term = 0.5 * np.mean(np.abs(ensemble_values[:, None] - ensemble_values[None, :]))
    return float(mae_term - spread_term)


valid_rows = event_timeline.dropna(subset=["forecast_24h_rainfall_mm"]).copy()
errors = valid_rows["forecast_24h_rainfall_mm"] - valid_rows["observed_or_synthetic_rainfall_mm"]
rainfall_mae = float(np.mean(np.abs(errors)))
rainfall_rmse = float(np.sqrt(np.mean(errors**2)))
brier_100 = brier_score(
    valid_rows["ensemble_probability_rain_gt_100mm_24h"],
    valid_rows["observed_or_synthetic_rainfall_mm"] > 100,
)
brier_200 = brier_score(
    valid_rows["ensemble_probability_rain_gt_200mm_24h"],
    valid_rows["observed_or_synthetic_rainfall_mm"] > 200,
)

verification_rows = []
peak_date = event_timeline.loc[event_timeline["observed_or_synthetic_rainfall_mm"].idxmax(), "date"]
for init in INIT_TIMES:
    init_ts = pd.Timestamp(init)
    ens = ensemble_ds.sel(init_time=init_ts)
    peak_window = ens.sel(time=slice(peak_date, peak_date + pd.Timedelta(hours=23)))
    max24_members = peak_window["precip_24h"].max(dim=["time", "lat", "lon"]).values
    obs_peak = float(event_timeline.loc[event_timeline["date"] == peak_date, "observed_or_synthetic_rainfall_mm"].iloc[0])
    probability_extreme = float((max24_members > 300).mean())
    lead_hours = float((peak_date - init_ts).total_seconds() / 3600)
    max_f24 = float(max24_members.max())
    max_f72 = float(peak_window["precip_72h"].max().item())
    first_alert = event_timeline.loc[event_timeline["recommended_alert_level"].isin(["RED", "PURPLE"]), "date"].min()
    alert_level = "PURPLE" if probability_extreme >= 0.35 or max_f24 > 300 else "RED" if max_f24 > 200 else "ORANGE"
    confidence = "High" if probability_extreme >= 0.50 else "Medium" if probability_extreme >= 0.25 else "Low"
    verification_rows.append(
        {
            "init_time": init_ts,
            "lead_time_to_peak_hours": lead_hours,
            "max_forecast_24h_rainfall": max_f24,
            "max_forecast_72h_rainfall": max_f72,
            "probability_extreme": probability_extreme,
            "first_alert_time": first_alert,
            "alert_level": alert_level,
            "confidence": confidence,
            "crps_like_peak_24h": crps_like_ensemble_score(max24_members, obs_peak),
            "explanation": "Synthetic ensemble indicates escalating extreme-rainfall risk near Hat Yai and upstream catchments.",
        }
    )

forecast_verification = pd.DataFrame(verification_rows)
forecast_verification.attrs["rainfall_mae"] = rainfall_mae
forecast_verification.attrs["rainfall_rmse"] = rainfall_rmse
forecast_verification.attrs["brier_100mm"] = brier_100
forecast_verification.attrs["brier_200mm"] = brier_200
forecast_verification.to_csv(OUTPUT_DIR / "forecast_verification.csv", index=False)

print(f"Rainfall MAE: {rainfall_mae:.1f} mm")
print(f"Rainfall RMSE: {rainfall_rmse:.1f} mm")
print(f"Brier score >100 mm: {brier_100:.3f}")
print(f"Brier score >200 mm: {brier_200:.3f}")
forecast_verification

# %% [markdown]
# ### Lead-time interpretation / คำอธิบายภาษาไทย
#
# ตารางนี้ตอบคำถามว่า “ระบบควรเห็นสัญญาณยกระดับเตือนภัยก่อนวันพีคได้กี่ชั่วโมง”
# ใน demo mode ค่าทั้งหมดมาจาก synthetic ensemble เพื่อสาธิต workflow ไม่ใช่ผล hindcast จริง
# ของเหตุการณ์หาดใหญ่

# %% [markdown]
# ## Section 9 — Visualization
#
# Required plots:
#
# 1. Daily rainfall timeline
# 2. 24h and 72h accumulated rainfall
# 3. Ensemble forecast plume
# 4. Probability of exceeding thresholds
# 5. Water level / flood proxy
# 6. Flood risk index timeline
# 7. Alert level timeline
# 8. Map of rainfall field around Hat Yai
# 9. Map of flood risk hotspot around Hat Yai
# 10. Optional animated map if possible

# %%
def shade_critical_period(ax: Any) -> None:
    """Add vertical shading for main flood escalation window."""
    ax.axvspan(pd.Timestamp(MAIN_ESCALATION_START), pd.Timestamp(MAIN_ESCALATION_END), color="#d62728", alpha=0.12, label="Main escalation window")


def save_current_figure(name: str) -> Path:
    """Save current Matplotlib figure under the figure output directory."""
    path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(path)
    return path


def plot_daily_rainfall() -> None:
    """Plot daily rainfall timeline."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(event_timeline["date"], event_timeline["observed_or_synthetic_rainfall_mm"], color="#3b82f6", label="Daily rainfall (synthetic/demo)")
    shade_critical_period(ax)
    ax.axhline(100, color="#f1c40f", linestyle="--", label="100 mm watch")
    ax.axhline(200, color="#d62728", linestyle="--", label="200 mm severe")
    ax.axhline(300, color="#7b3294", linestyle="--", label="300 mm extreme")
    ax.set_title("สัญญาณฝนสะสมก่อนน้ำท่วม / Daily Rainfall Timeline - Synthetic Demo")
    ax.set_ylabel("Rainfall (mm/day)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    save_current_figure("01_daily_rainfall_timeline.png")


def plot_accumulations() -> None:
    """Plot 24h and 72h accumulated rainfall timeline."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(event_timeline["date"], event_timeline["forecast_24h_rainfall_mm"], marker="o", label="Forecast 24h rainfall")
    ax.plot(event_timeline["date"], event_timeline["forecast_72h_rainfall_mm"], marker="o", label="Forecast 72h rainfall")
    ax.plot(observed_daily["date"], observed_daily["rainfall_72h_mm"], color="black", linewidth=1.5, label="Synthetic observed 72h rainfall")
    shade_critical_period(ax)
    ax.axhline(250, color="#ff7f0e", linestyle="--", label="72h orange threshold")
    ax.axhline(350, color="#d62728", linestyle="--", label="72h red threshold")
    ax.set_title("24h and 72h Accumulated Rainfall / ฝนสะสม 24 และ 72 ชั่วโมง")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    save_current_figure("02_accumulated_rainfall.png")


def plot_ensemble_plume(init_time: str = primary_init) -> None:
    """Plot ensemble plume for basin maximum 24h rainfall."""
    init_ts = pd.Timestamp(init_time)
    ens = ensemble_ds.sel(init_time=init_ts)
    plume = ens["precip_24h"].max(dim=["lat", "lon"]).to_pandas()
    fig, ax = plt.subplots(figsize=(12, 5))
    for member in plume.index:
        ax.plot(plume.columns, plume.loc[member], color="#60a5fa", alpha=0.35, linewidth=1)
    ax.plot(plume.columns, plume.mean(axis=0), color="#1d4ed8", linewidth=2.5, label="Ensemble mean")
    shade_critical_period(ax)
    ax.axhline(100, color="#f1c40f", linestyle="--")
    ax.axhline(200, color="#d62728", linestyle="--")
    ax.axhline(300, color="#7b3294", linestyle="--")
    ax.set_title("Ensemble Forecast Plume / ความไม่แน่นอนของฝนคาดการณ์")
    ax.set_ylabel("Domain-max 24h rainfall (mm)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    save_current_figure("03_ensemble_forecast_plume.png")


def plot_exceedance_probabilities() -> None:
    """Plot exceedance probabilities for rainfall thresholds."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(event_timeline["date"], event_timeline["ensemble_probability_rain_gt_100mm_24h"], marker="o", label="P(24h rain > 100 mm)")
    ax.plot(event_timeline["date"], event_timeline["ensemble_probability_rain_gt_200mm_24h"], marker="o", label="P(24h rain > 200 mm)")
    ax.plot(event_timeline["date"], event_timeline["ensemble_probability_rain_gt_300mm_24h"], marker="o", label="P(24h rain > 300 mm)")
    shade_critical_period(ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("ความน่าจะเป็นของฝนเกินเกณฑ์วิกฤต / Probability of Exceeding Critical Rainfall")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    save_current_figure("04_exceedance_probabilities.png")


def plot_water_and_risk() -> None:
    """Plot water-level proxy and flood-risk index."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(event_timeline["date"], event_timeline["water_level_proxy"], marker="o", label="Water level proxy")
    ax.plot(event_timeline["date"], event_timeline["flood_risk_index"], marker="o", label="Flood risk index")
    shade_critical_period(ax)
    ax.axhline(0.70, color="#d62728", linestyle="--", label="High risk reference")
    ax.axhline(0.85, color="#7b3294", linestyle="--", label="Extreme risk reference")
    ax.set_ylim(0, 1.1)
    ax.set_title("Water Level and Flood Proxy / ดัชนีน้ำและความเสี่ยงน้ำท่วม")
    ax.set_ylabel("Index")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    save_current_figure("05_water_level_and_flood_risk.png")


def plot_water_level_proxy() -> None:
    """Plot water-level proxy as a standalone executive chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(event_timeline["date"], event_timeline["water_level_proxy"], marker="o", color="#0891b2", linewidth=2.2)
    shade_critical_period(ax)
    ax.axhline(0.70, color="#d62728", linestyle="--", label="High water-level proxy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Water Level / River Response Proxy / ดัชนีน้ำในคลองและการตอบสนองของลุ่มน้ำ")
    ax.set_ylabel("Proxy index")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    save_current_figure("05b_water_level_proxy.png")


def plot_flood_risk_index() -> None:
    """Plot flood-risk index as a standalone executive chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(event_timeline["date"], event_timeline["flood_risk_index"], marker="o", color="#7b3294", linewidth=2.2)
    shade_critical_period(ax)
    ax.axhline(0.70, color="#d62728", linestyle="--", label="High risk")
    ax.axhline(0.85, color="#7b3294", linestyle="--", label="Extreme risk")
    ax.set_ylim(0, 1.05)
    ax.set_title("Flood Risk Index Timeline / ดัชนีความเสี่ยงน้ำท่วม")
    ax.set_ylabel("Risk index")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    save_current_figure("05c_flood_risk_index.png")


def plot_alert_timeline() -> None:
    """Plot alert level timeline."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    y = event_timeline["recommended_alert_level"].map(ALERT_SCORE)
    colors = event_timeline["recommended_alert_level"].map(ALERT_COLOR)
    ax.scatter(event_timeline["date"], y, s=95, c=colors)
    ax.plot(event_timeline["date"], y, color="#374151", alpha=0.5)
    shade_critical_period(ax)
    ax.set_yticks(range(len(ALERT_ORDER)), ALERT_ORDER)
    ax.set_title("จุดที่ระบบควรยกระดับการแจ้งเตือน / Alert Escalation Timeline")
    ax.set_ylabel("Alert level")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    save_current_figure("06_alert_level_timeline.png")


def plot_map_field(ds: xr.Dataset, variable: str, title: str, file_name: str, cmap: str = "Blues") -> None:
    """Plot a regional map using Cartopy when available, otherwise lon/lat pcolormesh."""
    time_index = int(np.argmax(ds[variable].max(dim=["lat", "lon"]).values))
    field = ds[variable].isel(time=time_index)
    fig = plt.figure(figsize=(9, 7))
    if HAS_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        mesh = ax.pcolormesh(ds.lon, ds.lat, field, cmap=cmap, shading="auto", transform=ccrs.PlateCarree())
        ax.coastlines(resolution="10m", linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    else:
        ax = plt.axes()
        mesh = ax.pcolormesh(ds.lon, ds.lat, field, cmap=cmap, shading="auto")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.scatter([CENTER_LON], [CENTER_LAT], c="red", s=80, marker="*", label="Hat Yai")
    ax.set_title(f"{title}\nValid: {pd.Timestamp(field.time.values):%Y-%m-%d %H:%M UTC} | Synthetic/demo")
    ax.legend(loc="upper right")
    plt.colorbar(mesh, ax=ax, shrink=0.75, label=variable)
    save_current_figure(file_name)


def create_optional_animation_placeholder() -> None:
    """Create a lightweight text placeholder for animation output."""
    (FIGURE_DIR / "10_optional_animation_note.txt").write_text(
        "Optional animated map can be generated with matplotlib.animation when needed. "
        "Static PNG maps are produced by default for robust executive demo playback.\n",
        encoding="utf-8",
    )


if HAS_MATPLOTLIB:
    plot_daily_rainfall()
    plot_accumulations()
    plot_ensemble_plume()
    plot_exceedance_probabilities()
    plot_water_and_risk()
    plot_water_level_proxy()
    plot_flood_risk_index()
    plot_alert_timeline()
    plot_map_field(deterministic_runs[primary_init], "precip_24h", "Map of Rainfall Field Around Hat Yai", "07_rainfall_field_map.png", "Blues")
    plot_map_field(downscaled_ds, "local_flood_risk_index", "Map of Flood Risk Hotspot Around Hat Yai", "08_flood_risk_hotspot_map.png", "magma")
    create_optional_animation_placeholder()
    plt.close("all")

print(f"Saved figures to: {FIGURE_DIR}")

# %% [markdown]
# ## Section 10 — Early Warning Decision Engine
#
# The warning engine turns rows from the event timeline into human-readable messages.
# Thai templates include:
#
# - เฝ้าระวังฝนตกหนัก
# - เตรียมพร้อมรับมือน้ำท่วมฉับพลัน
# - แจ้งเตือนอพยพพื้นที่ลุ่มต่ำ
# - ภาวะฉุกเฉินระดับรุนแรงมาก

# %%
def generate_warning_message(row: pd.Series, language: str = "th") -> str:
    """Generate a warning message from one alert timeline row.

    Parameters
    ----------
    row:
        A row from event_timeline.
    language:
        "th" for Thai, "en" for English.
    """
    issue_time = pd.Timestamp(row["date"]).strftime("%Y-%m-%d 08:00 ICT")
    level = row["recommended_alert_level"]
    rain24 = float(row["forecast_24h_rainfall_mm"])
    rain72 = float(row["forecast_72h_rainfall_mm"])
    confidence = "สูง" if level in {"RED", "PURPLE"} else "ปานกลาง" if level in {"ORANGE", "YELLOW"} else "ต่ำ"
    affected_area = "อ.หาดใหญ่ จ.สงขลา พื้นที่ลุ่มต่ำริมคลองอู่ตะเภา คลอง ร.1 และทางระบายน้ำสู่ทะเลสาบสงขลา"
    uncertainty = "ข้อมูลนี้เป็นเดโม/วิจัย ต้องตรวจยืนยันกับ TMD, ThaiWater, สถานีฝน, ระดับน้ำ และหน่วยงานที่มีอำนาจก่อนใช้จริง"
    source_note = str(row.get("data_source_note", "Earth2Studio-style synthetic demo forecast"))

    thai_headline = {
        "GREEN": "ติดตามสถานการณ์ตามปกติ",
        "YELLOW": "เฝ้าระวังฝนตกหนัก",
        "ORANGE": "เตรียมพร้อมรับมือน้ำท่วมฉับพลัน",
        "RED": "แจ้งเตือนอพยพพื้นที่ลุ่มต่ำ",
        "PURPLE": "ภาวะฉุกเฉินระดับรุนแรงมาก",
    }[level]
    thai_action = {
        "GREEN": "ติดตามประกาศและตรวจสอบระบบระบายน้ำ",
        "YELLOW": "ตรวจเครื่องสูบน้ำ จุดอพยพ และช่องทางสื่อสารประชาชน",
        "ORANGE": "เตรียมศูนย์พักพิง เครื่องสูบน้ำ เรือ และทีมกู้ภัยในพื้นที่เสี่ยง",
        "RED": "พิจารณาอพยพพื้นที่ลุ่มต่ำ โรงพยาบาล โรงเรียน และเส้นทางคมนาคมเสี่ยง",
        "PURPLE": "จัดลำดับภารกิจช่วยชีวิต อพยพเร่งด่วน และเปิดศูนย์บัญชาการเหตุฉุกเฉิน",
    }[level]

    if language.lower().startswith("th"):
        return (
            f"[{level}] {thai_headline}\n"
            f"เวลาออกประกาศ: {issue_time}\n"
            f"พื้นที่เสี่ยง: {affected_area}\n"
            f"ช่วงคาดการณ์: 24-72 ชั่วโมงข้างหน้า\n"
            f"คาดการณ์ฝนสูงสุด: 24 ชม. {rain24:.0f} มม.; 72 ชม. {rain72:.0f} มม.\n"
            f"ความเชื่อมั่น: {confidence}\n"
            f"คำแนะนำ: {thai_action}\n"
            f"หมายเหตุความไม่แน่นอน: {uncertainty}\n"
            f"แหล่งข้อมูล: {source_note}"
        )

    return (
        f"[{level}] Hat Yai flood-risk warning\n"
        f"Issue time: {issue_time}\n"
        f"Forecast window: next 24-72 hours\n"
        f"Expected peak rainfall: 24h {rain24:.0f} mm; 72h {rain72:.0f} mm\n"
        f"Affected area: Hat Yai lowlands, Khlong U-Taphao, Khlong R.1, Songkhla Lake drainage path\n"
        f"Recommended action: {row['suggested_action']}\n"
        f"Uncertainty note: research/demo only; verify with authorized agencies.\n"
        f"Data source note: {source_note}"
    )


warning_dates = pd.to_datetime(["2025-11-19", "2025-11-21", "2025-11-23", "2025-11-24", "2025-11-25"])
warnings_out = {}
for date in warning_dates:
    row = event_timeline.loc[event_timeline["date"] == date].iloc[0]
    warnings_out[str(date.date())] = generate_warning_message(row, language="th")
    print("\n" + "=" * 90)
    print(warnings_out[str(date.date())])

(OUTPUT_DIR / "example_warning_messages_th.json").write_text(json.dumps(warnings_out, ensure_ascii=False, indent=2), encoding="utf-8")

# %% [markdown]
# ## Section 11 — “What Should Have Been Monitored?”
#
# ตารางนี้เป็น checklist สำหรับ command center และ dashboard design.

# %%
monitoring_table = pd.DataFrame(
    [
        ("Meteorological", "Monsoon surge", "Track low-level wind and moisture transport into southern Thailand"),
        ("Meteorological", "Low pressure / storm influence", "Monitor pressure falls and cyclonic influence over Gulf/Andaman region"),
        ("Meteorological", "Moisture convergence", "Use 850 hPa moisture flux and integrated water vapor"),
        ("Meteorological", "24h/72h rainfall", "Escalate when accumulations cross watch/preparedness/severe thresholds"),
        ("Meteorological", "Ensemble probability", "Use probability of >100, >200, >300 mm/day rainfall"),
        ("Hydrological", "Upstream rainfall accumulation", "Track Khlong U-Taphao upstream catchments"),
        ("Hydrological", "River/canal level", "Use canal sensors, bridge gauges, and telemetry"),
        ("Hydrological", "Drainage capacity", "Represent Khlong R.1 and urban drainage bottlenecks"),
        ("Hydrological", "Lake/tidal backwater effect", "Monitor Songkhla Lake drainage path and downstream constraints"),
        ("Urban exposure", "Low-lying communities", "Prioritize neighborhoods with repeated inundation"),
        ("Urban exposure", "Hospitals and schools", "Protect critical services and evacuation-dependent groups"),
        ("Urban exposure", "Transport routes", "Track rail, airport, highway, and arterial road closures"),
        ("Urban exposure", "Power and telecom infrastructure", "Protect substations, cell towers, and backup power"),
        ("Response readiness", "Shelter readiness", "Open shelters before severe inundation"),
        ("Response readiness", "Pump readiness", "Pre-position pumps and fuel"),
        ("Response readiness", "Rescue boats", "Stage assets near high-risk lowlands"),
        ("Response readiness", "Evacuation routes", "Update road passability and alternative paths"),
        ("Response readiness", "Communication readiness", "Prepare multilingual, accessible warning messages"),
        ("Public warning", "Warning lead time", "Measure time from first alert to peak flood risk"),
        ("Public warning", "Message clarity", "Use clear action verbs and affected zones"),
        ("Public warning", "Escalation trigger", "Log which threshold triggered each alert"),
        ("Public warning", "Channels", "Cell Broadcast / LINE / SMS / siren / social media / command center"),
    ],
    columns=["dimension", "indicator", "operational_use"],
)
monitoring_table.to_csv(OUTPUT_DIR / "monitoring_dimensions.csv", index=False)
monitoring_table

# %% [markdown]
# ## Section 12 — Operational Architecture for SIAM.AI Demo
#
# **ต้นแบบ Sovereign AI Flood Watchdog**
#
# Production architecture:
#
# - **Data ingestion**: GFS / ERA5 / satellite rainfall / radar / rain gauges / water-level sensors / IoT / CCTV / social reports
# - **AI weather**: Earth2Studio deterministic + ensemble forecast
# - **Downscaling**: CorrDiff / regional model / radar nowcasting
# - **Hydrology**: rainfall-runoff model / canal routing / water-level prediction for Khlong U-Taphao, Khlong R.1, and Songkhla Lake drainage
# - **Risk model**: flood index + exposure map + critical asset map
# - **Alert decision**: rule-based thresholds + probabilistic confidence + human approval
# - **Dashboard**: maps, timelines, probability, alert message generator
# - **Notification**: Cell Broadcast, LINE OA, SMS, siren, government command center
# - **Audit**: every forecast, threshold, warning, and decision must be logged
#
# ภาษาไทย: ระบบจริงต้องมี human-in-the-loop และการรับรองจากหน่วยงานอุตุนิยมวิทยา อุทกวิทยา
# และป้องกันบรรเทาสาธารณภัยก่อนใช้เป็นระบบเตือนภัยประชาชน

# %% [markdown]
# ## Section 13 — Executive Demo Output
#
# Copy-paste-ready Thai summary for board presentation.

# %%
executive_summary_th = f"""
# Earth2Studio Disaster Replay Demo: Hat Yai Flood — November 2025

## เกิดอะไรขึ้น?
เดือนพฤศจิกายน 2568 หาดใหญ่เผชิญรูปแบบฝนสะสมรุนแรง โดยในเดโมนี้จำลองให้มีฝนต่อเนื่องตั้งแต่วันที่ 17 พฤศจิกายน
และมีช่วงยกระดับหลักระหว่างวันที่ 22-26 พฤศจิกายน ฝนรายวันพีคประมาณ {observed_daily['observed_or_synthetic_rainfall_mm'].max():.0f} มม.
ทำให้ดัชนีความเสี่ยงน้ำท่วมเพิ่มขึ้นอย่างรวดเร็วตามน้ำในคลองและพื้นที่ลุ่มต่ำในเมือง

## สัญญาณใดเกิดก่อนน้ำท่วม?
สัญญาณสำคัญคือฝนสะสม 24/72 ชั่วโมงที่สูงขึ้นต่อเนื่อง, ensemble probability ของฝนเกิน 100/200/300 มม.,
water-level proxy ที่ lag หลังฝน 6-24 ชั่วโมง, และความเปราะบางของระบบระบายน้ำเมืองบริเวณคลองอู่ตะเภา คลอง ร.1
และเส้นทางระบายสู่ทะเลสาบสงขลา

## Earth2Studio สามารถช่วย forecast อะไร?
Earth2Studio สามารถเป็นแกนของ deterministic + ensemble AI weather forecast เพื่อดูฝนหนักล่วงหน้า
ประเมินความไม่แน่นอน และส่งต่อให้ local downscaling / hydrological routing / risk model
เพื่อแปลงสัญญาณอากาศเป็นระดับเตือนภัยในพื้นที่จริง

## ต้องใช้ข้อมูลท้องถิ่นอะไรเพิ่ม?
ต้องแทน synthetic rainfall ด้วย GPM/TMD/ThaiWater/radar/rain gauge, เพิ่ม DEM และ drainage network,
ระดับน้ำคลองแบบ real-time, สถานะประตูน้ำ/เครื่องสูบน้ำ, แผนที่ชุมชนลุ่มต่ำ โรงพยาบาล โรงเรียน ถนน และโครงสร้างพื้นฐานสำคัญ

## ควรยกระดับแจ้งเตือนเมื่อใด?
ใน demo timeline ระบบเริ่มเห็นสัญญาณ YELLOW/ORANGE ก่อนช่วงพีคหลายวัน และควรยกระดับเป็น RED/PURPLE
เมื่อฝน 24 ชั่วโมงเกิน 200-300 มม., ฝน 72 ชั่วโมงเกิน 350 มม., ensemble probability สูงขึ้น,
water-level proxy สูง และ flood-risk index เกิน 0.85

## จะต่อยอดเป็น Sovereign AI Flood Watchdog ของไทยได้อย่างไร?
นำ Earth2Studio บนโครงสร้างพื้นฐาน GPU ของ SIAM.AI เชื่อมข้อมูลอากาศ-น้ำ-เมืองแบบ real-time
ทำ ensemble forecasting, downscaling, hydrological routing, risk scoring, alert decision,
dashboard และระบบแจ้งเตือนหลายช่องทาง โดยบันทึก audit trail ทุก forecast, threshold, warning และการตัดสินใจ
เพื่อให้รัฐมีระบบ AI ที่ควบคุมได้เอง โปร่งใส และเหมาะกับบริบทประเทศไทย

**หมายเหตุ:** ผลนี้เป็น research/demo เท่านั้น ไม่ใช่ประกาศเตือนภัยอย่างเป็นทางการ
"""

print(executive_summary_th)
(OUTPUT_DIR / "executive_summary_th.md").write_text(executive_summary_th, encoding="utf-8")

# %% [markdown]
# ## Section 14 — Next Steps
#
# 1. Replace synthetic rainfall with actual GPM/TMD/ThaiWater data.
# 2. Connect Earth2Studio to real GFS/ERA5 forecast initialization.
# 3. Add hydrological routing for Khlong U-Taphao and Khlong R.1.
# 4. Add DEM/topography and drainage network.
# 5. Add historical flood maps and impact reports.
# 6. Validate thresholds using past Hat Yai floods: 2000, 2010, 2025.
# 7. Build real-time dashboard.
# 8. Add human-in-the-loop warning approval workflow.
# 9. Deploy on SIAM.AI GPU infrastructure.
# 10. Convert Notebook into API + dashboard service.
#
# Final reminder: this notebook is for research/demo only. Do not use it as an official
# life-safety warning system unless validated and authorized by meteorological,
# hydrological, and disaster-management agencies.

# %%
artifact_summary = {
    "output_dir": str(OUTPUT_DIR),
    "event_timeline": str(OUTPUT_DIR / "event_timeline.csv"),
    "alert_timeline": str(OUTPUT_DIR / "alert_timeline.csv"),
    "forecast_verification": str(OUTPUT_DIR / "forecast_verification.csv"),
    "figures": str(FIGURE_DIR),
    "deterministic_outputs": str(DET_DIR),
    "ensemble_outputs": str(ENS_DIR),
    "demo_mode": DEMO_MODE,
    "safety_note": "Research/demo only; not an official life-safety warning system.",
}
(OUTPUT_DIR / "artifact_summary.json").write_text(json.dumps(artifact_summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(artifact_summary, indent=2, ensure_ascii=False))
print("Notebook completed successfully.")
