# %% [markdown]
# # Earth2Studio Disaster Replay Demo: Hat Yai Flood — November 2025
#
# **Real Earth2Studio-style inference notebook**
#
# ภาษาไทย: โน้ตบุ๊กนี้สร้างใหม่จากศูนย์เพื่อให้ใกล้กับ official Earth2Studio examples:
# โหลด data source จริง, โหลด prognostic model จริง, รัน deterministic inference จริง,
# เขียน output ผ่าน `ZarrBackend`, อ่าน output กลับมา inspect, แล้ว plot map/time series
# สำหรับกรณีน้ำท่วมหาดใหญ่เดือนพฤศจิกายน 2568 (November 2025)
#
# **Safety note:** Research/demo only. This is not an official life-safety flood warning.

# %% [markdown]
# ## 0. Executive Story / เรื่องเล่าสำหรับผู้บริหาร
#
# - พ.ย. 2568 = November 2025
# - พื้นที่ศึกษา: หาดใหญ่ จ.สงขลา และลุ่มน้ำรอบเมือง
# - จุดสนใจ: ฝนหนักสะสม, moisture surge, pressure/wind signals, และ risk proxy
# - วัตถุประสงค์: สาธิตว่า Earth2Studio สามารถเป็นแกนของ AI weather inference ได้จริง
#
# **Important limitation:** FCN3 / FourCastNet3 outputs atmospheric variables such as
# `t2m`, `msl`, `tcwv`, winds, humidity, and geopotential. It does **not** directly output
# precipitation in this environment. Therefore this notebook computes a clearly labeled
# **rainfall-risk proxy** from real model output variables. It is not observed rainfall
# and not forecast precipitation.

# %% [markdown]
# ## 1. Imports and Environment Check
#
# This follows the same pattern as the official examples: import Earth2Studio modules,
# instantiate model/data/IO, run inference, inspect Zarr output, then plot.

# %%
from __future__ import annotations

import json
import os
import sys
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

try:
    from IPython.display import display
except Exception:
    display = print

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception as exc:
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
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")
except Exception:
    logger = None

try:
    from earth2studio.data import ARCO, GFS, NCAR_ERA5
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px import DLWP, FCN, FCN3, Persistence, SFNO
    from earth2studio.perturbation import SphericalGaussian
    from earth2studio.run import deterministic, ensemble

    HAS_EARTH2STUDIO = True
except Exception as exc:
    HAS_EARTH2STUDIO = False
    EARTH2STUDIO_IMPORT_ERROR = repr(exc)
    print("Earth2Studio import failed:", EARTH2STUDIO_IMPORT_ERROR)

if HAS_MATPLOTLIB:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.figsize": (11, 5), "savefig.dpi": 160})

print("Environment summary")
print("-------------------")
print("Python:", sys.version.split()[0])
print("Earth2Studio available:", HAS_EARTH2STUDIO)
print("Torch available:", HAS_TORCH)
if HAS_TORCH:
    print("Torch:", torch.__version__)
    print("CUDA available:", HAS_GPU)
    if HAS_GPU:
        print("GPU:", torch.cuda.get_device_name(0))
print("Cartopy available:", HAS_CARTOPY)

# %% [markdown]
# ## 2. Case Configuration
#
# Thai executive context:
#
# โน้ตบุ๊กนี้วางให้เป็น disaster replay สำหรับหาดใหญ่ โดยใช้วันที่ initialization ตามโจทย์
# และเลือก output domain เฉพาะภาคใต้ตอนล่างเพื่อให้ inference output เล็กและ plot ได้เร็ว
# ขณะที่ model ยังทำ inference จาก global GFS initial condition ตาม workflow จริงของ Earth2Studio

# %%
CASE_NAME = "hatyai_flood_nov2025_real"
CENTER_LAT = 7.008
CENTER_LON = 100.474
BBOX = {"lat_min": 5.5, "lat_max": 8.5, "lon_min": 99.0, "lon_max": 102.0}

HIST_START = pd.Timestamp("2025-11-01 00:00")
HIST_END = pd.Timestamp("2025-11-30 23:59")
CRITICAL_START = pd.Timestamp("2025-11-17 00:00")
CRITICAL_END = pd.Timestamp("2025-11-28 23:59")
MAIN_ESCALATION_START = pd.Timestamp("2025-11-22 00:00")
MAIN_ESCALATION_END = pd.Timestamp("2025-11-26 23:59")

INIT_TIMES = pd.to_datetime(
    [
        "2025-11-17 00:00",
        "2025-11-19 00:00",
        "2025-11-21 00:00",
        "2025-11-23 00:00",
        "2025-11-24 00:00",
    ]
)

# Default real run: 72 hours because 24h and 72h risk-proxy windows are required.
# For a full 10-day replay, set HATYAI_REAL_NSTEPS=40 before running the notebook.
N_STEPS = int(os.environ.get("HATYAI_REAL_NSTEPS", "12"))
MAX_REAL_INITS = int(os.environ.get("HATYAI_MAX_REAL_INITS", str(len(INIT_TIMES))))
RUN_INIT_TIMES = INIT_TIMES[:MAX_REAL_INITS]

RUN_REAL_ENSEMBLE = os.environ.get("HATYAI_RUN_REAL_ENSEMBLE", "1").lower() in {"1", "true", "yes"}
ENSEMBLE_MEMBERS = int(os.environ.get("HATYAI_ENSEMBLE_MEMBERS", "2"))
ENSEMBLE_NSTEPS = int(os.environ.get("HATYAI_ENSEMBLE_NSTEPS", str(min(4, N_STEPS))))

OUTPUT_DIR = Path("outputs") / "hatyai_flood_nov2025_real"
FORECAST_DIR = OUTPUT_DIR / "forecast"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
for path in [FORECAST_DIR, FIGURE_DIR, TABLE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

OUTPUT_VARIABLES = np.array(["t2m", "msl", "tcwv", "u10m", "v10m", "u850", "v850", "q850", "z500"])
ENSEMBLE_OUTPUT_VARIABLES = np.array(["t2m", "msl", "tcwv", "u10m", "v10m", "u850", "v850", "q850"])
OUTPUT_COORDS = OrderedDict(
    {
        "variable": OUTPUT_VARIABLES,
        "lat": np.arange(BBOX["lat_max"], BBOX["lat_min"] - 0.001, -0.25),
        "lon": np.arange(BBOX["lon_min"], BBOX["lon_max"] + 0.001, 0.25),
    }
)
ENSEMBLE_OUTPUT_COORDS = OrderedDict(
    {
        "variable": ENSEMBLE_OUTPUT_VARIABLES,
        "lat": OUTPUT_COORDS["lat"],
        "lon": OUTPUT_COORDS["lon"],
    }
)

print("Case:", CASE_NAME)
print("Initialization dates:", [str(t) for t in RUN_INIT_TIMES])
print("N_STEPS:", N_STEPS, "=> hours:", N_STEPS * 6)
print("Output:", OUTPUT_DIR)

# %% [markdown]
# ## 3. Earth2Studio Data Source Loading
#
# Official examples often use:
#
# ```python
# from earth2studio.data import GFS
# data = GFS()
# ```
#
# We try **GFS first** because the installed Earth2Studio data source says it spans
# Jan 2021 to present and works for 2025-11-17. If GFS fails, we try ARCO and NCAR ERA5.
#
# ภาษาไทย: cell นี้โหลด dataset จริง ไม่ใช่ synthetic data โดยตรวจตัวแปรพื้นฐานก่อนรัน model

# %%
data_source_errors: list[str] = []
data = None
DATA_SOURCE_NAME = None
inspection_da = None
inspection_variables = ["t2m", "msl", "u10m", "v10m", "tcwv"]

if HAS_EARTH2STUDIO:
    for source_name, source_factory in [
        ("GFS", lambda: GFS(cache=True, verbose=False, async_timeout=600)),
        ("ARCO", lambda: ARCO(cache=True, verbose=False, async_timeout=600)),
        ("NCAR_ERA5", lambda: NCAR_ERA5(cache=True, verbose=False, async_timeout=600)),
    ]:
        try:
            candidate = source_factory()
            # Direct datasource call returns an xarray.DataArray.
            inspection_da = candidate([RUN_INIT_TIMES[0].to_pydatetime()], inspection_variables)
            data = candidate
            DATA_SOURCE_NAME = source_name
            print(f"Selected data source: {DATA_SOURCE_NAME}")
            break
        except Exception as exc:
            reason = f"{source_name} failed: {type(exc).__name__}: {exc}"
            data_source_errors.append(reason)
            print(reason)

if data is None:
    print("No real Earth2Studio data source was available.")
    print("\n".join(data_source_errors))

# %% [markdown]
# ## 4. Dataset Inspection
#
# Required inspection:
#
# - dimensions
# - coordinates
# - available variables
# - time range
# - spatial resolution
# - verify Hat Yai bbox is inside the data domain
# - select nearest grid point to Hat Yai

# %%
def lon_to_360(lon: float | np.ndarray) -> float | np.ndarray:
    """Convert longitude to 0-360 convention."""
    return np.mod(lon, 360.0)


def summarize_dataarray_domain(da: xr.DataArray) -> dict[str, Any]:
    """Summarize a real Earth2Studio data source sample."""
    lat = da["lat"].values
    lon = da["lon"].values
    lat_res = float(np.nanmedian(np.abs(np.diff(lat))))
    lon_res = float(np.nanmedian(np.abs(np.diff(lon))))
    bbox_inside = (
        BBOX["lat_min"] >= float(np.nanmin(lat))
        and BBOX["lat_max"] <= float(np.nanmax(lat))
        and lon_to_360(BBOX["lon_min"]) >= float(np.nanmin(lon_to_360(lon)))
        and lon_to_360(BBOX["lon_max"]) <= float(np.nanmax(lon_to_360(lon)))
    )
    nearest = da.sel(lat=CENTER_LAT, lon=lon_to_360(CENTER_LON), method="nearest")
    return {
        "dims": dict(da.sizes),
        "coords": list(da.coords),
        "variables": list(da["variable"].values) if "variable" in da.coords else [],
        "time_range": (str(pd.to_datetime(da["time"].values).min()), str(pd.to_datetime(da["time"].values).max())),
        "lat_range": (float(np.nanmin(lat)), float(np.nanmax(lat))),
        "lon_range": (float(np.nanmin(lon)), float(np.nanmax(lon))),
        "spatial_resolution_deg": (lat_res, lon_res),
        "hat_yai_bbox_inside": bool(bbox_inside),
        "nearest_grid_point": {
            "lat": float(nearest["lat"].values),
            "lon": float(nearest["lon"].values),
        },
    }


if inspection_da is not None:
    dataset_summary = summarize_dataarray_domain(inspection_da)
    print(json.dumps(dataset_summary, indent=2))
    inspection_ds = inspection_da.to_dataset(dim="variable")
    sample_path = FORECAST_DIR / "real_data_source_inspection_sample.zarr"
    inspection_ds.to_zarr(sample_path, mode="w")
    print("Saved real datasource inspection sample:", sample_path)
else:
    dataset_summary = {"error": "No real data source sample available"}

# %% [markdown]
# ## 5. Prognostic Model Loading
#
# We prefer `FCN3` / FourCastNet3. If it fails because weights, network, or optional
# packages are missing, the exact reason is printed and we try other Earth2Studio
# prognostic models. This mirrors the official model-loading examples:
#
# ```python
# package = FCN3.load_default_package()
# model = FCN3.load_model(package)
# ```
#
# ภาษาไทย: cell นี้โหลด model จริง ไม่ใช่ fake forecast model

# %%
model = None
MODEL_NAME = None
model_load_errors: list[str] = []

if HAS_EARTH2STUDIO:
    for model_name, model_cls in [("FCN3", FCN3), ("FCN", FCN), ("DLWP", DLWP), ("SFNO", SFNO)]:
        try:
            package = model_cls.load_default_package()
            model = model_cls.load_model(package)
            MODEL_NAME = model_name
            print(f"Selected model: {MODEL_NAME}")
            input_vars = list(model.input_coords()["variable"])
            print("Input variable count:", len(input_vars))
            print("First 20 input variables:", input_vars[:20])
            break
        except Exception as exc:
            reason = f"{model_name} failed: {type(exc).__name__}: {exc}"
            model_load_errors.append(reason)
            print(reason)

if model is None:
    print("No pretrained prognostic model loaded.")
    print("\n".join(model_load_errors))

# %% [markdown]
# ## 6. Deterministic Forecast Inference with ZarrBackend
#
# This is intentionally close to the official Earth2Studio deterministic workflow:
#
# ```python
# from earth2studio.io import ZarrBackend
# import earth2studio.run as run
# io = ZarrBackend(file_name="outputs/...")
# io = run.deterministic([init_time], nsteps, model, data, io, output_coords=...)
# ```
#
# Real-first policy:
#
# 1. Try `REAL_EARTH2STUDIO_INFERENCE` using selected model + selected data source.
# 2. If all inference runs fail, fall back to `REAL_SAMPLE_DATASET` from GFS/ARCO.
# 3. Only if no real data can be loaded, use `SYNTHETIC_FALLBACK`.

# %%
def init_label(init_time: pd.Timestamp) -> str:
    """Compact initialization label for filenames."""
    return pd.Timestamp(init_time).strftime("%Y%m%d%H")


def open_forecast_dataset(path: Path) -> xr.Dataset:
    """Open a Zarr forecast dataset."""
    return xr.open_zarr(path)


deterministic_records: list[dict[str, Any]] = []
deterministic_paths: list[Path] = []
real_inference_errors: list[str] = []
INFERENCE_MODE = "UNSET"

if model is not None and data is not None:
    for init_time in RUN_INIT_TIMES:
        forecast_path = FORECAST_DIR / f"deterministic_{MODEL_NAME}_{DATA_SOURCE_NAME}_{init_label(init_time)}.zarr"
        try:
            chunks = {"time": 1, "lead_time": 1}
            io = ZarrBackend(file_name=str(forecast_path), chunks=chunks, backend_kwargs={"overwrite": True})
            io = deterministic(
                [init_time.to_pydatetime()],
                N_STEPS,
                model,
                data,
                io,
                output_coords=OUTPUT_COORDS,
                device=torch.device("cuda" if HAS_GPU else "cpu") if HAS_TORCH else None,
                verbose=True,
            )
            deterministic_paths.append(forecast_path)
            deterministic_records.append(
                {
                    "init_time": init_time,
                    "path": str(forecast_path),
                    "status": "success",
                    "model": MODEL_NAME,
                    "data_source": DATA_SOURCE_NAME,
                }
            )
        except Exception as exc:
            reason = f"deterministic {init_time} failed: {type(exc).__name__}: {exc}"
            real_inference_errors.append(reason)
            deterministic_records.append(
                {
                    "init_time": init_time,
                    "path": str(forecast_path),
                    "status": "failed",
                    "model": MODEL_NAME,
                    "data_source": DATA_SOURCE_NAME,
                    "reason": reason,
                }
            )
            print(reason)
            print(traceback.format_exc(limit=3))

if deterministic_paths:
    INFERENCE_MODE = "REAL_EARTH2STUDIO_INFERENCE"
else:
    print("No deterministic real inference path succeeded.")

pd.DataFrame(deterministic_records).to_csv(TABLE_DIR / "deterministic_run_log.csv", index=False)
pd.DataFrame(deterministic_records)

# %% [markdown]
# ## 7. Fallback Dataset Construction
#
# If real Earth2Studio inference fails, use a real sample data source as fallback before
# synthetic data. This fallback is explicitly labeled and does not pretend to be model
# inference.

# %%
def make_real_sample_forecast_from_datasource(source: Any, init_times: pd.DatetimeIndex) -> list[Path]:
    """Create a persistence-style dataset from real GFS/ARCO fields if model inference fails."""
    paths: list[Path] = []
    for init_time in init_times:
        da = source([init_time.to_pydatetime()], inspection_variables)
        ds0 = da.to_dataset(dim="variable").sel(
            lat=slice(BBOX["lat_max"], BBOX["lat_min"]),
            lon=slice(BBOX["lon_min"], BBOX["lon_max"]),
        )
        lead_times = pd.to_timedelta(np.arange(N_STEPS + 1) * 6, unit="h")
        expanded = xr.concat([ds0.squeeze("time", drop=True)] * len(lead_times), dim="lead_time")
        expanded = expanded.assign_coords(time=[init_time], lead_time=lead_times).expand_dims("time")
        expanded.attrs["mode"] = "REAL_SAMPLE_DATASET"
        expanded.attrs["note"] = "Real data source fields persisted across lead time; not model inference."
        path = FORECAST_DIR / f"real_sample_dataset_{DATA_SOURCE_NAME}_{init_label(init_time)}.zarr"
        expanded.to_zarr(path, mode="w")
        paths.append(path)
    return paths


def make_synthetic_last_resort(init_times: pd.DatetimeIndex) -> list[Path]:
    """Create a last-resort synthetic dataset only if real data and real inference fail."""
    paths: list[Path] = []
    lats = OUTPUT_COORDS["lat"]
    lons = OUTPUT_COORDS["lon"]
    lon2d, lat2d = np.meshgrid(lons, lats)
    for init_time in init_times:
        lead_times = pd.to_timedelta(np.arange(N_STEPS + 1) * 6, unit="h")
        t = np.arange(len(lead_times))
        spatial = np.exp(-(((lat2d - CENTER_LAT) ** 2) / (2 * 0.35**2) + ((lon2d - CENTER_LON) ** 2) / (2 * 0.35**2)))
        pulse = np.exp(-0.5 * ((t - 8) / 3.0) ** 2)
        arr = 45 + 25 * pulse[:, None, None] * spatial[None, :, :]
        ds = xr.Dataset(
            {
                "tcwv": (("time", "lead_time", "lat", "lon"), arr[None].astype("float32")),
                "t2m": (("time", "lead_time", "lat", "lon"), (298 + 0 * arr)[None].astype("float32")),
                "msl": (("time", "lead_time", "lat", "lon"), (100800 - 900 * pulse[:, None, None] * spatial[None])[None].astype("float32")),
                "u10m": (("time", "lead_time", "lat", "lon"), (4 + 5 * spatial[None])[None].astype("float32")),
                "v10m": (("time", "lead_time", "lat", "lon"), (3 + 4 * spatial[None])[None].astype("float32")),
                "q850": (("time", "lead_time", "lat", "lon"), (0.014 + 0.006 * pulse[:, None, None] * spatial[None])[None].astype("float32")),
                "u850": (("time", "lead_time", "lat", "lon"), (6 + 7 * spatial[None])[None].astype("float32")),
                "v850": (("time", "lead_time", "lat", "lon"), (5 + 6 * spatial[None])[None].astype("float32")),
            },
            coords={"time": [init_time], "lead_time": lead_times, "lat": lats, "lon": lons},
            attrs={"mode": "SYNTHETIC_FALLBACK", "note": "Last-resort synthetic data only."},
        )
        path = FORECAST_DIR / f"synthetic_fallback_{init_label(init_time)}.zarr"
        ds.to_zarr(path, mode="w")
        paths.append(path)
    return paths


if not deterministic_paths:
    if data is not None:
        try:
            deterministic_paths = make_real_sample_forecast_from_datasource(data, RUN_INIT_TIMES)
            INFERENCE_MODE = "REAL_SAMPLE_DATASET"
        except Exception as exc:
            print("REAL_SAMPLE_DATASET fallback failed:", repr(exc))
            deterministic_paths = make_synthetic_last_resort(RUN_INIT_TIMES)
            INFERENCE_MODE = "SYNTHETIC_FALLBACK"
    else:
        deterministic_paths = make_synthetic_last_resort(RUN_INIT_TIMES)
        INFERENCE_MODE = "SYNTHETIC_FALLBACK"

print("INFERENCE_MODE:", INFERENCE_MODE)
print("Forecast paths:")
for p in deterministic_paths:
    print(" -", p)

# %% [markdown]
# ## 8. Open and Inspect Forecast Output
#
# Required inspection after inference:
#
# - dimensions and coordinates
# - variables in output Zarr
# - time and lead-time range
# - spatial resolution
# - nearest grid point to Hat Yai

# %%
forecast_datasets = [open_forecast_dataset(path) for path in deterministic_paths]
primary_ds = forecast_datasets[0]

print("Primary forecast dataset")
print("------------------------")
print(primary_ds)
print("Dimensions:", dict(primary_ds.sizes))
print("Coordinates:", list(primary_ds.coords))
print("Variables:", list(primary_ds.data_vars))

lead_hours = pd.to_timedelta(primary_ds["lead_time"].values).total_seconds() / 3600
valid_times = pd.Timestamp(primary_ds["time"].values[0]) + pd.to_timedelta(primary_ds["lead_time"].values)
print("Initialization time:", pd.Timestamp(primary_ds["time"].values[0]))
print("Valid time range:", valid_times.min(), "to", valid_times.max())
print("Lead hours:", lead_hours[:5], "...", lead_hours[-1])
print("Spatial resolution:", float(abs(np.diff(primary_ds.lat.values).mean())), float(abs(np.diff(primary_ds.lon.values).mean())))

nearest_point = primary_ds.sel(lat=CENTER_LAT, lon=CENTER_LON, method="nearest")
print("Nearest Hat Yai grid point:", float(nearest_point.lat.values), float(nearest_point.lon.values))

# %% [markdown]
# ## 9. Derived Variables: Wind, Moisture Flux, and Rainfall-Risk Proxy
#
# Since FCN3 does not output precipitation here, this notebook builds a **risk proxy**
# from real model output:
#
# - `tcwv`: total column water vapor
# - `q850`: 850 hPa specific humidity
# - `u850/v850`: low-level wind and moisture transport proxy
# - `u10m/v10m`: near-surface wind
# - `msl`: mean sea level pressure
#
# ภาษาไทย: ค่า proxy นี้ไม่ใช่ฝนจริง แต่เป็นสัญญาณความเสี่ยงจาก moisture + wind + pressure
# เพื่อสาธิต early warning logic เมื่อ model output ไม่มี precipitation

# %%
def add_derived_risk_variables(ds: xr.Dataset) -> xr.Dataset:
    """Add wind speed, moisture-flux proxy, and rainfall-risk proxy to a forecast dataset."""
    out = ds.copy()
    if "u10m" in out and "v10m" in out:
        out["wind10m"] = np.sqrt(out["u10m"] ** 2 + out["v10m"] ** 2)
    else:
        out["wind10m"] = xr.zeros_like(next(iter(out.data_vars.values())))

    if "u850" in out and "v850" in out:
        out["wind850"] = np.sqrt(out["u850"] ** 2 + out["v850"] ** 2)
    else:
        out["wind850"] = xr.zeros_like(out["wind10m"])

    tcwv = out["tcwv"] if "tcwv" in out else xr.zeros_like(out["wind10m"]) + 45
    q850 = out["q850"] if "q850" in out else xr.zeros_like(out["wind10m"]) + 0.012
    q850_gkg = q850 * 1000.0
    msl = out["msl"] if "msl" in out else xr.zeros_like(out["wind10m"]) + 101000
    msl_hpa = xr.where(msl > 2000, msl / 100.0, msl)

    out["moisture_flux_850_proxy"] = out["wind850"] * q850_gkg
    tcwv_norm = ((tcwv - 45.0) / 25.0).clip(0, 1.5)
    q_norm = ((q850_gkg - 10.0) / 10.0).clip(0, 1.5)
    wind_norm = (out["wind10m"] / 16.0).clip(0, 1.5)
    flux_norm = ((out["moisture_flux_850_proxy"] - 80.0) / 180.0).clip(0, 1.5)
    pressure_norm = ((1010.0 - msl_hpa) / 18.0).clip(0, 1.5)

    # Unit is mm-equivalent proxy per 6h, not precipitation.
    out["rainfall_risk_proxy_6h"] = (
        8.0 + 26.0 * tcwv_norm + 22.0 * q_norm + 16.0 * wind_norm + 18.0 * flux_norm + 12.0 * pressure_norm
    ).clip(0, 90)
    out["rainfall_risk_proxy_24h"] = out["rainfall_risk_proxy_6h"].rolling(lead_time=4, min_periods=1).sum()
    out["rainfall_risk_proxy_72h"] = out["rainfall_risk_proxy_6h"].rolling(lead_time=12, min_periods=1).sum()
    out["risk_index"] = (out["rainfall_risk_proxy_24h"] / 220.0 * 0.55 + out["rainfall_risk_proxy_72h"] / 520.0 * 0.45).clip(0, 1)
    out.attrs["risk_proxy_note"] = "Rainfall-risk proxy from real model variables; not actual precipitation."
    return out


derived_datasets = [add_derived_risk_variables(ds) for ds in forecast_datasets]
primary_derived = derived_datasets[0]
print("Derived variables:", [v for v in primary_derived.data_vars if "proxy" in v or "risk" in v or "wind" in v])

# %% [markdown]
# ## 10. Optional Real Ensemble Inference
#
# This section follows the official ensemble workflow:
#
# ```python
# from earth2studio.perturbation import SphericalGaussian
# from earth2studio.run import ensemble
# sg = SphericalGaussian(noise_amplitude=0.05)
# io = ensemble([init_time], nsteps, nensemble, model, data, io, sg)
# ```
#
# The default ensemble is small (`2` members, first initialization only) so the notebook
# remains runnable. Increase environment variables for a heavier run.

# %%
ensemble_path = None
ensemble_ds = None
ensemble_error = None

if RUN_REAL_ENSEMBLE and model is not None and data is not None and INFERENCE_MODE == "REAL_EARTH2STUDIO_INFERENCE":
    try:
        ensemble_path = FORECAST_DIR / f"ensemble_{MODEL_NAME}_{DATA_SOURCE_NAME}_{init_label(RUN_INIT_TIMES[0])}.zarr"
        sg = SphericalGaussian(noise_amplitude=0.05)
        io_ens = ZarrBackend(
            file_name=str(ensemble_path),
            chunks={"ensemble": 1, "time": 1, "lead_time": 1},
            backend_kwargs={"overwrite": True},
        )
        io_ens = ensemble(
            [RUN_INIT_TIMES[0].to_pydatetime()],
            ENSEMBLE_NSTEPS,
            ENSEMBLE_MEMBERS,
            model,
            data,
            io_ens,
            sg,
            batch_size=1,
            output_coords=ENSEMBLE_OUTPUT_COORDS,
        )
        ensemble_ds = add_derived_risk_variables(xr.open_zarr(ensemble_path))
        print("Real ensemble inference succeeded:", ensemble_path)
    except Exception as exc:
        ensemble_error = f"{type(exc).__name__}: {exc}"
        print("Real ensemble inference failed:", ensemble_error)
else:
    print("Real ensemble skipped. RUN_REAL_ENSEMBLE:", RUN_REAL_ENSEMBLE, "INFERENCE_MODE:", INFERENCE_MODE)

# %% [markdown]
# ## 11. Build Forecast Tables and Alert Levels
#
# Alert logic:
#
# If actual precipitation exists:
# - 24h rainfall > 100 mm = YELLOW
# - 24h rainfall > 150 mm or 72h rainfall > 250 mm = ORANGE
# - 24h rainfall > 200 mm or 72h rainfall > 350 mm = RED
# - 24h rainfall > 300 mm = PURPLE
#
# Here precipitation is unavailable, so the same thresholds are applied to the clearly
# labeled **rainfall-risk proxy**, not rainfall.

# %%
ALERT_ORDER = ["GREEN", "YELLOW", "ORANGE", "RED", "PURPLE"]
ALERT_SCORE = {level: i for i, level in enumerate(ALERT_ORDER)}
ALERT_COLOR = {"GREEN": "#2ca02c", "YELLOW": "#f1c40f", "ORANGE": "#ff7f0e", "RED": "#d62728", "PURPLE": "#7b3294"}


def valid_time_values(ds: xr.Dataset) -> pd.DatetimeIndex:
    """Return valid times from Earth2Studio time + lead_time coordinates."""
    base_time = pd.Timestamp(ds["time"].values[0])
    return pd.DatetimeIndex(base_time + pd.to_timedelta(ds["lead_time"].values))


def decide_alert_from_proxy(proxy24: float, proxy72: float, risk_index: float) -> str:
    """Apply threshold-style alerting to rainfall-risk proxy."""
    if proxy24 > 300 or risk_index > 0.90:
        return "PURPLE"
    if proxy24 > 200 or proxy72 > 350 or risk_index > 0.78:
        return "RED"
    if proxy24 > 150 or proxy72 > 250 or risk_index > 0.62:
        return "ORANGE"
    if proxy24 > 100 or risk_index > 0.45:
        return "YELLOW"
    return "GREEN"


def action_for_alert(level: str) -> str:
    """Return recommended action for an alert level."""
    return {
        "GREEN": "Normal monitoring",
        "YELLOW": "Heavy rain watch",
        "ORANGE": "Flood preparedness",
        "RED": "Severe flood warning / evacuation advisory",
        "PURPLE": "Extreme emergency",
    }[level]


forecast_rows: list[dict[str, Any]] = []
alert_rows: list[dict[str, Any]] = []

for ds in derived_datasets:
    init_time = pd.Timestamp(ds["time"].values[0])
    vt = valid_time_values(ds)
    point = ds.sel(lat=CENTER_LAT, lon=CENTER_LON, method="nearest")
    domain = ds.max(dim=["lat", "lon"])
    for i, valid_time in enumerate(vt):
        lead_hour = int(pd.to_timedelta(ds["lead_time"].values[i]).total_seconds() / 3600)
        proxy24 = float(point["rainfall_risk_proxy_24h"].isel(lead_time=i).values.squeeze())
        proxy72 = float(point["rainfall_risk_proxy_72h"].isel(lead_time=i).values.squeeze())
        risk_index = float(point["risk_index"].isel(lead_time=i).values.squeeze())
        domain_proxy24 = float(domain["rainfall_risk_proxy_24h"].isel(lead_time=i).values.squeeze())
        level = decide_alert_from_proxy(domain_proxy24, float(domain["rainfall_risk_proxy_72h"].isel(lead_time=i).values.squeeze()), risk_index)
        row = {
            "init_time": init_time,
            "valid_time": valid_time,
            "lead_hour": lead_hour,
            "source_mode": INFERENCE_MODE,
            "model": MODEL_NAME if MODEL_NAME else "none",
            "data_source": DATA_SOURCE_NAME if DATA_SOURCE_NAME else "none",
            "point_proxy_24h": proxy24,
            "point_proxy_72h": proxy72,
            "domain_proxy_24h": domain_proxy24,
            "domain_proxy_72h": float(domain["rainfall_risk_proxy_72h"].isel(lead_time=i).values.squeeze()),
            "tcwv_point": float(point["tcwv"].isel(lead_time=i).values.squeeze()) if "tcwv" in point else np.nan,
            "wind10m_point": float(point["wind10m"].isel(lead_time=i).values.squeeze()),
            "risk_index_point": risk_index,
            "recommended_alert_level": level,
            "suggested_action": action_for_alert(level),
            "precipitation_note": "rainfall-risk proxy, not actual precipitation",
        }
        forecast_rows.append(row)
        alert_rows.append({k: row[k] for k in ["init_time", "valid_time", "lead_hour", "recommended_alert_level", "suggested_action", "risk_index_point"]})

forecast_table = pd.DataFrame(forecast_rows)
alert_timeline = pd.DataFrame(alert_rows)

summary_rows = []
for init_time, group in forecast_table.groupby("init_time"):
    max_row = group.loc[group["domain_proxy_72h"].idxmax()]
    first_orange_plus = group[group["recommended_alert_level"].map(ALERT_SCORE) >= ALERT_SCORE["ORANGE"]]
    summary_rows.append(
        {
            "init_time": init_time,
            "max_domain_proxy_24h": group["domain_proxy_24h"].max(),
            "max_domain_proxy_72h": group["domain_proxy_72h"].max(),
            "max_risk_valid_time": max_row["valid_time"],
            "first_orange_or_higher_time": first_orange_plus["valid_time"].min() if not first_orange_plus.empty else pd.NaT,
            "highest_alert_level": max(group["recommended_alert_level"], key=lambda x: ALERT_SCORE[x]),
            "source_mode": INFERENCE_MODE,
        }
    )

forecast_summary = pd.DataFrame(summary_rows)
event_timeline = (
    forecast_table.assign(date=lambda x: pd.to_datetime(x["valid_time"]).dt.date)
    .sort_values(["date", "init_time", "lead_hour"])
    .groupby("date", as_index=False)
    .tail(1)
)

event_timeline.to_csv(TABLE_DIR / "event_timeline.csv", index=False)
alert_timeline.to_csv(TABLE_DIR / "alert_timeline.csv", index=False)
forecast_summary.to_csv(TABLE_DIR / "forecast_summary.csv", index=False)

display(forecast_summary)
display(event_timeline.head(12))

# %% [markdown]
# ## 12. Visualization Helpers
#
# Plotting uses Cartopy when available and falls back to lon/lat pcolormesh.
#
# ภาษาไทย: ภาพทั้งหมดใช้ output ที่มาจาก real inference หรือ fallback mode ที่ระบุไว้ชัดเจนใน title

# %%
def savefig(name: str) -> Path:
    """Save current figure to the figure directory."""
    path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(path)
    return path


def shade_critical(ax: Any) -> None:
    """Shade the critical Hat Yai flood window."""
    ax.axvspan(MAIN_ESCALATION_START, MAIN_ESCALATION_END, color="#d62728", alpha=0.12, label="Critical flood window")


def plot_domain_map() -> None:
    """Plot regional domain map around southern Thailand."""
    fig = plt.figure(figsize=(8, 7))
    if HAS_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([98.5, 102.5, 5.0, 9.0], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f0")
        ax.add_feature(cfeature.OCEAN, facecolor="#dbeafe")
        ax.coastlines(resolution="10m", linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    else:
        ax = plt.axes()
        ax.set_xlim(98.5, 102.5)
        ax.set_ylim(5.0, 9.0)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.plot([BBOX["lon_min"], BBOX["lon_max"], BBOX["lon_max"], BBOX["lon_min"], BBOX["lon_min"]],
            [BBOX["lat_min"], BBOX["lat_min"], BBOX["lat_max"], BBOX["lat_max"], BBOX["lat_min"]],
            color="black", linewidth=2, label="Analysis bbox")
    ax.scatter([CENTER_LON], [CENTER_LAT], color="red", marker="*", s=130, label="Hat Yai")
    ax.set_title("Regional Domain Around Southern Thailand / ขอบเขตพื้นที่ศึกษาหาดใหญ่")
    ax.legend(loc="upper right")
    savefig("01_regional_domain_map.png")


def plot_field_map(ds: xr.Dataset, variable: str, lead_hour: int, title: str, file_name: str, cmap: str = "viridis") -> None:
    """Plot a forecast field map."""
    lead_index = int(np.argmin(np.abs(pd.to_timedelta(ds["lead_time"].values).total_seconds() / 3600 - lead_hour)))
    field = ds[variable].isel(time=0, lead_time=lead_index)
    valid_time = valid_time_values(ds)[lead_index]
    fig = plt.figure(figsize=(9, 7))
    if HAS_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        mesh = ax.pcolormesh(ds.lon, ds.lat, field, transform=ccrs.PlateCarree(), cmap=cmap, shading="auto")
        ax.coastlines(resolution="10m", linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    else:
        ax = plt.axes()
        mesh = ax.pcolormesh(ds.lon, ds.lat, field, cmap=cmap, shading="auto")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.scatter([CENTER_LON], [CENTER_LAT], c="red", s=90, marker="*", label="Hat Yai")
    ax.set_title(f"{title}\nvalid={valid_time:%Y-%m-%d %H:%M UTC} | {INFERENCE_MODE}")
    ax.legend(loc="upper right")
    plt.colorbar(mesh, ax=ax, shrink=0.75, label=variable)
    savefig(file_name)


def point_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Extract nearest Hat Yai point forecast as a DataFrame."""
    point = ds.sel(lat=CENTER_LAT, lon=CENTER_LON, method="nearest").isel(time=0)
    df = pd.DataFrame({"valid_time": valid_time_values(ds)})
    for var in ["t2m", "msl", "tcwv", "wind10m", "rainfall_risk_proxy_6h", "rainfall_risk_proxy_24h", "rainfall_risk_proxy_72h", "risk_index"]:
        if var in point:
            values = point[var].values
            df[var] = np.asarray(values).reshape(-1)
    if "t2m" in df:
        df["t2m_c"] = df["t2m"] - 273.15
    if "msl" in df:
        df["msl_hpa"] = np.where(df["msl"] > 2000, df["msl"] / 100.0, df["msl"])
    return df


def plot_point_timeseries(ds: xr.Dataset) -> None:
    """Plot time series at nearest Hat Yai grid point."""
    df = point_dataframe(ds)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["valid_time"], df["tcwv"], marker="o", label="TCWV")
    ax1.plot(df["valid_time"], df["wind10m"], marker="o", label="10m wind speed")
    shade_critical(ax1)
    ax1.set_ylabel("TCWV / wind")
    ax1.set_title("Time Series at Hat Yai Nearest Grid Point / สัญญาณพยากรณ์ที่จุดหาดใหญ่")
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %Hh"))
    savefig("03_hatyai_point_timeseries.png")


def plot_proxy_accumulations(ds: xr.Dataset) -> None:
    """Plot 24h and 72h rainfall-risk proxy at Hat Yai."""
    df = point_dataframe(ds)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["valid_time"], df["rainfall_risk_proxy_24h"], marker="o", label="24h risk proxy")
    ax.plot(df["valid_time"], df["rainfall_risk_proxy_72h"], marker="o", label="72h risk proxy")
    shade_critical(ax)
    ax.axhline(100, color="#f1c40f", linestyle="--", label="YELLOW proxy threshold")
    ax.axhline(150, color="#ff7f0e", linestyle="--", label="ORANGE proxy threshold")
    ax.axhline(200, color="#d62728", linestyle="--", label="RED proxy threshold")
    ax.set_title("24h / 72h Rainfall-Risk Proxy / ดัชนี proxy ฝนสะสม 24 และ 72 ชั่วโมง")
    ax.set_ylabel("Proxy units (not mm rainfall)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %Hh"))
    savefig("04_rainfall_risk_proxy_accumulation.png")


def plot_ensemble_plume() -> None:
    """Plot real ensemble plume if ensemble output exists."""
    if ensemble_ds is None:
        print("No real ensemble dataset available; skipping ensemble plume.")
        return
    ens_point = ensemble_ds.sel(lat=CENTER_LAT, lon=CENTER_LON, method="nearest").isel(time=0)
    vt = valid_time_values(ensemble_ds)
    fig, ax = plt.subplots(figsize=(12, 5))
    for member in ensemble_ds["ensemble"].values:
        y = ens_point["rainfall_risk_proxy_24h"].sel(ensemble=member).values.reshape(-1)
        ax.plot(vt, y, color="#60a5fa", alpha=0.45, linewidth=1.3)
    mean_y = ens_point["rainfall_risk_proxy_24h"].mean(dim="ensemble").values.reshape(-1)
    ax.plot(vt, mean_y, color="#1d4ed8", linewidth=2.5, label="ensemble mean")
    ax.axhline(100, color="#f1c40f", linestyle="--")
    ax.axhline(150, color="#ff7f0e", linestyle="--")
    ax.set_title("Real Earth2Studio Ensemble Plume / Ensemble rainfall-risk proxy")
    ax.set_ylabel("24h proxy units (not rainfall)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %Hh"))
    savefig("05_real_ensemble_plume.png")


def plot_alert_timeline() -> None:
    """Plot alert-level timeline from forecast output."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    plot_df = alert_timeline.copy()
    plot_df["score"] = plot_df["recommended_alert_level"].map(ALERT_SCORE)
    plot_df["color"] = plot_df["recommended_alert_level"].map(ALERT_COLOR)
    ax.scatter(plot_df["valid_time"], plot_df["score"], c=plot_df["color"], s=65)
    shade_critical(ax)
    ax.set_yticks(range(len(ALERT_ORDER)), ALERT_ORDER)
    ax.set_title("Alert-Level Timeline from Forecast Output / จุดที่ระบบควรยกระดับการแจ้งเตือน")
    ax.set_ylabel("Alert level")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    savefig("06_alert_level_timeline.png")


def plot_lead_time_comparison() -> None:
    """Plot forecast comparison across initialization dates."""
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = pd.to_datetime(forecast_summary["init_time"]).dt.strftime("%b %d")
    ax.bar(labels, forecast_summary["max_domain_proxy_72h"], color="#2563eb")
    ax.axhline(250, color="#ff7f0e", linestyle="--", label="ORANGE proxy threshold")
    ax.axhline(350, color="#d62728", linestyle="--", label="RED proxy threshold")
    ax.set_title("Forecast Lead-Time Comparison Across Init Dates / เปรียบเทียบสัญญาณจากแต่ละรอบพยากรณ์")
    ax.set_ylabel("Max domain 72h proxy (not rainfall)")
    ax.legend(loc="upper left")
    savefig("07_forecast_lead_time_comparison.png")


def plot_risk_index_timeline(ds: xr.Dataset) -> None:
    """Plot flood-risk index at Hat Yai point."""
    df = point_dataframe(ds)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["valid_time"], df["risk_index"], marker="o", color="#7b3294")
    shade_critical(ax)
    ax.axhline(0.62, color="#ff7f0e", linestyle="--", label="ORANGE")
    ax.axhline(0.78, color="#d62728", linestyle="--", label="RED")
    ax.axhline(0.90, color="#7b3294", linestyle="--", label="PURPLE")
    ax.set_ylim(0, 1.05)
    ax.set_title("Flood-Risk Index from Real Forecast Variables / ดัชนีความเสี่ยงจาก output model")
    ax.set_ylabel("Risk index")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %Hh"))
    savefig("08_risk_index_timeline.png")


if HAS_MATPLOTLIB:
    plot_domain_map()
    plot_field_map(primary_derived, "tcwv", 24, "Forecast TCWV Field Around Hat Yai", "02_forecast_tcwv_field_map.png", "Blues")
    plot_point_timeseries(primary_derived)
    plot_proxy_accumulations(primary_derived)
    plot_ensemble_plume()
    plot_alert_timeline()
    plot_lead_time_comparison()
    plot_risk_index_timeline(primary_derived)
    plt.close("all")

print("Saved figures to:", FIGURE_DIR)

# %% [markdown]
# ## 13. Thai Executive Interpretation
#
# **ทำไมต้องใช้ Earth2Studio**
#
# Earth2Studio ช่วยให้ทีมไทยสามารถรัน AI weather inference แบบ reproducible:
# มี data source, model, IO backend, deterministic/ensemble workflow และ output ที่ inspect ได้
# ไม่ใช่แค่ dashboard ที่วาดกราฟจากค่าคงที่
#
# **Dataset ที่ใช้คืออะไร**
#
# Notebook นี้พยายามใช้ `GFS` เป็น data source หลัก เพราะโหลดข้อมูลจริงวันที่ 2025-11-17 ได้
# หาก GFS ล้มเหลวจะลอง ARCO/NCAR ERA5 และ fallback ที่ label ชัดเจน
#
# **Model ที่ใช้คืออะไร**
#
# Model หลักคือ `FCN3` ถ้าโหลดสำเร็จ โดยเป็น global prognostic model ของ Earth2Studio
# ที่ให้ output เป็นตัวแปรบรรยากาศหลายระดับ เช่น TCWV, MSLP, winds, humidity, geopotential
#
# **Inference ทำอะไรจริง**
#
# `earth2studio.run.deterministic` fetch initial condition จาก data source จริง,
# ส่งเข้า model จริงบน CUDA/CPU, เขียน output ลง Zarr และ notebook อ่านกลับมา plot
#
# **ข้อจำกัด**
#
# ใน environment นี้ FCN3 ไม่มี precipitation output โดยตรง จึงใช้ rainfall-risk proxy จาก moisture/wind/pressure
# ไม่ควรเรียก proxy นี้ว่า “ฝนจริง” หรือ “ปริมาณฝนพยากรณ์จริง”
#
# **ข้อมูลจริงที่ควรเชื่อมต่อเพิ่ม**
#
# TMD, ThaiWater, GPM IMERG, GSMaP, radar, rain gauge, water-level sensors,
# DEM, drainage map, Khlong U-Taphao, Khlong R.1, Songkhla Lake drainage constraints,
# flood maps, exposure layers, shelters, hospitals, schools, roads, pumps, and rescue assets.

# %% [markdown]
# ## 14. Run Metadata and Completion

# %%
run_metadata = {
    "case": CASE_NAME,
    "mode": INFERENCE_MODE,
    "data_source_used": DATA_SOURCE_NAME,
    "model_used": MODEL_NAME,
    "real_inference_success": INFERENCE_MODE == "REAL_EARTH2STUDIO_INFERENCE",
    "real_ensemble_success": ensemble_ds is not None,
    "ensemble_error": ensemble_error,
    "n_steps": N_STEPS,
    "forecast_hours": int(N_STEPS * 6),
    "init_times_requested": [str(t) for t in INIT_TIMES],
    "init_times_run": [str(t) for t in RUN_INIT_TIMES],
    "forecast_outputs": [str(p) for p in deterministic_paths],
    "figures_dir": str(FIGURE_DIR),
    "tables_dir": str(TABLE_DIR),
    "data_source_errors": data_source_errors,
    "model_load_errors": model_load_errors,
    "real_inference_errors": real_inference_errors,
    "precipitation_note": "No direct precipitation in selected model output; rainfall-risk proxy used and clearly labeled.",
    "safety_note": "Research/demo only. Not an official life-safety warning system.",
}

(OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(run_metadata, indent=2, ensure_ascii=False))
print("Notebook completed successfully.")
