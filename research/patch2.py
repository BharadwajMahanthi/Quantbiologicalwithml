#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch2.py – update the seven JSON files used by quantv1.2.py
with the Chernobyl ‘Reference Site’ dataset (AE02F4E8).

Author: <you> • 2025-07-06
"""

from __future__ import annotations
import json, csv, re, io, zipfile, shutil, logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────
# 1.  CONFIG --- edit just these two folder paths
# ──────────────────────────────────────────────────────────────
JSON_ROOT   = Path(r"C:\Users\mbpd1\Downloads\pp-mywork\Quantbiologicalwithml")
AE_PACKAGE  = Path(r"C:\Users\mbpd1\Downloads\pp-mywork\ae02f4e8-9486-4b47-93ef-e49dd9ddecd4")

# ──────────────────────────────────────────────────────────────
BACKUP_SUFFIX = ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE      = JSON_ROOT / "json_update.log"

pd.options.mode.chained_assignment = None     # silence pandas warning

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, "w", "utf-8"),
              logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 2.  Generic helpers
# ──────────────────────────────────────────────────────────────
def load_json(name: str) -> Any:
    p = JSON_ROOT / name
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)

def save_json(name: str, data: Any) -> None:
    p = JSON_ROOT / name
    if p.exists():
        p_backup = p.with_suffix(p.suffix + BACKUP_SUFFIX)
        shutil.copy2(p, p_backup)
        log.info(f"backup  {p_backup.name}")
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
    log.info(f"updated {name}")

def read_csv(path: Path, **kw) -> pd.DataFrame:
    """Try UTF-8 first, then latin-1; let pandas guess the separator."""
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python", **kw)
        except UnicodeDecodeError:
            continue
    raise

def first_match(cols: list[str], *patterns: str) -> str | None:
    for p in patterns:
        for c in cols:
            if re.fullmatch(p, c, flags=re.I):
                return c
    return None

# ──────────────────────────────────────────────────────────────
# 3.  Load existing JSON files
# ──────────────────────────────────────────────────────────────
species_json   = load_json("species_data.json")
mushroom_json  = load_json("mushroom_data.json")
location_json  = load_json("location_data.json")
locations_json = load_json("locations.json")
radiation_json = load_json("radiation_data.json")
zones_json     = load_json("zones.json")
empirical_json = load_json("empirical_data.json")

# ──────────────────────────────────────────────────────────────
# 4.  Read AE02F4E8 CSVs
# ──────────────────────────────────────────────────────────────
data_dir = AE_PACKAGE / "data"
csvs = {
    "soil_rad"     : data_dir / "Soil_radionuclide_data.csv",
    "biota_total"  : data_dir / "Biota_total_dose.csv",
}

dfs: dict[str, pd.DataFrame] = {k: read_csv(v) for k, v in csvs.items()}
soil_df  = dfs["soil_rad"]
total_df = dfs["biota_total"]

# Detect coordinate column names
lat_col = first_match(soil_df.columns, r"lat.*", r"latitude.*")
lon_col = first_match(soil_df.columns, r"lon.*", r"longitude.*")

if not lat_col or not lon_col:
    raise ValueError("Could not find latitude / longitude columns in "
                     "Soil_radionuclide_data.csv")

# Detect ambient γ-dose columns (µSv h-1)
amb_cols = [c for c in soil_df.columns if re.search("micro.?Sv_per_hour", c, re.I)]
if not amb_cols:
    raise ValueError("No ambient dose columns in Soil_radionuclide_data.csv")

# ──────────────────────────────────────────────────────────────
# 5.  Build / extend radiation_data.json and locations.json
# ──────────────────────────────────────────────────────────────
rad_out = radiation_json.copy()
rad_out.setdefault("_units", "µSv per hour")

for _, row in soil_df.iterrows():
    lat, lon = float(row[lat_col]), float(row[lon_col])
    key = f"RefSite_{lat:.5f}_{lon:.5f}"

    meas = row[amb_cols].astype(float).dropna().tolist()
    if not meas:
        continue
    background = float(np.mean(meas))

    rad_out[key] = {
        "Background_µSv_hr"  : round(background, 3),
        "Measurement_µSv_hr" : [round(x, 3) for x in (meas if len(meas) >= 2 else meas*2)],
    }

    if key not in locations_json:
        locations_json[key] = {
            "Latitude"         : lat,
            "Longitude"        : lon,
            "Habitat"          : "ReferenceSite",
            "Radiation_µSv_hr" : round(background, 3),
        }

# ──────────────────────────────────────────────────────────────
# 6.  Build / extend empirical_data.json  (mean total dose rate)
# ──────────────────────────────────────────────────────────────
emp_out = empirical_json.copy()

dose_cols = [c for c in total_df.columns if c.endswith("_microGy_h-1")]
if not dose_cols:
    raise ValueError("Cannot find dose-rate columns in Biota_total_dose.csv")

# Convert to long form and average
long = total_df.melt(id_vars="Sample_ID", value_vars=dose_cols,
                     var_name="species_stat", value_name="dose")
long[["Species","stat"]] = long.species_stat.str.split("_total_dose_rate_", expand=True)
dose_mean = long.groupby("Species").dose.mean()

for sp_raw, val in dose_mean.items():
    sp = sp_raw.replace("_", " ").title()
    emp_out.setdefault(sp, {})
    emp_out[sp]["Radiation_µSv_hr"] = round(float(val)/1000, 6)   # µGy≈µSv

# ──────────────────────────────────────────────────────────────
# 7.  Add any new species to species_data.json (minimal defaults)
# ──────────────────────────────────────────────────────────────
DEFAULT_SPEC = {
    "Melanin"          : False,
    "Common Name"      : "",
    "Habitat Weight"   : 1.0,
    "Preferred Habitat": "ReferenceSite",
    "Substrate"        : "Unknown",
    "Min"              : 0.05,
    "Max"              : 0.2,
    "Median_Bqkg"      : 0.1,
    "CapDiameter_cm"   : [2, 8],
    "StemLength_cm"    : [2, 8],
    "Edibility"        : "Unknown",
}

for sp in emp_out:
    if sp not in species_json:
        species_json[sp] = DEFAULT_SPEC.copy()
        species_json[sp]["Common Name"] = sp.split()[-1]

# ──────────────────────────────────────────────────────────────
# 8.  Save back – with backups
# ──────────────────────────────────────────────────────────────
save_json("radiation_data.json", rad_out)
save_json("empirical_data.json",  emp_out)
save_json("species_data.json",    species_json)
save_json("locations.json",       locations_json)

log.info("All JSON files updated ✓ – see json_update.log for details.")
print("✅ JSON files updated successfully")
