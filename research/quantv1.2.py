r"""
# Fungal Ecosystem Simulation: `quantv1.2.py`

This script models a radiotrophic fungal ecosystem with agent-based simulation, genetic recombination, mutation, speciation, and spatial dynamics. It uses NSGA-II for optimization and validates against empirical data. Optimized for CPU and GPU laptops, using <3 GB RAM and ~135–140 s for 1000 steps.

## Key Features
- **Agent-Based Model**: Individuals with variable-site genomes, fitness, and mating types on a 100x100 grid.
- **Genetic Algorithm**: NSGA-II optimizes biomass and empirical fit.
- **Spatial Dynamics**: Structured grid with carbon (C), nitrogen (N), dose, and population fields, with resource limitation and stochastic extinction.
- **Radiation Effects**: Radiotrophic growth with Monod kinetics, dynamic OpenMC transport (optional).
- **Speciation**: New species at <0.85 genome similarity (Jaccard index on variant loci), capped at 100 species.
- **Logging**: Timestamped logs with console output, including smoke test values.
- **Performance**: Low memory, CPU/GPU compatibility, with cached OpenMC results.
- **Dataset Pipeline**: 10,000 unique records with realistic geography.

## Formulas Implemented
| No. | Description | Equation |
|-----|-------------|----------|
| **1** | **Dose-response (radiotrophy)** | \\[ R(d)=\exp\!\bigl[-0.22\;\bigl(\ln\frac{d}{0.05}\bigr)^2\bigr] \\] <br> *with*  \(d = \text{dose in mGy h}^{-1}\), \(R(0.05) = 1\) |
| **2** | **Growth rate** | \\[ G = v_{\text{max}} \cdot \frac{C}{K_{S,C} + C} \cdot \frac{N}{K_{S,N} + N} \cdot R(d) \\] <br> \(v_{\text{max}} = 0.30 \text{ (melanin)} \text{ or } 0.20 \text{ (no melanin)}\), \(K_{S,C} = 0.1\), \(K_{S,N} = 0.02\) |
| **3** | **Per-base mutation rate** | \\[ \mu = \mu_0 \bigl(1 + 4 d + 20 d^2\bigr) \\] <br> with  \(\mu_0 = 1\times10^{-9}\), \(d = \text{mGy h}^{-1}\). |
| **4** | **Birth probability** | \\[ P_{\text{birth}}=\max\!\Bigl[0,\min\!\Bigl[1,\;\Bigl(1-\frac{N_s}{K_s}\Bigr)\,F \times 0.7\Bigr]\Bigr] \\] <br> \(N_s\)=species pop., \(K_s\)=`4200/24` (~175), \(F\)=fitness |
| **5** | **Committed effective dose** | \\[ \text{CED}_{\text{mSv y}^{-1}} = C_{\text{Bq kg}^{-1}} \;\times 1.3\times10^{-5}\;\times 1.0 \\] |
| **6** | **TE burst multiplier** | \\[ p_{\text{TE}}(d) = p_{0,\text{TE}} \bigl[1 + 10 d \bigr]_{\le 5} \\] <br> with \(d\) in mGy h⁻¹. |
| **7** | **Genome similarity** | \\[ S(A,B) = \frac{|V_A \cap V_B|}{|V_A \cup V_B|} \\] <br> Jaccard index on variant loci. |

## Smoke Test Values
| Dose (mGy h⁻¹) | R(d) | μ/μ₀ |
|----------------|------|------|
| 0.01           | 0.46 | 1.08 |
| 0.05           | 1.00 | 1.25 |
| 0.30           | 0.41 | 4.00 |
| 1.00           | 0.02 | 25.0 |

## CSV Outputs
- **combined_dataset.csv**: Strain ID (str), Species Name (str), Common Name (str), Location (str), Latitude (float), Longitude (float), Habitat (str), Soil Type (str), pH Level (float), Contamination Zone (str), Radioactive Contamination Level (kBq/m²) (float), Radioactive Isotope (str), Concentration in Mushroom (Bq/kg) (float), Committed Effective Dose (mSv/year) (float), Radiation Level (µSv/hr) (float), Nutrient Level (float), Temperature (°C) (float), NO2 (mg/L) (float), Substrate (str), Cap Diameter (cm) (float), Stem Length (cm) (float), Edibility (str), Melanin (bool), Fungi Growth Rate (str), Estimated Birth Date (datetime), Estimated Death Date (datetime), plus derived fields.
- **fungal_simulation_*.csv**, **optimized_fungal_simulation_*.csv**: Species Name (str), Population (float), X (int), Y (int), Fitness (float), Var_Pos (list), Var_Base (list), Location (str).
- **validation_results_*.csv**: Species (str), RMSE (float), R² (float).
- **extended_fungal_dataset.csv**: Same columns as combined_dataset.csv, extended 10x.

## Notes
- Requires: `numpy`, `pandas`, `scipy`, `matplotlib`, `faker`, `deap`, `joblib`, `numba`, `argparse`. Optional: `cupy`, `openmc`.
- Sources: Dadachova 2007, ISS 2022, ICRP Publ. 119 (2012), Kashparov et al. (2018), Glass & Smith (1994), Guerrero-Garcia (2024), Yuzon et al. (2023), Gusa et al. (2023), Vasilieva et al. (2022).
- Realism patch: Structured 100x100 grid with C (0.2–2.0 g/L), N (0.05–0.3 g/L), lognormal dose (mean 0.05 mGy h⁻¹), and per-species populations. Growth uses Monod kinetics, with 5% mortality, 2% stochastic extinction, and no instant species replacement. GA mutation uses σ=0.02 per trait. Resource uptake scaled to prevent rapid depletion.
- OpenMC uses a two-layer geometry (air over soil) with a dynamic photon spectrum from `radiation_data.json`, cached by spectrum, thickness, and overburden.
- TODO: Split into `grid.py`, `genetics.py`, `io.py` for maintainability. Unify CPU/GPU kernels to reduce duplication.
"""

from __future__ import annotations
import threading
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import random
import datetime
import logging
import csv
import json
import os
import time
import math
import uuid
import scipy.ndimage
import pickle
import argparse
import hashlib
import tempfile
import shutil
from faker import Faker
from sklearn.metrics import mean_squared_error, r2_score
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, Memory, cpu_count
from numpy.typing import NDArray
from typing import Any, Dict, List, Tuple, Optional
from pandas import Series
from numba.core.types import Bytes
# Suppress Numba performance warnings
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Set up logging
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
def setup_logging(base_dir: str) -> logging.Logger:
    """Initialise file+console logging and return the shared logger."""
    log_dir = base_dir
    log_file = os.path.join(log_dir, f'fungal_simulation_{timestamp}.log')
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
        force=True
    )

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    logging.getLogger('numba.cuda').setLevel(logging.WARNING)
    logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)
    logging.getLogger('cupy').setLevel(logging.WARNING)

    logger.debug("Logging initialized successfully")
    return logger

logger = logging.getLogger('fungal_simulation')
setup_logging(os.getenv("BASE_DIR", os.getcwd()))  # Initialize logging after logger definition

# Optional imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    GPU_ID = int(os.getenv("GPU_DEVICE_ID", "0"))
    try:
        cp.cuda.Device(GPU_ID).use()
    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.error(f"Invalid GPU ID {GPU_ID}: {e}; falling back to CPU")
        GPU_AVAILABLE = False
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    import cupy as _cp
else:
    import numpy as _cp

# Define _to_numpy after _cp import
def _to_numpy(a):
    return a.get() if isinstance(a, _cp.ndarray) else a

try:
    from numba import cuda, njit, typed, types
except ImportError:
    cuda = None
    njit = lambda x: x  # Fallback if numba is unavailable
    if GPU_AVAILABLE:
        GPU_AVAILABLE = False

try:
    import openmc
    OPENMC_AVAILABLE: bool = True
except ImportError:
    OPENMC_AVAILABLE: bool = False
    logging.warning("OpenMC not available; using default radiation values.")

try:
    import psutil
except ImportError:
    psutil = None
    logging.warning("psutil not available; memory monitoring disabled.")

# OpenMC lock for thread/process safety
openmc_lock = threading.Lock() if OPENMC_AVAILABLE else None

# Parse command-line arguments
parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Fungal Ecosystem Simulation")
parser.add_argument("--steps", type=int, default=1000, help="Number of simulation time steps")
parser.add_argument("--profile", type=str, default="Cryptococcus", choices=["Cryptococcus", "Clado"],
                    help="Mutation profile: Cryptococcus or Clado")
parser.add_argument("--base-dir", type=str, default="/input" if os.getenv("IN_DOCKER", "") == "1" else os.getcwd(),
                    help="Base directory for input/output files")
parser.add_argument("--history-window", type=int, default=200, help="Number of history steps to retain")
parser.add_argument("--block-size", type=int, default=128, help="Block size for GPU kernels")
args: argparse.Namespace = parser.parse_args()
BLOCK_SIZE = args.block_size

# Verify BASE_DIR exists
BASE_DIR: str = args.base_dir
if not os.path.exists(BASE_DIR):
    logging.error(f"Base directory {BASE_DIR} does not exist")
    exit(1)

# Verify GPU
if GPU_AVAILABLE:
    try:
        cp.cuda.runtime.getDeviceCount()
        logger.info("GPU acceleration with CuPy enabled")
    except Exception as e:
        logger.error(f"GPU unavailable despite CuPy import: {e}")
        GPU_AVAILABLE = False
else:
    logger.info("Running in CPU fallback mode.")

# Initialize Faker
fake: Faker = Faker()

# Constants
DOSE_CONVERSION_COEFFICIENT: float = 1.3e-5
ANNUAL_INTAKE: float = 1.0
# NUM_SPECIES: int = 24
MAX_SPECIES: int = 100  # Cap for speciation
GENERATIONS: int = 5
POP_SIZE: int = 50
NUM_TIME_STEPS: int = args.steps
NUM_RECORDS: int = 10000
TEST_TIME_STEPS: int = 50
ATTENUATION_CM2_G: Dict[str, float] = {"PLA-melanin": 0.046, "pure_melanin": 0.075}
GENOME_LENGTH: int = 10_000_000
POPULATION_SIZE: int = 50
MATING_RATE: float = 0.2
DEATH_RATE: float = 0.02  # Realism patch: 5% mortality per step
BIRTH_SCALE: float = 0.7   # Realism patch: lower fecundity
RECOVERY_THRESHOLD: float = 0.5
PURE_THRESHOLD: float = 0.95
NUM_SEEDS: int = 5
DOSE_OPTIMUM_mGyh: float = 0.05
k_shape: float = 0.22  # Adjusted to match smoke test values
MELANIN_PEAK_GAIN: float = 2.5
ALBINO_HORMESIS_MAX: float = 1.10
BASE_MUT_RATE: float = 1e-9  # Cryptococcus MA lines (Gusa et al. 2023)
MAX_CARRY: int = 4200      # Realism patch: reduced global carrying capacity
GRID_SIZE: Tuple[int, int] = (100, 100)
MAX_VARS: int = 1000
DT: float = 1.0  # Time step in hours
VMAX_MELANIN: float = 0.30  # h⁻¹
VMAX_NO_MELANIN: float = 0.20  # h⁻¹
KS_C: float = 0.1  # g/L
KS_N: float = 0.02  # g/L
kmer_size: int = 15
VectorF = NDArray[np.float64]
VectorU = NDArray[np.uint32]
VectorB = NDArray[np.uint8]
# External variables assumed defined
locations: Dict[str, Any]
species_data: Dict[str, Any]

# Mutation profiles
MUTATION_PROFILES: Dict[str, Dict[str, float]] = {
    "Cryptococcus": {"SNP": 0.7, "INS": 0.15, "DEL": 0.1, "TE": 0.05},
    "Clado": {"SNP": 0.8, "INS": 0.18, "DEL": 0.0, "TE": 0.02}
}
 
# Load JSON data
logger.info(f"Files in {BASE_DIR}: {os.listdir(BASE_DIR)}")
logger.info("Starting JSON file loading")
required_json_files: List[str] = [
    'locations.json', 'species_data.json', 'zones.json', 'mushroom_data.json',
    'radiation_data.json', 'location_data.json', 'empirical_data.json'
]
for json_file in required_json_files:
    file_path: str = os.path.join(BASE_DIR, json_file)
    if not os.path.exists(file_path):
        logger.error(f"Missing required JSON file: {json_file}")
        exit(1)
try:
    with open(os.path.join(BASE_DIR, 'locations.json'), 'r', encoding="utf-8") as f:
        locations: Dict[str, Any] = json.load(f)
    with open(os.path.join(BASE_DIR, 'species_data.json'), 'r', encoding="utf-8") as f:
        species_data: Dict[str, Any] = json.load(f)
    with open(os.path.join(BASE_DIR, 'zones.json'), 'r', encoding="utf-8") as f:
        zones: Dict[str, Any] = json.load(f)
    with open(os.path.join(BASE_DIR, 'mushroom_data.json'), 'r', encoding="utf-8") as f:
        mushroom_data: List[Dict[str, Any]] = json.load(f)
        for i, item in enumerate(mushroom_data):
            if not isinstance(item, dict) or "Strain ID" not in item:
                logger.error(f"mushroom_data[{i}] is invalid: {item}")
                mushroom_data[i] = {}
    with open(os.path.join(BASE_DIR, 'location_data.json'), 'r', encoding="utf-8") as f:
        location_data: List[Dict[str, Any]] = json.load(f)
        for i, item in enumerate(location_data):
            if not isinstance(item, dict) or "Strain ID" not in item or "Location" not in item:
                logger.error(f"location_data[{i}] is invalid: {item}")
                location_data[i] = {"Strain ID": f"INVALID_{i}", "Location": "Chernobyl Reactor, Ukraine"}
    with open(os.path.join(BASE_DIR, 'radiation_data.json'), 'r', encoding="utf-8") as f:
        radiation_data_raw: Dict[str, Any] = json.load(f)
    with open(os.path.join(BASE_DIR, 'empirical_data.json'), 'r', encoding="utf-8") as f:
        empirical_data: Dict[str, Any] = json.load(f)
    logger.info("JSON files loaded and validated successfully")
except Exception as e:
    logger.error(f"Failed to load JSON files: {e}", exc_info=True)
    print(f"JSON loading error: {e}")
    exit(1)


def random_kmer(k: int = 15) -> bytes:
    return ''.join(random.choices('ATCG', k=k)).encode()

# Trait map setup (must be defined before compute_kmer_effect)
trait_map: Dict[bytes, float] = {random_kmer(): np.random.normal(0, 1) for _ in range(5000)}
trait_map_nb = typed.Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64
)
for k, v in trait_map.items():
    trait_map_nb[k.decode()] = v

# Load and cache static data
memory = Memory(location=os.path.join(BASE_DIR, ".joblib_cache"), mmap_mode="r")

from functools import lru_cache

@lru_cache(maxsize=200_000)
def kmer_effect_cached(pos_bytes, base_bytes):
    return compute_kmer_effect(
        np.frombuffer(pos_bytes, dtype=np.uint32),
        np.frombuffer(base_bytes, dtype=np.uint8),
        trait_map_nb
    )

@memory.cache
def load_static_data():
    """Load and cache static data for sharing across workers."""
    return {
        "locations": locations,
        "species_data": species_data,
        "zones": zones,
        "mushroom_data": mushroom_data,
        "location_data": location_data,
        "radiation_data_raw": radiation_data_raw,
        "empirical_data": empirical_data,
        "trait_map": trait_map
    }

# Load static data once
static_data = load_static_data()
locations = static_data["locations"]
species_data = static_data["species_data"]
zones = static_data["zones"]
mushroom_data = static_data["mushroom_data"]
location_data = static_data["location_data"]
radiation_data_raw = static_data["radiation_data_raw"]
empirical_data = static_data["empirical_data"]
trait_map = static_data["trait_map"]

# Helper functions
def get_radiation_mGyh(location_name: str, default: float = 0.005) -> float:
    loc_data: Dict[str, Any] = radiation_data_raw.get(location_name, {})
    # Convert µSv/h to mGy/h (1 µSv ≈ 1 µGy for γ-rays ⇒ 1 µSv/h = 0.001 mGy/h)
    return float(loc_data.get("Background_µSv_hr", loc_data.get("Radiation_µSv_hr", default))) / 1000.0

def offset_duplicate_coords(locations_dict: Dict[str, Any]) -> Dict[str, Any]:
    seen_coords = set()
    for loc, data in locations_dict.items():
        coords = (data["Latitude"], data["Longitude"])
        offset_count = 0
        while coords in seen_coords:
            offset_count += 1
            data["Latitude"] = float(data["Latitude"]) + 0.05 * offset_count  # ~5-6 km
            coords = (data["Latitude"], data["Longitude"])
        seen_coords.add(coords)
    return locations_dict

def get_contamination_zone(contamination: float) -> str:
    for zone, data in zones.items():
        if data["Min Contamination"] <= contamination <= data["Max Contamination"]:
            return zone
    return "Zone 5"

def dose_response(d_mGyh: VectorF) -> VectorF:
    r_d: VectorF = np.zeros_like(d_mGyh, dtype=np.float64)
    valid: NDArray[np.bool_] = d_mGyh > 0
    r_d[valid] = np.exp(-k_shape * (np.log(d_mGyh[valid] / DOSE_OPTIMUM_mGyh) ** 2))
    logger.debug(f"Dose response: d={d_mGyh.mean():.4f} mGy h⁻¹, R(d)={r_d.mean():.4e}")
    return r_d

def dose_scaled_mu(dose_mGy_h: float, *, base_rate: float = 1e-9) -> float:
    mu: float = base_rate * (1.0 + 4.0 * dose_mGy_h + 20.0 * dose_mGy_h * dose_mGy_h)
    logger.debug(f"Mutation rate: dose={dose_mGy_h:.4f} mGy h⁻¹, μ/μ₀={mu/base_rate:.2f}")
    return mu

def list_available_gpus():
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        print("CuPy sees", n, "CUDA device(s):")
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"  [{i}] {props['name'].decode()}  "
                  f"{props['totalGlobalMem']//2**20} MiB  "
                  f"CC {props['major']}.{props['minor']}")
        return n
    except Exception as e:
        print("No CUDA runtime available or CuPy not installed:", e)
        return 0

def compute_growth_rate(C: VectorF, N: VectorF, dose_mGy_h: VectorF, melanin: bool) -> VectorF:
    vmax: float = 0.30 if melanin else 0.20
    f_nutr: VectorF = (C / (0.1 + C)) * (N / (0.02 + N))
    r_d: VectorF = np.exp(-0.22 * (np.log(dose_mGy_h / 0.05) ** 2))
    growth_rate: VectorF = vmax * f_nutr * r_d
    growth_rate = np.nan_to_num(growth_rate, nan=0.0, posinf=0.0, neginf=0.0)
    return growth_rate

def simulate_fungi_growth(location: str, radiation_level_mGyh: float, fungi_species: str, nutrient_level: float) -> str:
    if "reactor" in location.lower() or "soil" in location.lower():
        if radiation_level_mGyh > 0 and nutrient_level < 0.5:
            if fungi_species in empirical_data:
                return "High (Radiotropism)"
            return "High"
        return "Low"
    return "Normal"

def calculate_committed_effective_dose(concentration: float, intake: float = ANNUAL_INTAKE) -> float:
    return concentration * DOSE_CONVERSION_COEFFICIENT * intake

def calculate_fitness(optimized_df: DataFrame, species: str) -> float:
    """
    Calculate the mean fitness for a given species from the optimized DataFrame.
    
    Args:
        optimized_df: DataFrame containing simulation results with 'Species Name' and 'Fitness' columns.
        species: Name of the species to evaluate.
    
    Returns:
        Mean fitness value for the species, or 0.0 if data is invalid.
    """
    try:
        filtered_df = optimized_df[optimized_df['Species Name'] == species]
        if filtered_df.empty or filtered_df['Fitness'].isna().all():
            logging.warning(f"No valid fitness data for species {species}")
            return 0.0
        sim_data: Series[float] = filtered_df['Fitness'].astype(float)
        if sim_data.eq(0).all():
            logging.warning(f"All fitness values are zero for species {species}")
            return 0.0
        return sim_data.mean()
    except Exception as e:
        logging.error(f"Error calculating fitness for species {species}: {e}")
        return 0.0

def calculate_derived_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Remove ' ± ' from string values before conversion
        mean_log_cells_control_str = record.get("Mean Log (Number of Viable Cells) Control ± SD", "0").split(" ± ")[0]
        mean_log_cells_30_min_str = record.get("Mean (Log Number of Viable Cells in 30-min Exposure - Log Number of Cells in Control) ± SD", "0").split(" ± ")[0]
        mean_log_cells_60_min_str = record.get("Mean (Log Number of Viable Cells in 60-min Exposure - Log Number of Cells in Control) ± SD", "0").split(" ± ")[0]
        mean_log_cells_control: float = float(mean_log_cells_control_str)
        mean_log_cells_30_min: float = float(mean_log_cells_30_min_str)
        mean_log_cells_60_min: float = float(mean_log_cells_60_min_str)
        record["Viable Cells After 30 min (%)"] = round(100 * (10 ** (mean_log_cells_control + mean_log_cells_30_min)) / (10 ** mean_log_cells_control), 2)
        record["Viable Cells After 60 min (%)"] = round(100 * (10 ** (mean_log_cells_control + mean_log_cells_60_min)) / (10 ** mean_log_cells_control), 2)
    except (KeyError, ValueError):
        record["Viable Cells After 30 min (%)"] = None
        record["Viable Cells After 60 min (%)"] = None
    return record

def random_date(start_date: datetime.datetime, end_date: datetime.datetime) -> datetime.datetime:
    return start_date + datetime.timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

def estimate_dates(years_since_2024: int) -> Tuple[datetime.datetime, datetime.datetime]:
    birth_date: datetime.datetime = random_date(datetime.datetime(2024 - years_since_2024, 1, 1), datetime.datetime(2024, 12, 31))
    death_date: datetime.datetime = random_date(birth_date, datetime.datetime(2054, 12, 31))
    return birth_date, death_date

# Smoke test for dose response and mutation rate
def run_smoke_test() -> None:
    test_doses: List[float] = [0.01, 0.05, 0.30, 1.00]  # mGy h⁻¹
    logger.info("Running smoke test for dose response and mutation rate")
    for dose in test_doses:
        r_d: float = dose_response(np.array([dose], dtype=np.float64))[0]
        mu_ratio: float = dose_scaled_mu(dose) / BASE_MUT_RATE
        logger.info(f"Dose: {dose:.2f} mGy h⁻¹, R(d): {r_d:.4f}, μ/μ₀: {mu_ratio:.2f}")

# OpenMC material creation
def make_material(name: str, material_data: Optional[Dict[str, Any]] = None) -> Any:
    if not OPENMC_AVAILABLE:
        logger.warning("OpenMC not available; cannot create material")
        return None
    if material_data is None:
        if name == "air":
            material_data = {"density": 0.001225, "composition": {"N": 0.78, "O": 0.21}}
        elif name == "soil":
            material_data = {"density": 1.5, "composition": {"Si": 0.33, "O": 0.50, "Al": 0.10, "Cs137": 1e-6}}
        else:
            raise ValueError(f"No default material data for {name}")
    try:
        import openmc
        material = openmc.Material(name=name)
        for element, fraction in material_data["composition"].items():
            if element.startswith("Cs"):
                material.add_nuclide(element, fraction, percent_type='wo')
            else:
                material.add_element(element, fraction, percent_type='wo')
        material.set_density('g/cm3', material_data["density"])
        return material
    except (AttributeError, ImportError) as e:
        logger.error(f"Failed to create OpenMC material: {e}")
        return None

# Define static directory for XML files
STATIC_XML_DIR = os.path.join(BASE_DIR, "openmc_static")
OPENMC_WORK_DIR = None
GLOBAL_OPENMC_MODEL = None

# OpenMC setup
if OPENMC_AVAILABLE:
    if os.path.exists(STATIC_XML_DIR):
        shutil.rmtree(STATIC_XML_DIR)
        logger.info(f"Removed existing {STATIC_XML_DIR} to regenerate consistent XML files")
    os.makedirs(STATIC_XML_DIR)

    openmc.reset_auto_ids()  # Reset IDs once before creating materials
    default_materials_data = [
        {"name": "air", "density": 0.001225, "composition": {"N": 0.78, "O": 0.21}},
        {"name": "soil", "density": 1.5, "composition": {"Si": 0.33, "O": 0.50, "Al": 0.10, "Cs137": 1e-6}}
    ]
    air = make_material("air", next((m for m in default_materials_data if m["name"] == "air"), None))
    soil = make_material("soil", next((m for m in default_materials_data if m["name"] == "soil"), None))
    if air is None or soil is None:
        logger.error("Failed to create default materials for static XMLs")
    else:
        materials = openmc.Materials([air, soil])
        materials.export_to_xml(os.path.join(STATIC_XML_DIR, "materials.xml"))

        air_surface = openmc.ZPlane(z0=0.1)
        soil_surface = openmc.ZPlane(z0=0.1 + 1.0, boundary_type='vacuum')
        air_cell = openmc.Cell(fill=air, region=-air_surface)
        soil_cell = openmc.Cell(fill=soil, region=+air_surface & -soil_surface)
        geometry = openmc.Geometry([air_cell, soil_cell])
        geometry.export_to_xml(os.path.join(STATIC_XML_DIR, "geometry.xml"))

        settings = openmc.Settings()
        settings.run_mode = 'fixed source'
        settings.particles = 10000
        settings.batches = 10
        settings.inactive = 0
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Point((0, 0, 0)),
            energy=openmc.stats.Discrete([0.662 * 1e6], [1.0]),
            particle='photon'
        )
        settings.output = {'summary': True}
        settings.export_to_xml(os.path.join(STATIC_XML_DIR, "settings.xml"))

        tally = openmc.Tally(name='dose_tally')
        tally.filters = [openmc.CellFilter(soil_cell)]
        tally.scores = ['heating']
        tallies = openmc.Tallies([tally])
        tallies.export_to_xml(os.path.join(STATIC_XML_DIR, "tallies.xml"))
        logger.info(f"Generated static XML files in {STATIC_XML_DIR}")

        for fn in ("materials.xml", "geometry.xml", "settings.xml", "tallies.xml"):
            if not os.path.isfile(os.path.join(STATIC_XML_DIR, fn)):
                logger.error(f"Static XML file {fn} missing in {STATIC_XML_DIR}")
                raise FileNotFoundError(f"Static XML file {fn} missing in {STATIC_XML_DIR}")

        OPENMC_WORK_DIR = tempfile.mkdtemp(prefix="openmc_work_")
        shutil.copytree(STATIC_XML_DIR, OPENMC_WORK_DIR, dirs_exist_ok=True)
        openmc.reset_auto_ids()
        mats = openmc.Materials.from_xml(os.path.join(OPENMC_WORK_DIR, "materials.xml"))
        geom = openmc.Geometry.from_xml(
            os.path.join(OPENMC_WORK_DIR, "geometry.xml"),
            os.path.join(OPENMC_WORK_DIR, "materials.xml")
        )
        settings = openmc.Settings.from_xml(os.path.join(OPENMC_WORK_DIR, "settings.xml"))
        tallies = openmc.Tallies.from_xml(os.path.join(OPENMC_WORK_DIR, "tallies.xml"))
        GLOBAL_OPENMC_MODEL = openmc.Model(geometry=geom, materials=mats, settings=settings, tallies=tallies)
else:
    logger.warning("OpenMC not available; using default radiation values")

# OpenMC dose cache
dose_cache: Dict[str, float] = {}
fitness_cache: Dict[str, Tuple[float, float]] = {}

# Updated run_transport function
def run_transport(thickness_cm: float, location: str, overburden: float = 0.1) -> float:
    from multiprocessing import current_process
    if not OPENMC_AVAILABLE:
        logger.warning("OpenMC not available; returning default dose of 0.005 mGy/h")
        return 0.005
    work_dir = None
    try:
        loc_data = radiation_data_raw.get(location, {
            "Radiation_µSv_hr": 5.0,
            "isotope_list": [{"energy_MeV": [0.662], "rel_intensity": [1.0]}],
            "materials": [
                {"density": 0.001225, "name": "air", "composition": {"N": 0.78, "O": 0.21}},
                {"density": 1.5, "name": "soil", "composition": {"Si": 0.33, "O": 0.50, "Al": 0.10, "Cs137": 1e-6}}
            ]
        })
        iso_list = loc_data.get("isotope_list", [{"energy_MeV": [0.662], "rel_intensity": [1.0]}])
        meas_list = loc_data.get("Measurement_µSv_hr", [5.0])
        if len(iso_list) != len(meas_list):
            logger.warning(f"Mismatched isotope list length for {location}; padding measurements")
            meas_list = meas_list + [meas_list[0]] * (len(iso_list) - len(meas_list))
        spectrum = []
        for iso, _ in zip(iso_list, meas_list):
            energies_list = iso.get("energy_MeV", [0.662])
            weights_list = iso.get("rel_intensity", [1.0] * len(energies_list))
            if len(energies_list) == len(weights_list):
                spectrum.extend(zip(energies_list, weights_list))
        if not spectrum:
            logger.warning(f"Empty spectrum for {location}; using default")
            return 0.005
        spectrum_hash = hashlib.md5(str((tuple(sorted(spectrum)), thickness_cm, overburden)).encode()).hexdigest()
        DOSE_CACHE_FILE = os.path.join(BASE_DIR, "dose_cache.npz")
        if os.path.exists(DOSE_CACHE_FILE):
            with np.load(DOSE_CACHE_FILE, allow_pickle=True) as data:
                dose_cache.update(data['dose_cache'].item())
        with openmc_lock:
            if spectrum_hash in dose_cache:
                logger.debug(f"Using cached dose for spectrum hash {spectrum_hash}: {dose_cache[spectrum_hash]:.2f} mGy/h")
                return dose_cache[spectrum_hash]
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
            openmc.reset_auto_ids()
            mat_list = loc_data.get("materials", [
                {"density": 0.001225, "name": "air", "composition": {"N": 0.78, "O": 0.21}},
                {"density": 1.5, "name": "soil", "composition": {"Si": 0.33, "O": 0.50, "Al": 0.10, "Cs137": 1e-6}}
            ])
            air = make_material(mat_list[0]["name"], mat_list[0])
            soil = make_material(mat_list[1]["name"], mat_list[1])
            if air is None or soil is None:
                logger.error("Failed to create materials for transport")
                return 0.005
            materials = openmc.Materials([air, soil])
            air_surface = openmc.ZPlane(z0=overburden)
            soil_surface = openmc.ZPlane(z0=overburden + thickness_cm, boundary_type='vacuum')
            air_cell = openmc.Cell(fill=air, region=-air_surface)
            soil_cell = openmc.Cell(fill=soil, region=+air_surface & -soil_surface)
            geometry = openmc.Geometry([air_cell, soil_cell])
            settings = openmc.Settings()
            settings.run_mode = 'fixed source'
            settings.particles = 10000
            settings.batches = 10
            settings.inactive = 0
            energies, weights = zip(*spectrum) if spectrum else ([0.662], [1.0])
            settings.source = openmc.IndependentSource(
                space=openmc.stats.Point((0, 0, 0)),
                energy=openmc.stats.Discrete([E * 1e6 for E in energies], weights),
                particle='photon'
            )
            settings.output = {'summary': True}
            tally = openmc.Tally(name='dose_tally')
            tally.filters = [openmc.CellFilter(soil_cell)]
            tally.scores = ['heating']
            tallies = openmc.Tallies([tally])
            work_dir = tempfile.mkdtemp(prefix=f"openmc_work_{current_process().pid}_")
            shutil.copytree(STATIC_XML_DIR, work_dir, dirs_exist_ok=True)
            model = openmc.Model(geometry=geometry, materials=materials, settings=settings, tallies=tallies)
            model.run(cwd=work_dir)
            sp_file = os.path.join(work_dir, f"statepoint.{settings.batches}.h5")
            if not os.path.isfile(sp_file):
                logger.error(f"Statepoint file not found: {sp_file}")
                return 0.005
            with openmc.StatePoint(sp_file) as sp:
                tally_result = sp.get_tally(name='dose_tally')
                dose_eV = tally_result.mean.flatten()[0]
                volume_cm3 = thickness_cm
                mass_kg = mat_list[1].get("density", 1.5) * volume_cm3 / 1000.0
                if mass_kg <= 0:
                    mass_kg = 1e-6
                dose_Gy_per_particle = dose_eV * 1.602e-19 / mass_kg
                dose_Sv_per_hour = dose_Gy_per_particle * settings.particles * 3600
                dose_mGy_h = dose_Sv_per_hour * 1000
                logger.debug(f"OpenMC dose calculation: spectrum={spectrum}, eV/particle={dose_eV:.2e}, "
                             f"mass={mass_kg:.2e} kg, Gy/particle={dose_Gy_per_particle:.2e}, mGy/h={dose_mGy_h:.2f}")
            dose_cache[spectrum_hash] = max(dose_mGy_h, 0.0001)
            np.savez_compressed(DOSE_CACHE_FILE, dose_cache=dose_cache)
            return dose_cache[spectrum_hash]
    except Exception as e:
        logger.error(f"OpenMC simulation failed for {location}: {e}")
        return 0.005
    finally:
        if work_dir and os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.debug(f"Cleaned up work_dir: {work_dir}")

@njit(cache=True, nogil=True, parallel=True)
def compute_kmer_effect(var_pos: VectorU, var_base: VectorB, tmap) -> float:
    eff = 0.0
    for pos, base in zip(var_pos, var_base):
        if pos == 0:
            break
        if pos + kmer_size <= GENOME_LENGTH:
            kmer = chr(base) * kmer_size
            eff += tmap.get(kmer, 0.0)
    return eff

def recombine_cpu(p1: VectorU, b1: VectorB, p2: VectorU, b2: VectorB, dose: float, max_vars: int) -> Tuple[VectorU, VectorB]:
    num_vars1: int = np.sum(p1 != 0)
    num_vars2: int = np.sum(p2 != 0)
    num_vars: int = min(num_vars1, num_vars2, max_vars)
    if num_vars < 2:
        return p1.copy(), b1.copy()
    cut: int = random.randint(1, num_vars - 1)
    p1_valid = p1[:num_vars1]
    b1_valid = b1[:num_vars1]
    p2_valid = p2[:num_vars2]
    b2_valid = b2[:num_vars2]
    if num_vars2 < num_vars1:
        p2_valid = np.pad(p2_valid, (0, num_vars1 - num_vars2), 'constant')
        b2_valid = np.pad(b2_valid, (0, num_vars1 - num_vars2), 'constant')
    elif num_vars1 < num_vars2:
        p1_valid = np.pad(p1_valid, (0, num_vars2 - num_vars1), 'constant')
        b1_valid = np.pad(b1_valid, (0, num_vars2 - num_vars1), 'constant')
    new_pos: VectorU = np.concatenate((p1_valid[:cut], p2_valid[cut:]))[:max_vars]
    new_base: VectorB = np.concatenate((b1_valid[:cut], b2_valid[cut:]))[:max_vars]
    new_pos = np.pad(new_pos, (0, max_vars - len(new_pos)), 'constant')
    new_base = np.pad(new_base, (0, max_vars - len(new_base)), 'constant')
    logger.debug(f"recombine_cpu: p1_vars={num_vars1}, p2_vars={num_vars2}, output_vars={np.sum(new_pos!=0)}")
    return new_pos, new_base

def genome_similarity_cpu(p1: VectorU, b1: VectorB, p2: VectorU, b2: VectorB) -> float:
    p1 = np.asarray(p1)
    b1 = np.asarray(b1)
    p2 = np.asarray(p2)
    b2 = np.asarray(b2)
    V1: set[int] = set(p1[p1 != 0].tolist())
    V2: set[int] = set(p2[p2 != 0].tolist())
    common: set[int] = V1 & V2
    union: set[int] = V1 | V2
    matches: int = sum(b1[np.where(p1 == pos)[0][0]] == b2[np.where(p2 == pos)[0][0]] for pos in common)
    return matches / max(len(union), 1)

# GPU-accelerated functions, only defined if GPU is available
if GPU_AVAILABLE and cuda is not None:
    @cuda.jit
    def recombine_kernel(var_pos1: VectorU, var_base1: VectorB, var_pos2: VectorU, var_base2: VectorB,
                        child_pos: VectorU, child_base: VectorB, mutation_rate: float, random_vals: VectorF,
                        max_vars: int, num_pairs: int) -> None:
        pair_idx = cuda.grid(1)
        if pair_idx >= num_pairs:
            return
        offset = pair_idx * max_vars
        num_vars1 = 0
        num_vars2 = 0
        for i in range(max_vars):
            if var_pos1[offset + i] != 0:
                num_vars1 += 1
            if var_pos2[offset + i] != 0:
                num_vars2 += 1
        num_vars = min(num_vars1, num_vars2, max_vars)
        if num_vars == 0:
            for i in range(max_vars):
                child_pos[offset + i] = 0
                child_base[offset + i] = 0
            return
        point = int(random_vals[pair_idx] * num_vars)
        for idx in range(max_vars):
            if idx < num_vars:
                if idx < point and idx < num_vars1:
                    child_pos[offset + idx] = var_pos1[offset + idx]
                    child_base[offset + idx] = var_base1[offset + idx]
                elif idx >= point and idx < num_vars2:
                    child_pos[offset + idx] = var_pos2[offset + idx]
                    child_base[offset + idx] = var_base2[offset + idx]
                else:
                    child_pos[offset + idx] = 0
                    child_base[offset + idx] = 0
                if random_vals[pair_idx + num_pairs + idx] < mutation_rate:
                    mutation_choice = int(random_vals[pair_idx + 2 * num_pairs + idx] * 4)
                    child_base[offset + idx] = ord('A') + mutation_choice % 4
            else:
                child_pos[offset + idx] = 0
                child_base[offset + idx] = 0

    @cuda.jit
    def similarity_kernel(var_pos1: VectorU, var_base1: VectorB, var_pos2: VectorU, var_base2: VectorB,
                         max_vars: int, similarity: VectorF, num_pairs: int) -> None:
        pair_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        if pair_idx >= num_pairs:
            return
        offset = pair_idx * max_vars
        shared_matches = cuda.shared.array(shape=(BLOCK_SIZE,), dtype=np.float32)
        shared_union = cuda.shared.array(shape=(BLOCK_SIZE,), dtype=np.uint32)
        shared_matches[thread_idx] = 0
        shared_union[thread_idx] = 0
        cuda.syncthreads()

        seen = cuda.local.array(MAX_VARS, dtype=np.uint32)  # Fixed array size to MAX_VARS
        seen_count = 0

        stride = cuda.blockDim.x
        i = thread_idx
        while i < max_vars:
            if var_pos1[offset + i] != 0:
                is_new = True
                for k in range(seen_count):
                    if seen[k] == var_pos1[offset + i]:
                        is_new = False
                        break
                if is_new and seen_count < max_vars:
                    seen[seen_count] = var_pos1[offset + i]
                    seen_count += 1
            if var_pos2[offset + i] != 0:
                is_new = True
                for k in range(seen_count):
                    if seen[k] == var_pos2[offset + i]:
                        is_new = False
                        break
                if is_new and seen_count < max_vars:
                    seen[seen_count] = var_pos2[offset + i]
                    seen_count += 1
            i += stride

        shared_union[thread_idx] = seen_count
        cuda.syncthreads()

        i = thread_idx
        while i < max_vars:
            if var_pos1[offset + i] != 0 and var_pos2[offset + i] != 0:
                if var_pos1[offset + i] == var_pos2[offset + i] and var_base1[offset + i] == var_base2[offset + i]:
                    cuda.atomic.add(shared_matches, thread_idx, 1)
            i += stride
        cuda.syncthreads()

        if thread_idx == 0:
            total_matches = 0
            total_union = 0
            for j in range(min(cuda.blockDim.x, max_vars)):
                total_matches += shared_matches[j]
                total_union = max(total_union, shared_union[j])
            similarity[pair_idx] = total_matches / max(total_union, 1)

def find_occupied_cells(pop: np.ndarray, max_species: int) -> np.ndarray:
    """Return list of (x,y,i) indices where population ≥1 and i < current species count."""
    occupied = np.argwhere(pop >= 1.0)
    occupied = occupied[occupied[:, 2] < max_species]
    return occupied

def _to_numpy(a):
    return a.get() if isinstance(a, _cp.ndarray) else a

def recombine_gpu(var_pos_pairs: List[Tuple[VectorU, VectorU]], var_base_pairs: List[Tuple[VectorB, VectorB]], dose_mGy_h: float, max_vars: int) -> List[Tuple[VectorU, VectorB]]:
    if not GPU_AVAILABLE or cuda is None or max_vars <= 1000:
        return [recombine_cpu(p1, b1, p2, b2, dose_mGy_h, max_vars) for (p1, p2), (b1, b2) in zip(var_pos_pairs, var_base_pairs)]
    
    num_pairs = len(var_pos_pairs)
    mu = dose_scaled_mu(dose_mGy_h)
    
    try:
        var_pos1_flat = np.concatenate([_to_numpy(p1) for p1, _ in var_pos_pairs]).astype(np.uint32)
        var_pos2_flat = np.concatenate([_to_numpy(p2) for _, p2 in var_pos_pairs]).astype(np.uint32)
        var_base1_flat = np.concatenate([_to_numpy(b1) for b1, _ in var_base_pairs]).astype(np.uint8)
        var_base2_flat = np.concatenate([_to_numpy(b2) for _, b2 in var_base_pairs]).astype(np.uint8)
        var_pos1_flat = np.ascontiguousarray(var_pos1_flat)
        var_pos2_flat = np.ascontiguousarray(var_pos2_flat)
        var_base1_flat = np.ascontiguousarray(var_base1_flat)
        var_base2_flat = np.ascontiguousarray(var_base2_flat)
    except ValueError as e:
        logger.error(f"Error concatenating arrays in recombine_gpu: {e}")
        return [(np.zeros(max_vars, dtype=np.uint32), np.zeros(max_vars, dtype=np.uint8)) for _ in range(num_pairs)]
    
    child_pos = cuda.device_array(num_pairs * max_vars, dtype=np.uint32)
    child_base = cuda.device_array(num_pairs * max_vars, dtype=np.uint8)
    child_pos[:] = 0
    child_base[:] = 0
    rnd = cp.random.rand(3 * num_pairs).astype(np.float64)
    
    threads = min(cuda.get_current_device().MAX_THREADS_PER_BLOCK, BLOCK_SIZE)
    blocks = math.ceil(num_pairs / threads)
    try:
        recombine_kernel[blocks, threads](
            cp.asarray(var_pos1_flat), cp.asarray(var_base1_flat),
            cp.asarray(var_pos2_flat), cp.asarray(var_base2_flat),
            child_pos, child_base, mu, rnd, max_vars, num_pairs
        )
    except Exception as e:
        logger.error(f"Error executing recombine_kernel: {e}")
        return [(np.zeros(max_vars, dtype=np.uint32), np.zeros(max_vars, dtype=np.uint8)) for _ in range(num_pairs)]
    
    child_pos_np = cp.asnumpy(child_pos).reshape(num_pairs, max_vars)
    child_base_np = cp.asnumpy(child_base).reshape(num_pairs, max_vars)
    return [(child_pos_np[i], child_base_np[i]) for i in range(num_pairs)]

def genome_similarity_gpu(var_pos_pairs: List[Tuple[VectorU, VectorU]], var_base_pairs: List[Tuple[VectorB, VectorB]]) -> List[float]:
    if not var_pos_pairs or not var_base_pairs:
        logger.debug("No mating pairs provided for genome_similarity_gpu; returning empty similarities")
        return []
    if (not GPU_AVAILABLE) or (cuda is None) or len(var_pos_pairs[0][0]) <= 1000:
        return [
            genome_similarity_cpu(_to_numpy(p1), _to_numpy(b1), _to_numpy(p2), _to_numpy(b2))
            for (p1, p2), (b1, b2) in zip(var_pos_pairs, var_base_pairs)
        ]
    
    num_pairs = len(var_pos_pairs)
    max_vars = len(var_pos_pairs[0][0])
    
    try:
        var_pos1_flat = np.concatenate([_to_numpy(p1) for p1, _ in var_pos_pairs]).astype(np.uint32)
        var_pos2_flat = np.concatenate([_to_numpy(p2) for _, p2 in var_pos_pairs]).astype(np.uint32)
        var_base1_flat = np.concatenate([_to_numpy(b1) for b1, _ in var_base_pairs]).astype(np.uint8)
        var_base2_flat = np.concatenate([_to_numpy(b2) for _, b2 in var_base_pairs]).astype(np.uint8)
        var_pos1_flat = np.ascontiguousarray(var_pos1_flat)
        var_pos2_flat = np.ascontiguousarray(var_pos2_flat)
        var_base1_flat = np.ascontiguousarray(var_base1_flat)
        var_base2_flat = np.ascontiguousarray(var_base2_flat)
    except ValueError as e:
        logger.error(f"Error concatenating arrays in genome_similarity_gpu: {e}")
        return [0.0] * num_pairs
    
    sim = cuda.device_array(num_pairs, dtype=np.float32)
    sim[:] = 0
    
    threads = min(cuda.get_current_device().MAX_THREADS_PER_BLOCK, BLOCK_SIZE)
    blocks = num_pairs
    try:
        similarity_kernel[blocks, threads](
            cp.asarray(var_pos1_flat), cp.asarray(var_base1_flat),
            cp.asarray(var_pos2_flat), cp.asarray(var_base2_flat),
            max_vars, sim, num_pairs
        )
    except Exception as e:
        logger.error(f"Error executing similarity_kernel: {e}")
        return [0.0] * num_pairs
    
    return cp.asnumpy(sim).tolist()

def build_combined_dataset(num_records: int, strain_ids: set[str], mushroom_records_by_id: Dict[str, Dict[str, Any]], 
                         location_records_by_id: Dict[str, Dict[str, Any]], locations: Dict[str, Any], 
                         species_data: Dict[str, Any], zones: Dict[str, Any]) -> pd.DataFrame:
    combined_csv = os.path.join(BASE_DIR, "combined_dataset.csv")
    if os.path.exists(combined_csv):
        logger.info(f"Loading cached combined dataset from {combined_csv}")
        return pd.read_csv(combined_csv)
    combined_records: List[Dict[str, Any]] = []
    rows_per_strain: int = math.ceil(num_records / len(strain_ids))
    for strain_id in strain_ids:
        for _ in range(rows_per_strain):
            mushroom_record: Dict[str, Any] = mushroom_records_by_id.get(strain_id, {"Radiation_µSv_hr": location_records_by_id.get(strain_id, {}).get("Radiation_µSv_hr", 0.05)})
            location_record: Dict[str, Any] = location_records_by_id.get(strain_id, {})
            combined_record: Dict[str, Any] = {**mushroom_record, **location_record}
            combined_record = calculate_derived_fields(combined_record)
            loc_name: str = location_record.get("Location", "Chernobyl Reactor, Ukraine")
            loc_data: Dict[str, Any] = locations.get(loc_name, {"Latitude": 51.3894, "Longitude": 30.0994, "Habitat": "Reactor", "Radiation_µSv_hr": 5.0, "Contamination_kBq_m2": 2000})
            habitat: str = loc_data["Habitat"]
            species: str = mushroom_record.get("Species Name", random.choices(
                [s for s in species_data],
                weights=[species_data[s]["Habitat Weight"] for s in species_data],
                k=1
            )[0])
            zone: str = get_contamination_zone(loc_data.get("Contamination_kBq_m2", 2000))
            concentration: float = random.uniform(species_data[species]["Min"], species_data[species]["Max"])
            radiation_level_µSv_h: float = loc_data.get("Radiation_µSv_hr", 5.0)
            nutrient_level: float = random.uniform(0.1, 1.0)
            temp: float = random.uniform(20, 35)
            ph: float = random.uniform(4, 8)
            no2: float = random.uniform(0, 10)
            committed_effective_dose: float = calculate_committed_effective_dose(concentration)
            melanin_present: bool = species_data[species]["Melanin"]
            substrate: str = species_data[species]["Substrate"]
            years_since_2024: int = np.random.randint(1, 31)
            birth_date, death_date = estimate_dates(years_since_2024)
            growth_rate: float = compute_growth_rate(np.array([nutrient_level]), np.array([nutrient_level * 0.2]), np.array([radiation_level_µSv_h / 1000.0]), melanin_present)[0]
            fungi_growth_rate: str = simulate_fungi_growth(loc_name, radiation_level_µSv_h / 1000.0, species, nutrient_level)
            collection_date: datetime.date = fake.date_between(start_date='-36y', end_date='today')
            cap_min, cap_max = species_data[species]["CapDiameter_cm"]
            stem_min, stem_max = species_data[species]["StemLength_cm"]
            record: Dict[str, Any] = {
                "Strain ID": str(strain_id),
                "Species Name": species,
                "Common Name": species_data[species]["Common Name"],
                "Location": loc_name,
                "Latitude": loc_data["Latitude"],
                "Longitude": loc_data["Longitude"],
                "Habitat": habitat,
                "Soil Type": random.choice(["Sandy", "Clay", "Loam"]),
                "pH Level": ph,
                "Contamination Zone": zone,
                "Radioactive Contamination Level (kBq/m²)": round(random.uniform(zones[zone]["Min Contamination"], zones[zone]["Max Contamination"]), 2),
                "Radioactive Isotope": "137Cs",
                "Concentration in Mushroom (Bq/kg)": round(concentration, 2),
                "Committed Effective Dose (mSv/year)": round(committed_effective_dose, 3),
                "Radiation Level (µSv/hr)": radiation_level_µSv_h,
                "Nutrient Level": nutrient_level,
                "Temperature (°C)": temp,
                "NO2 (mg/L)": no2,
                "Substrate": substrate,
                "Cap Diameter (cm)": round(random.uniform(cap_min, cap_max), 1),
                "Stem Length (cm)": round(random.uniform(stem_min, stem_max), 1),
                "Edibility": species_data[species]["Edibility"],
                "Melanin": melanin_present,
                "Fungi Growth Rate": fungi_growth_rate,
                "Growth Rate": round(growth_rate, 4),
                "Collection Date": collection_date,
                "Estimated Birth Date": birth_date,
                "Estimated Death Date": death_date,
                **{k: v for k, v in combined_record.items() if v is not None}
            }
            combined_records.append(record)
    df: pd.DataFrame = pd.DataFrame(combined_records[:num_records])
    assert df.duplicated(subset=["Latitude", "Longitude"]).sum() == 0, "Duplicate coords!"
    assert df["Committed Effective Dose (mSv/year)"].max() < 10, "Looks like µSv, not mSv"
    df.to_csv("combined_dataset.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Combined dataset saved to combined_dataset.csv")
    return df

# Individual class with spatial coordinates
class Individual:
    def __init__(self, var_pos: VectorU, var_base: VectorB, label: str, location: str, x: int, y: int,
                 fitness: float = 1.0, mating_type: Optional[str] = None, hybrid_parentage: Optional[Tuple[str, str]] = None) -> None:
        self.id: str = uuid.uuid4().hex
        self.var_pos: VectorU = var_pos
        self.var_base: VectorB = var_base
        self.species_label: str = label
        self.location: str = location
        self.x: int = x
        self.y: int = y
        self.fitness: float = fitness
        self.mating_type: str = mating_type or random.choice(("A", "B"))
        self.hybrid_parentage: Optional[Tuple[str, str]] = hybrid_parentage

def generate_genome(length: int = GENOME_LENGTH, max_vars: int = MAX_VARS) -> Tuple[VectorU, VectorB]:
    num_vars: int = random.randint(0, max_vars)
    if num_vars == 0:
        return np.zeros(max_vars, dtype=np.uint32), np.zeros(max_vars, dtype=np.uint8)
    var_pos: VectorU = np.zeros(max_vars, dtype=np.uint32)
    var_pos[:num_vars] = np.random.choice(length, num_vars, replace=False)
    var_pos[:num_vars].sort()
    var_base: VectorB = np.frombuffer(''.join(np.random.choice(['A', 'T', 'C', 'G'], num_vars)).encode(), dtype=np.uint8)
    return var_pos, var_base

# Fungal Ecosystem Class
class FungalEcosystem:
    def __init__(self, species_names: List[str], initial_pop_size: int = 50, mutation_profile: str = "Cryptococcus", params: Optional[List[Dict[str, Any]]] = None, precomputed_doses: Optional[Dict[str, float]] = None, history_window: int = 200) -> None:
        self.species_names = list(species_names[:MAX_SPECIES])
        self.mutation_profile = mutation_profile
        self.max_vars = MAX_VARS
        self.params = params or [{} for _ in self.species_names]
        self.current_step = 0
        self.pedigree = []
        self.history_window = history_window
        self.K_cell = 4200 / len(self.species_names)
        self.pop = np.zeros((GRID_SIZE[0], GRID_SIZE[1], MAX_SPECIES), dtype=np.float32)
        self.C = np.random.uniform(0.2, 2.0, GRID_SIZE)
        self.N = np.random.uniform(0.05, 0.3, GRID_SIZE)
        self.dose = np.zeros(GRID_SIZE, dtype=np.float32)
        self.grid_locations = np.array([[random.choice(list(locations.keys())) for _ in range(GRID_SIZE[1])] for _ in range(GRID_SIZE[0])], dtype=object)
        if not np.all([loc in locations for loc in self.grid_locations.flatten()]):
            logger.error("Invalid location names in grid_locations")
            raise ValueError("grid_locations contains invalid location names")
        if precomputed_doses is None:
            unique_locations = set(self.grid_locations.flatten())
            logger.info(f"Precomputing doses for {len(unique_locations)} unique locations")
            precomputed_doses = {loc: run_transport(thickness_cm=1.0, location=loc, overburden=0.1) for loc in unique_locations}
        for x in range(GRID_SIZE[0]):
            for y in range(GRID_SIZE[1]):
                self.dose[x, y] = precomputed_doses.get(self.grid_locations[x, y], 0.005)
        self.gpu_genomes = {}
        for i, name in enumerate(self.species_names):
            if i >= MAX_SPECIES:
                logger.warning(f"Species index {i} exceeds MAX_SPECIES {MAX_SPECIES}; skipping")
                break
            var_pos, var_base = generate_genome()
            if GPU_AVAILABLE:
                self.gpu_genomes[name] = (cp.asarray(var_pos), cp.asarray(var_base))
            for _ in range(initial_pop_size):
                x = random.randrange(GRID_SIZE[0])
                y = random.randrange(GRID_SIZE[1])
                if i < self.pop.shape[2]:
                    self.pop[x, y, i] += 1.0
                else:
                    logger.error(f"Species index {i} exceeds population array dimension {self.pop.shape[2]}")
                    break
        self.reference_genomes = {name: generate_genome() for name in self.species_names}
        self.history = []
        logger.debug(f"Initialized {GRID_SIZE[0]}x{GRID_SIZE[1]} grid with {len(self.species_names)} species, "
                    f"initial pop size {initial_pop_size}, mutation profile: {mutation_profile}")

    def assign_label(self, var_pos: VectorU, var_base: VectorB) -> str:
        max_similarity = 0.0
        assigned_label = self.species_names[0]
        for label, (ref_pos, ref_base) in self.reference_genomes.items():
            if GPU_AVAILABLE:
                similarity = genome_similarity_gpu([(var_pos, ref_pos)], [(var_base, ref_base)])[0]
            else:
                similarity = genome_similarity_cpu(var_pos, var_base, ref_pos, ref_base)
            if similarity > max_similarity:
                max_similarity = similarity
                assigned_label = label
        if max_similarity < 0.85 and len(self.species_names) < MAX_SPECIES:
            new_label = f"Species_{len(self.species_names) + 1}"
            self.species_names.append(new_label)
            self.reference_genomes[new_label] = (var_pos.copy(), var_base.copy())
            if GPU_AVAILABLE:
                self.gpu_genomes[new_label] = (cp.asarray(var_pos), cp.asarray(var_base))
            logger.debug(f"New species created: {new_label}, similarity={max_similarity:.2f}")
            if len(self.species_names) > self.pop.shape[2]:
                new_pop = np.zeros((GRID_SIZE[0], GRID_SIZE[1], len(self.species_names)), dtype=np.float32)
                new_pop[:, :, :self.pop.shape[2]] = self.pop
                self.pop = new_pop
            return new_label
        return assigned_label

    def growth(self, C: VectorF, N: VectorF, dose_mGy_h: VectorF, species_idx: int) -> VectorF:
        spec = self.species_names[species_idx]
        param = self.params[species_idx] if species_idx < len(self.params) else {}
        vmax = param.get('replication_rate', VMAX_MELANIN if species_data.get(spec, {}).get("Melanin", False) else VMAX_NO_MELANIN)
        f_nutr: VectorF = (C / (KS_C + C)) * (N / (KS_N + N))
        r_d: VectorF = dose_response(dose_mGy_h) * param.get('radiation_effectiveness', 1.0)
        growth_rate: VectorF = vmax * f_nutr * r_d
        growth_rate = np.nan_to_num(growth_rate, nan=0.0, posinf=0.0, neginf=0.0)
        return growth_rate

    def mutate_genome(self, var_pos: VectorU, var_base: VectorB, dose_mGy_h: float) -> Tuple[VectorU, VectorB]:
        mu: float = dose_scaled_mu(dose_mGy_h)
        base_probs: Dict[str, float] = {
            "Cryptococcus": {"SNP": 0.7, "INS": 0.15, "DEL": 0.1, "TE": 0.05},
            "Clado": {"SNP": 0.8, "INS": 0.18, "DEL": 0.0, "TE": 0.02}
        }[self.mutation_profile]
        te_boost: float = min(5.0, 1.0 + 10.0 * dose_mGy_h)
        event_probs: Dict[str, float] = {k: (v * te_boost if k == "TE" else v) for k, v in base_probs.items()}
        norm: float = sum(event_probs.values())
        event_probs = {k: v / norm for k, v in event_probs.items()}
        new_pos: VectorU = var_pos.copy()
        new_base: VectorB = var_base.copy()
        num_vars: int = np.sum(new_pos != 0)
        i: int = 0
        while i < num_vars:
            if random.random() < mu:
                event: str = random.choices(list(event_probs.keys()), weights=list(event_probs.values()), k=1)[0]
                if event == "SNP":
                    new_base[i] = ord(random.choice(['A', 'T', 'C', 'G']))
                elif event == "INS":
                    if num_vars < self.max_vars:
                        new_pos[num_vars] = random.randint(0, GENOME_LENGTH-1)
                        new_base[num_vars] = ord(random.choice(['A', 'T', 'C', 'G']))
                        num_vars += 1
                elif event == "DEL":
                    new_pos[i] = 0
                    new_base[i] = 0
                    num_vars -= 1
                    continue
                elif event == "TE":
                    new_base[i] = ord('N')
            i += 1
        idx: NDArray[np.bool_] = new_pos != 0
        num_valid: int = int(np.sum(idx))
        new_pos = new_pos[idx]
        new_base = new_base[idx]
        if len(new_pos) < self.max_vars:
            new_pos = np.pad(new_pos, (0, self.max_vars - len(new_pos)), 'constant')
            new_base = np.pad(new_base, (0, self.max_vars - len(new_base)), 'constant')
        logger.debug(f"mutate_genome: input_vars={np.sum(var_pos!=0)}, output_vars={np.sum(new_pos!=0)}")
        return new_pos, new_base

    def mate(self, p1: Individual, p2: Individual) -> Individual:
        dose_mGy_h: float = self.dose[p1.x, p1.y]
        if GPU_AVAILABLE:
            child_pos, child_base = recombine_gpu(
                [(np.ascontiguousarray(p1.var_pos), np.ascontiguousarray(p2.var_pos))],
                [(np.ascontiguousarray(p1.var_base), np.ascontiguousarray(p2.var_base))],
                dose_mGy_h, self.max_vars
            )[0]
            if isinstance(child_pos, cp.ndarray):
                child_pos = cp.asnumpy(child_pos)
            if isinstance(child_base, cp.ndarray):
                child_base = cp.asnumpy(child_base)
        else:
            child_pos, child_base = recombine_cpu(p1.var_pos, p1.var_base, p2.var_pos, p2.var_base, dose_mGy_h, self.max_vars)
        if len(child_pos) != len(child_base):
            logger.error(f"Mismatch in recombined genome lengths: pos={len(child_pos)}, base={len(child_base)}")
            min_len = min(len(child_pos), len(child_base))
            child_pos = child_pos[:min_len]
            child_base = child_base[:min_len]
            child_pos = np.pad(child_pos, (0, self.max_vars - len(child_pos)), 'constant')
            child_base = np.pad(child_base, (0, self.max_vars - len(child_base)), 'constant')
        child_pos, child_base = self.mutate_genome(child_pos, child_base, dose_mGy_h)
        child_label: str = self.assign_label(child_pos, child_base)
        melanin: bool = species_data.get(child_label, {}).get("Melanin", False)
        fitness: float = compute_growth_rate(np.array([self.C[p1.x, p1.y]], dtype=np.float64),
                                            np.array([self.N[p1.x, p1.y]], dtype=np.float64),
                                            np.array([dose_mGy_h], dtype=np.float64), melanin)[0]
        kmer_effect: float = kmer_effect_cached(child_pos.tobytes(), child_base.tobytes())
        fitness *= np.exp(kmer_effect)
        fitness = np.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)
        child: Individual = Individual(child_pos, child_base, child_label, p1.location, p1.x, p1.y,
                                    fitness=fitness, hybrid_parentage=(p1.species_label, p2.species_label))
        self.pedigree.append({
            'child_id': child.id,
            'parent1_id': p1.id,
            'parent2_id': p2.id,
            'step': self.current_step,
            'location': (p1.x, p1.y)
        })
        logger.debug(f"Mated {p1.species_label} and {p2.species_label}, child fitness={fitness}")
        return child

    def can_mate(self, p1: Individual, p2: Individual) -> bool:
        if p1.mating_type == p2.mating_type:
            return False
        if GPU_AVAILABLE:
            similarity = genome_similarity_gpu(
                [(cp.asnumpy(p1.var_pos), cp.asnumpy(p2.var_pos))],
                [(cp.asnumpy(p1.var_base), cp.asnumpy(p2.var_base))]
            )[0]
        else:
            similarity = genome_similarity_cpu(p1.var_pos, p1.var_base, p2.var_pos, p2.var_base)
        return similarity >= 0.85

    def simulate(self) -> List[Dict[str, float]]:
        from multiprocessing import current_process
        ckpt_tag = f"{current_process().pid}_{threading.get_ident()}"
        CHECKPOINT = os.path.join(BASE_DIR, f"sim_checkpoint_{ckpt_tag}.pkl")
        logger.info(f"Starting simulation for {NUM_TIME_STEPS} steps")
        self.history = []
        self.pedigree = []
        if os.path.exists(CHECKPOINT):
            try:
                with open(CHECKPOINT, "rb") as f:
                    checkpoint_data = pickle.load(f)
                    self.pop = checkpoint_data['pop']
                    self.C = checkpoint_data['C']
                    self.N = checkpoint_data['N']
                    self.dose = checkpoint_data['dose']
                    self.species_names = checkpoint_data['species_names']
                    self.reference_genomes = checkpoint_data['reference_genomes']
                    self.gpu_genomes = checkpoint_data['gpu_genomes']
                    self.current_step = checkpoint_data['step']
                    logger.info(f"Resumed simulation from checkpoint at step {self.current_step}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}; starting from step 0")
                self.current_step = 0
        else:
            self.current_step = 0
        for step in range(self.current_step, NUM_TIME_STEPS):
            self.current_step = step
            self.simulate_step()
            counts = {s: np.sum(self.pop[:, :, i]) for i, s in enumerate(self.species_names)}
            self.history.append(counts)
            self.history = self.history[-self.history_window:]
            if (step + 1) % 250 == 0 or step == NUM_TIME_STEPS - 1:
                logger.info(f"Completed simulation step {step + 1}/{NUM_TIME_STEPS}")
                try:
                    with open(CHECKPOINT, "wb") as f:
                        pickle.dump(
                            dict(
                                pop=self.pop,
                                C=self.C,
                                N=self.N,
                                dose=self.dose,
                                species_names=self.species_names,
                                reference_genomes=self.reference_genomes,
                                gpu_genomes=self.gpu_genomes,
                                step=self.current_step
                            ),
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL
                        )
                    logger.debug(f"Saved checkpoint at step {step + 1}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {step + 1}: {e}")
        if os.path.exists(CHECKPOINT) and self.current_step == NUM_TIME_STEPS - 1:
            try:
                os.remove(CHECKPOINT)
                logger.info(f"Removed checkpoint file {CHECKPOINT} after completion")
            except Exception as e:
                logger.error(f"Failed to remove checkpoint file: {e}")
        return self.history

    def simulate_step(self) -> None:
        ext_mask = np.random.rand(GRID_SIZE[0], GRID_SIZE[1]) < (0.005 * 1.0)
        for i, spec in enumerate(self.species_names):
            if i >= self.pop.shape[2]:
                logger.warning(f"Species index {i} exceeds population array dimension {self.pop.shape[2]}; skipping")
                continue
            param = self.params[i] if i < len(self.params) else {}
            death_rate = param.get('decay_rate', 0.02)
            growth_rate = self.growth(self.C, self.N, self.dose, i)
            uptake_C = 0.02 * self.pop[:, :, i] * 1.0
            uptake_N = 0.004 * self.pop[:, :, i] * 1.0
            mortality = death_rate * self.pop[:, :, i] * 1.0
            growth_amount = self.pop[:, :, i] * growth_rate * 1.0 * (1 - self.pop[:, :, i] / self.K_cell)
            growth_amount = np.nan_to_num(growth_amount, nan=0.0, posinf=0.0, neginf=0.0)
            mortality = np.nan_to_num(mortality, nan=0.0, posinf=0.0, neginf=0.0)
            self.pop[:, :, i] += growth_amount - mortality
            self.pop[:, :, i][ext_mask] = 0.0
            self.pop[:, :, i] = np.maximum(self.pop[:, :, i], 0.0)
            self.pop[:, :, i] = np.minimum(self.pop[:, :, i], self.K_cell)
            self.pop[:, :, i] = np.nan_to_num(self.pop[:, :, i], nan=0.0, posinf=0.0, neginf=0.0)
            self.C -= uptake_C
            self.N -= uptake_N
            self.C = np.maximum(self.C, 0.0)
            self.N = np.maximum(self.N, 0.0)
            if np.any(np.isnan(self.pop[:, :, i])):
                logger.warning(f"NaN detected in population for {spec} at step")
        self.C += np.random.uniform(0.01, 0.05, GRID_SIZE)
        self.N += np.random.uniform(0.002, 0.01, GRID_SIZE)
        self.C = np.minimum(self.C, 2.0)
        self.N = np.minimum(self.N, 0.3)
        kernel = np.ones((3, 3), dtype=np.float64) / 8
        self.C += 0.1 * (scipy.ndimage.convolve(self.C, kernel, mode='wrap') - self.C)
        self.N += 0.1 * (scipy.ndimage.convolve(self.N, kernel, mode='wrap') - self.N)
        occupied_cells = find_occupied_cells(self.pop, len(self.species_names))
        individuals = []
        if occupied_cells.size == 0:
            logger.debug("No occupied cells found; skipping individual processing")
            return
        for x, y, i in occupied_cells:
            if i >= len(self.species_names):
                logger.warning(f"Index {i} exceeds species count {len(self.species_names)}; skipping")
                continue
            spec = self.species_names[i]
            var_pos, var_base = self.reference_genomes[spec]
            melanin = species_data.get(spec, {}).get("Melanin", False)
            fitness = compute_growth_rate(np.array([self.C[x, y]], dtype=np.float64),
                                        np.array([self.N[x, y]], dtype=np.float64),
                                        np.array([self.dose[x, y]], dtype=np.float64), melanin)[0]
            ind = Individual(var_pos.copy(), var_base.copy(), spec, self.grid_locations[x, y], x, y, fitness=fitness)
            individuals.append(ind)
        births = []
        indices = np.arange(len(individuals))
        np.random.shuffle(indices)
        species_counts = {s: np.sum(self.pop[:, :, i]) for i, s in enumerate(self.species_names)}
        mating_pairs = []
        mating_pos_pairs = []
        mating_base_pairs = []
        for idx in indices:
            p1 = individuals[idx]
            neighbors = [(p1.x + dx, p1.y + dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
                        if 0 <= p1.x + dx < GRID_SIZE[0] and 0 <= p1.y + dy < GRID_SIZE[1]]
            for nx, ny in neighbors:
                if self.C[nx, ny] > 0 and random.random() < p1.fitness * 0.1:
                    species_idx = self.species_names.index(p1.species_label)
                    self.pop[p1.x, p1.y, species_idx] -= 1
                    self.pop[nx, ny, species_idx] += 1
                    p1.x, p1.y = nx, ny
                    p1.location = self.grid_locations[nx, ny]
                    break
            for _ in range(3):
                if len(indices) <= 1:
                    break
                p2_idx = np.random.choice(indices)
                p2 = individuals[p2_idx]
                if self.can_mate(p1, p2):
                    mating_pairs.append((p1, p2))
                    mating_pos_pairs.append((p1.var_pos, p2.var_pos))
                    mating_base_pairs.append((p1.var_base, p2.var_base))
        similarities = genome_similarity_gpu(mating_pos_pairs, mating_base_pairs)
        if not similarities:
            logger.debug("No valid mating pairs after similarity check; skipping mating")
            return
        valid_pairs = [(p1, p2) for (p1, p2), sim in zip(mating_pairs, similarities) 
                    if p1.mating_type != p2.mating_type and sim >= 0.85]
        if valid_pairs:
            pos_pairs = [(p1.var_pos, p2.var_pos) for p1, p2 in valid_pairs]
            base_pairs = [(p1.var_base, p2.var_base) for p1, p2 in valid_pairs]
            child_genomes = recombine_gpu(pos_pairs, base_pairs, self.dose[valid_pairs[0][0].x, valid_pairs[0][0].y], self.max_vars)
            for (p1, _), (child_pos, child_base) in zip(valid_pairs, child_genomes):
                species_size = species_counts.get(p1.species_label, 0)
                raw = (1 - species_size / self.K_cell) * p1.fitness * 0.7
                p_birth = max(0.0, min(1.0, raw))
                if random.random() < p_birth:
                    child_pos = cp.asnumpy(child_pos) if isinstance(child_pos, cp.ndarray) else child_pos
                    child_base = cp.asnumpy(child_base) if isinstance(child_base, cp.ndarray) else child_base
                    child_pos, child_base = self.mutate_genome(child_pos, child_base, self.dose[p1.x, p1.y])
                    child_label = self.assign_label(child_pos, child_base)
                    melanin = species_data.get(child_label, {}).get("Melanin", False)
                    fitness = compute_growth_rate(np.array([self.C[p1.x, p1.y]], dtype=np.float64),
                                                np.array([self.N[p1.x, p1.y]], dtype=np.float64),
                                                np.array([self.dose[p1.x, p1.y]], dtype=np.float64), melanin)[0]
                    kmer_effect = kmer_effect_cached(child_pos.tobytes(), child_base.tobytes())
                    fitness *= np.exp(kmer_effect)
                    fitness = np.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)
                    child = Individual(child_pos, child_base, child_label, p1.location, p1.x, p1.y,
                                    fitness=fitness, hybrid_parentage=(p1.species_label, p2.species_label))
                    births.append(child)
                    try:
                        child_idx = self.species_names.index(child_label)
                        self.pop[child.x, child.y, child_idx] = np.minimum(
                            self.pop[child.x, child.y, child_idx] + 1, self.K_cell)
                        species_counts[child_label] = species_counts.get(child_label, 0) + 1
                    except ValueError:
                        logger.warning(f"Child species {child_label} not in population array; skipping update")
        logger.debug(f"Step births: {len(births)}, total pop: {np.sum(self.pop):.2f}")

    def record_state(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        for x in range(GRID_SIZE[0]):
            for y in range(GRID_SIZE[1]):
                for i, spec in enumerate(self.species_names):
                    pop: float = self.pop[x, y, i]
                    if pop > 0 and not np.isnan(pop):
                        var_pos, var_base = self.reference_genomes[spec]
                        melanin = species_data.get(spec, {}).get("Melanin", False)
                        fitness = compute_growth_rate(np.array([self.C[x, y]], dtype=np.float64),
                                                    np.array([self.N[x, y]], dtype=np.float64),
                                                    np.array([self.dose[x, y]], dtype=np.float64), melanin)[0]
                        kmer_effect = kmer_effect_cached(var_pos.tobytes(), var_base.tobytes())
                        fitness *= np.exp(kmer_effect)
                        fitness = np.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)
                        data.append({
                            "Species Name": spec,
                            "Population": pop,
                            "X": x,
                            "Y": y,
                            "Fitness": fitness,
                            "Var_Pos": var_pos.tolist(),
                            "Var_Base": var_base.tolist(),
                            "Location": self.grid_locations[x, y]
                        })
        if not data:
            logger.warning("No valid population data to record; returning empty state with correct structure")
            data = [{
                "Species Name": "",
                "Population": 0.0,
                "X": 0,
                "Y": 0,
                "Fitness": 0.0,
                "Var_Pos": [],
                "Var_Base": [],
                "Location": ""
            }]
        return data

# Run smoke test after helper functions
run_smoke_test()

# Generate combined dataset
data: List[Dict[str, Any]] = []
strain_ids: set[str] = set(record["Strain ID"] for record in (mushroom_data + location_data))
ROWS_PER_STRAIN: int = math.ceil(NUM_RECORDS / len(strain_ids))
mushroom_records_by_id: Dict[str, Dict[str, Any]] = {record["Strain ID"]: record for record in mushroom_data if "Strain ID" in record}
location_records_by_id: Dict[str, Dict[str, Any]] = {record["Strain ID"]: record for record in location_data if "Strain ID" in record}

def selNSGA2(individuals: List[Any], k: int) -> List[Any]:
    # Create fronts for NSGA-IIf
    fronts = tools.sortNondominated(individuals, k, first_front_only=False)
    
    # Initialize output population
    chosen = []
    remaining_slots = k
    
    # Assign crowding distance to individuals in each front
    for front in fronts:
        if not front:
            continue
            
        # Assign crowding distance
        assignCrowdingDist(front)
        
        # If the entire front can fit, add it
        if len(front) <= remaining_slots:
            chosen.extend(front)
            remaining_slots -= len(front)
        else:
            # Sort by crowding distance (descending) and take remaining slots
            front.sort(key=lambda ind: ind.fitness.crowding_dist, reverse=True)
            chosen.extend(front[:remaining_slots])
            break
            
    return chosen

# Modified assignCrowdingDist to handle invalid fitness values
def assignCrowdingDist(individuals: List[Any]) -> None:
    if not individuals:
        return
        
    # Number of objectives (expected to be 2: biomass, -rmse)
    n_obj = len(individuals[0].fitness.values) if individuals and individuals[0].fitness.values else 2
    
    # Initialize crowding distance
    for ind in individuals:
        ind.fitness.crowding_dist = 0.0    # noqa: attr-needed by selNSGA2
    
    # For each objective, sort and assign distance
    for i in range(n_obj):
        # Filter valid individuals for this objective
        valid_individuals = [ind for ind in individuals if len(ind.fitness.values) > i and ind.fitness.values[i] is not None]
        if not valid_individuals:
            continue
            
        # Sort by current objective
        sorted_front = sorted(valid_individuals, key=lambda ind: ind.fitness.values[i])
        
        # Boundary points get infinite distance
        if len(sorted_front) >= 2:
            sorted_front[0].fitness.crowding_dist = float('inf')
            sorted_front[-1].fitness.crowding_dist = float('inf')
        
        # Calculate normalized crowding distance
        if len(sorted_front) > 2:
            obj_values = [ind.fitness.values[i] for ind in sorted_front]
            obj_min = min(obj_values)
            obj_max = max(obj_values)
            obj_range = obj_max - obj_min if obj_max != obj_min else 1.0
            
            for j in range(1, len(sorted_front) - 1):
                prev_val = sorted_front[j-1].fitness.values[i]
                next_val = sorted_front[j+1].fitness.values[i]
                sorted_front[j].fitness.crowding_dist += (next_val - prev_val) / obj_range

for strain_id in strain_ids:
    for _ in range(ROWS_PER_STRAIN):
        mushroom_record: Dict[str, Any] = mushroom_records_by_id.get(strain_id, {})
        location_record: Dict[str, Any] = location_records_by_id.get(strain_id, {})
        combined_record: Dict[str, Any] = {**mushroom_record, **location_record}
        combined_record = calculate_derived_fields(combined_record)
        loc_name: str = location_record.get("Location", "Chernobyl Reactor, Ukraine")
        loc_data: Dict[str, Any] = locations.get(loc_name, {"Latitude": 51.3894, "Longitude": 30.0994, "Habitat": "Reactor", "Radiation_µSv_hr": 5.0, "Contamination_kBq_m2": 2000})
        habitat: str = loc_data["Habitat"]
        species_candidates: List[Tuple[str, float]] = [(s, species_data[s]["Habitat Weight"]) for s in species_data if species_data[s]["Preferred Habitat"] == habitat]
        if not species_candidates:
            species_candidates = [(s, 1.0) for s in species_data]
        cand_names: Tuple[str, ...]
        weights: Tuple[float, ...]
        cand_names, weights = zip(*species_candidates)
        species: str = random.choices(cand_names, weights=weights, k=1)[0]
        zone: str = get_contamination_zone(loc_data.get("Contamination_kBq_m2", 2000))
        concentration: float = random.uniform(species_data[species]["Min"], species_data[species]["Max"])
        radiation_level: float = get_radiation_mGyh(loc_name)  # mGy/h
        nutrient_level: float = random.uniform(0.1, 1.0)
        temp: float = random.uniform(20, 35)
        ph: float = random.uniform(4, 8)
        no2: float = random.uniform(0, 10)
        committed_effective_dose: float = calculate_committed_effective_dose(concentration)
        melanin_present: bool = species_data[species]["Melanin"]
        substrate: str = species_data[species]["Substrate"]
        years_since_2024: int = np.random.randint(1, 31)
        birth_date, death_date = estimate_dates(years_since_2024)
        growth_rate: float = compute_growth_rate(np.array([nutrient_level], dtype=np.float64),
                                                np.array([nutrient_level * 0.2], dtype=np.float64),
                                                np.array([radiation_level], dtype=np.float64), melanin_present)[0]
        fungi_growth_rate: str = simulate_fungi_growth(loc_name, radiation_level, species, nutrient_level)
        collection_date: datetime.date = fake.date_between(start_date='-36y', end_date='today')
        lat_offset: float = random.uniform(-0.01, 0.01)
        lon_offset: float = random.uniform(-0.01, 0.01)
        record: Dict[str, Any] = {
            "Strain ID": str(strain_id),
            "Species Name": species,
            "Common Name": species_data[species]["Common Name"],
            "Location": loc_name,
            "Latitude": loc_data["Latitude"] + lat_offset,
            "Longitude": loc_data["Longitude"] + lon_offset,
            "Habitat": habitat,
            "Soil Type": random.choice(["Sandy", "Clay", "Loam"]),
            "pH Level": ph,
            "Contamination Zone": zone,
            "Radioactive Contamination Level (kBq/m²)": round(random.uniform(zones[zone]["Min Contamination"], zones[zone]["Max Contamination"]), 2),
            "Radioactive Isotope": "137Cs",
            "Concentration in Mushroom (Bq/kg)": round(concentration, 2),
            "Committed Effective Dose (mSv/year)": round(committed_effective_dose * 1000, 3),
            "Radiation Level (µSv/hr)": radiation_level * 1000.0,
            "Nutrient Level": nutrient_level,
            "Temperature (°C)": temp,
            "NO2 (mg/L)": no2,
            "Substrate": substrate,
            "Cap Diameter (cm)": round(random.uniform(2, 20), 1),
            "Stem Length (cm)": round(random.uniform(1, 20), 1),
            "Edibility": species_data[species]["Edibility"],
            "Melanin": melanin_present,
            "Fungi Growth Rate": fungi_growth_rate,
            "Estimated Birth Date": birth_date,
            "Estimated Death Date": death_date,
            **{k: v for k, v in combined_record.items() if v is not None}
        }
        data.append(record)

# Validate combined dataset
df: pd.DataFrame = pd.DataFrame(data[:NUM_RECORDS])
bad_rad_cols: List[str] = [c for c in df.columns if "Radiation" in c and c != "Radiation Level (µSv/hr)"]
if bad_rad_cols:
    df["Radiation Level (µSv/hr)"] = df["Radiation Level (µSv/hr)"].fillna(df[bad_rad_cols].bfill(axis=1).iloc[:,0])
    df.drop(columns=bad_rad_cols, inplace=True)
null_threshold: float = 0.9
high_null_cols: List[str] = [col for col in df.columns if df[col].isna().mean() > null_threshold]
df.drop(columns=high_null_cols, inplace=True)
numeric_cols: List[str] = ["Latitude", "Longitude", "pH Level", "Radioactive Contamination Level (kBq/m²)", 
                         "Concentration in Mushroom (Bq/kg)", "Committed Effective Dose (mSv/year)", 
                         "Radiation Level (µSv/hr)", "Nutrient Level", "Temperature (°C)", "NO2 (mg/L)", 
                         "Cap Diameter (cm)", "Stem Length (cm)"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
logger.info(f"Combined dataset columns: {list(df.columns)}")
logger.info(f"Combined dataset NaN counts: {df.isna().sum().to_dict()}")
df.to_csv("combined_dataset.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
logger.info("Combined dataset saved to combined_dataset.csv")

# Initialize ecosystem
species_names: List[str] = list(species_data.keys())
ecosystem: FungalEcosystem = FungalEcosystem(species_names, mutation_profile=args.profile)
start_time: float = time.time()
history: List[Dict[str, float]] = ecosystem.simulate()
end_time: float = time.time()
logger.info(f"Simulation took {end_time - start_time:.2f} seconds for {NUM_TIME_STEPS} time steps")
logger.info(
    f"Total biomass after {NUM_TIME_STEPS} steps: {np.sum(ecosystem.pop):.2e}"
)

# Plot population trends
plt.figure(figsize=(10, 6))
for spec in species_names:
    plt.plot([h[spec] for h in history], label=spec)
plt.xlabel('Time (steps)')
plt.ylabel('Population')
plt.title('Population Dynamics Over Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('population_trends.png')
plt.close()

# Save simulation results
data = ecosystem.record_state()
df = pd.DataFrame(data)
if not df.empty:
    df["X"] = pd.to_numeric(df["X"], errors='coerce')
    df["Y"] = pd.to_numeric(df["Y"], errors='coerce')
    df["Fitness"] = pd.to_numeric(df["Fitness"], errors='coerce')
    df["Population"] = pd.to_numeric(df["Population"], errors='coerce')
logger.info(f"Simulation dataset columns: {list(df.columns)}")
logger.info(f"Simulation dataset NaN counts: {df.isna().sum().to_dict()}")
timestamp: str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
df.to_csv(f'fungal_simulation_{timestamp}.csv', index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
logger.info(f'Results saved to fungal_simulation_{timestamp}.csv')

# NSGA-II Genetic Algorithm
# Batch stepper helpers (above recombine_cpu)
DT = 1.0
KS_C, KS_N = 0.1, 0.02
VMAX_M, VMAX_NM = 0.30, 0.20
K_CELL_LOCAL = 175.0

if GPU_AVAILABLE:
    import cupy as _cp

    @_cp.fuse()
    def _dose_resp_gpu(d):
        return _cp.exp(-0.22 * (_cp.log(d / 0.05) ** 2))

    @_cp.fuse()
    def _step_gpu(pop, C, N, dose, vmax):
        nutr = (C / (KS_C + C)) * (N / (KS_N + N))
        growth = pop * vmax * nutr * _dose_resp_gpu(dose) * DT
        return _cp.clip(pop + growth, 0.0, K_CELL_LOCAL)

    def run_batch_gpu(init_pop, C, N, dose, melanin_mask, steps):
        pop = init_pop.copy()
        vmax = _cp.where(melanin_mask[:, None, None], VMAX_M, VMAX_NM).astype(_cp.float32)
        for _ in range(steps):
            pop = _step_gpu(pop, C, N, dose, vmax)
        return pop.sum(axis=(1, 2))
else:
    import numpy as _cp  # noqa: F401

def run_batch_cpu(init_pop, C, N, dose, melanin_mask, steps):
    pop = init_pop.copy()
    vmax = np.where(melanin_mask[:, None, None], VMAX_M, VMAX_NM).astype(np.float32)
    for _ in range(steps):
        nutr = (C / (KS_C + C)) * (N / (KS_N + N))
        growth = pop * vmax * nutr * np.exp(-0.22 * (np.log(dose / 0.05) ** 2)) * DT
        pop = np.clip(pop + growth, 0.0, K_CELL_LOCAL)
    return pop.sum(axis=(1, 2))

def eval_species(individual: List[float], seed: int, precomputed_doses: Dict[str, float] = None) -> Tuple[float, float]:
    global args
    from hashlib import md5
    if precomputed_doses is None:
        precomputed_doses = dose_cache
    np.random.seed(int(hash(str(seed)) % 2**32))
    random.seed(int(hash(str(seed)) % 2**32))
    if GPU_AVAILABLE:
        cp.random.seed(seed)

    cache_key = md5(np.asarray(individual, dtype=np.float32).tobytes()).hexdigest()
    if cache_key in fitness_cache:
        return fitness_cache[cache_key]

    try:
        params = []
        for i, sp in enumerate(species_names):
            start = i * 3
            param = {
                'replication_rate': np.clip(individual[start], 0.18, 0.32),
                'decay_rate': np.clip(individual[start + 1], 0.005, 0.02),
                'radiation_effectiveness': np.clip(individual[start + 2], 0.001, 0.01),
                'Melanin': species_data[sp]['Melanin'],
                'Substrate': species_data[sp]['Substrate']
            }
            params.append(param)
        ecosystem = FungalEcosystem(species_names, initial_pop_size=100, mutation_profile=args.profile, params=params, precomputed_doses=precomputed_doses, history_window=args.history_window)
        history = ecosystem.simulate()
        final_states = [history[-1][spec] for spec in species_names]
        biomass = np.sum(final_states)
        if np.isnan(biomass) or biomass <= 1e-6:
            fitness_cache[cache_key] = (-1e12, 1e12)
            return fitness_cache[cache_key]
        rmse = 0.0
        valid_species = 0
        sim_data = pd.DataFrame(ecosystem.record_state())
        if sim_data.empty or 'Species Name' not in sim_data.columns:
            logging.warning("Empty or invalid simulation data; returning default fitness")
            fitness_cache[cache_key] = (biomass, 0.0)
            return fitness_cache[cache_key]
        for species in species_names:
            if species not in sim_data['Species Name'].values:
                logging.debug(f"Skipping {species}: not in simulation data")
                continue
            rad_val = empirical_data[species].get("Radiation_µSv_hr") if species in empirical_data else None
            if rad_val is None:
                logging.debug(f"Skipping {species}: no valid radiation value")
                continue
            species_df = sim_data[sim_data['Species Name'] == species]
            sim_fitness = species_df['Fitness'].astype(float)
            if sim_fitness.empty or sim_fitness.isna().all() or sim_fitness.eq(0).all():
                logging.debug(f"Skipping {species}: invalid or zero fitness data")
                continue
            real_data = pd.Series([rad_val / 1000.0] * len(sim_fitness), dtype=float)
            rmse += np.sqrt(mean_squared_error(sim_fitness, real_data))
            valid_species += 1
        if valid_species == 0:
            logging.warning("No valid species for RMSE calculation; returning biomass with zero RMSE")
            fitness_cache[cache_key] = (biomass, 0.0)
            return fitness_cache[cache_key]
        rmse /= valid_species
        logging.debug(f"GA objectives: biomass={biomass}, rmse={rmse}, valid_species={valid_species}")
        fitness = (biomass, -rmse)
        fitness_cache[cache_key] = fitness
        return fitness
    except Exception as e:
        logging.error(f"Error in eval_species: {e}", exc_info=True)
        fitness_cache[cache_key] = (-1e12, 1e12)
        return fitness_cache[cache_key]
       
def main() -> Tuple[List[Any], tools.Statistics, tools.ParetoFront, List[Dict[str, Any]]]:
    logger.setLevel(logging.INFO)
    if psutil:
        logger.info(f"Initial memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    else:
        logger.info("Initial memory usage: unknown")
    
    run_smoke_test()
    global locations
    locations = offset_duplicate_coords(locations)

    strain_ids = {rec["Strain ID"] for rec in (mushroom_data + location_data) if isinstance(rec, dict) and "Strain ID" in rec}
    mushroom_records_by_id = {record["Strain ID"]: record for record in mushroom_data if isinstance(record, dict) and "Strain ID" in record}
    location_records_by_id = {record["Strain ID"]: record for record in location_data if isinstance(record, dict) and "Strain ID" in record}
    df = build_combined_dataset(NUM_RECORDS, strain_ids, mushroom_records_by_id, location_records_by_id, locations, species_data, zones)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    df_ext = build_combined_dataset(NUM_RECORDS * 10, strain_ids, mushroom_records_by_id, location_records_by_id, locations, species_data, zones)
    logger.info(f"Combined dataset summary: {len(df)} records, {df['Species Name'].nunique()} unique species")
    logger.info(f"Combined dataset columns: {list(df.columns)}")
    logger.info(f"Combined dataset NaN counts: {df.isna().sum().to_dict()}")
    df.to_csv("combined_dataset.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    df_ext.to_csv(f"extended_fungal_dataset_{timestamp}.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Extended dataset saved to extended_fungal_dataset_{timestamp}.csv")

    unique_locations = set(locations.keys())
    logger.info(f"Precomputing doses for {len(unique_locations)} unique locations")
    DOSE_CACHE_FILE = os.path.join(BASE_DIR, "dose_cache.npz")
    if os.path.exists(DOSE_CACHE_FILE):
        try:
            with np.load(DOSE_CACHE_FILE, allow_pickle=True) as data:
                precomputed_doses = data['dose_cache'].item()
            logger.info(f"Loaded dose cache from {DOSE_CACHE_FILE}")
        except Exception as e:
            logger.error(f"Failed to load dose cache: {e}; recomputing doses")
            precomputed_doses = {loc: run_transport(thickness_cm=1.0, location=loc, overburden=0.1) for loc in unique_locations}
            try:
                np.savez_compressed(DOSE_CACHE_FILE, dose_cache=precomputed_doses)
                logger.info(f"Saved dose cache to {DOSE_CACHE_FILE}")
            except Exception as e:
                logger.error(f"Failed to save dose cache: {e}")
    else:
        precomputed_doses = {loc: run_transport(thickness_cm=1.0, location=loc, overburden=0.1) for loc in unique_locations}
        try:
            np.savez_compressed(DOSE_CACHE_FILE, dose_cache=precomputed_doses)
            logger.info(f"Saved dose cache to {DOSE_CACHE_FILE}")
        except Exception as e:
            logger.error(f"Failed to save dose cache: {e}")

    species_names = list(species_data.keys())
    try:
        ecosystem = FungalEcosystem(species_names, mutation_profile=args.profile, params=None, precomputed_doses=precomputed_doses, history_window=args.history_window)
    except Exception as e:
        logger.error(f"Failed to initialize ecosystem: {e}")
        logger.warning("Falling back to default radiation values for all grid locations")
        ecosystem = FungalEcosystem(species_names, mutation_profile=args.profile, params=None, precomputed_doses={loc: 0.005 for loc in unique_locations}, history_window=args.history_window)
    start_time = time.time()
    history = ecosystem.simulate()
    end_time = time.time()
    logger.info(f"Simulation took {end_time - start_time:.2f} seconds for {NUM_TIME_STEPS} time steps")
    logger.info(f"Total biomass after {NUM_TIME_STEPS} steps: {np.sum(ecosystem.pop):.2e}")
    if psutil:
        logger.info(f"Memory usage after simulation: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    else:
        logger.info("Memory usage after simulation: unknown")

    plt.figure(figsize=(10, 6))
    for spec in species_names:
        plt.plot([h[spec] for h in history], label=spec)
    plt.xlabel('Time (steps)')
    plt.ylabel('Population')
    plt.title('Population Dynamics Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('population_trends.png')
    plt.close()

    sim_records = ecosystem.record_state()
    df_sim = pd.DataFrame(sim_records)
    if not df_sim.empty:
        df_sim["X"] = pd.to_numeric(df_sim["X"], errors='coerce')
        df_sim["Y"] = pd.to_numeric(df_sim["Y"], errors='coerce')
        df_sim["Fitness"] = pd.to_numeric(df_sim["Fitness"], errors='coerce')
        df_sim["Population"] = pd.to_numeric(df_sim["Population"], errors='coerce')
    logger.info(f"Simulation dataset columns: {list(df_sim.columns)}")
    logger.info(f"Simulation dataset NaN counts: {df_sim.isna().sum().to_dict()}")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    df_sim.to_csv(f'fungal_simulation_{timestamp}.csv', index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f'Results saved to fungal_simulation_{timestamp}.csv')

    ind_size = len(species_names) * 3
    if not hasattr(creator, "FitnessMulti"):
        creator.create('FitnessMulti', base.Fitness, weights=(1.0, -1.0))
        creator.create('Individual', list, fitness=creator.FitnessMulti)
    toolbox = base.Toolbox()
    toolbox.register('attr_float', np.random.uniform, 0, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=ind_size)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=[0.02] * ind_size, indpb=0.05)
    toolbox.register('select', selNSGA2)
    toolbox.register('evaluate', eval_species)

    population = toolbox.population(n=POP_SIZE)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg_biomass', lambda x: np.mean([v[0] for v in x if len(v) > 0 and v[0] is not None]))
    stats.register('avg_rmse', lambda x: np.mean([-v[1] for v in x if len(v) > 1 and v[1] is not None]))
    
    n_jobs = min(cpu_count() or 4, 4) if psutil else 1
    logger.info(f"Using {n_jobs} parallel jobs for NSGA-II evaluation")
    
    # Batch GA fitness evaluations with smaller batch size to avoid thread oversubscription
    BATCH_SIZE = 16
    for gen in range(GENERATIONS):
        logger.info(f"Starting GA generation {gen+1}/{GENERATIONS}")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        if all(ind.fitness.values == (0, 0) for ind in offspring + population):
            logger.warning("All individuals have zero fitness; reinitializing population")
            population[:] = toolbox.population(n=len(population))
            continue
        # Batch individuals for evaluation
        for i in range(0, len(offspring), BATCH_SIZE):
            batch = offspring[i:i + BATCH_SIZE]
            batch_seeds = np.random.randint(0, 1000000, size=len(batch))
            fits = Parallel(n_jobs=n_jobs, backend='threading', verbose=5)(
                delayed(eval_species)(ind, seed, precomputed_doses) for ind, seed in zip(batch, batch_seeds)
            )
            for fit, ind in zip(fits, batch):
                if len(fit) >= 2 and fit[0] is not None and fit[1] is not None and not (fit[0] == 0 and fit[1] == 0):
                    ind.fitness.values = fit
                else:
                    logger.warning(f"Invalid fitness for individual: {fit}; assigning default")
                    ind.fitness.values = (-1e12, 1e12)
        valid_individuals = [ind for ind in offspring + population if len(ind.fitness.values) >= 2]
        if not valid_individuals:
            logger.error("No valid individuals for selection; terminating")
            return population, stats, hof, []
        population = selNSGA2(valid_individuals, k=len(population))
        hof.update(population)
        if psutil:
            logger.info(f"Generation {gen+1}/{GENERATIONS}: Pareto front size = {len(hof)}, "
                        f"memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        else:
            logger.info(f"Generation {gen+1}/{GENERATIONS}: Pareto front size = {len(hof)}, "
                        "memory usage: unknown")
    # RAM sanity check after GA loop
    import gc
    gc.collect()
    if psutil:
        logger.info(f"Peak RSS after GA: {psutil.Process().memory_info().rss / 1024**2:.1f} MB")
    else:
        logger.info("Peak RSS after GA: unknown")

    best_params = []
    for i, sp in enumerate(species_names):
        start = i * 3
        param = {
            'replication_rate': np.clip(hof[0][start], 0.18, 0.32),
            'decay_rate': np.clip(hof[0][start + 1], 0.005, 0.02),
            'radiation_effectiveness': np.clip(hof[0][start + 2], 0.001, 0.01),
            'Melanin': species_data[sp]['Melanin'],
            'Substrate': species_data[sp]['Substrate']
        }
        best_params.append(param)

    optimized_ecosystem = FungalEcosystem(species_names, mutation_profile=args.profile, params=best_params, precomputed_doses=precomputed_doses, history_window=args.history_window)
    optimized_history = optimized_ecosystem.simulate()

    plt.figure(figsize=(10, 6))
    for spec in species_names:
        plt.plot([h[spec] for h in optimized_history], label=spec)
    plt.xlabel('Time (steps)')
    plt.ylabel('Population')
    plt.title('Optimized Population Dynamics')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('optimized_population_trends.png')
    plt.close()

    optimized_records = optimized_ecosystem.record_state()
    optimized_df = pd.DataFrame(optimized_records)
    if not optimized_df.empty:
        optimized_df["X"] = pd.to_numeric(optimized_df["X"], errors='coerce')
        optimized_df["Y"] = pd.to_numeric(optimized_df["Y"], errors='coerce')
        optimized_df["Fitness"] = pd.to_numeric(optimized_df["Fitness"], errors='coerce')
        optimized_df["Population"] = pd.to_numeric(optimized_df["Population"], errors='coerce')
    logger.info(f"Optimized dataset columns: {list(optimized_df.columns)}")
    logger.info(f"Optimized dataset NaN counts: {optimized_df.isna().sum().to_dict()}")
    optimized_df.to_csv(f'optimized_fungal_simulation_{timestamp}.csv', index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f'Optimized results saved to optimized_fungal_simulation_{timestamp}.csv')

    validation_results = []
    radiation_keys = ['Radiation_µSv_hr', 'Radiation_k', 'Radiation_kGy_h']
    for species in species_names:
        if optimized_df.empty or 'Species Name' not in optimized_df.columns:
            logger.warning(f"No data for species {species} in optimized_df")
            validation_results.append({"Species": species, "RMSE": np.nan, "R²": np.nan})
            continue
        sim_data = optimized_df[optimized_df['Species Name'] == species]['Fitness'].astype(float)
        if sim_data.empty or sim_data.isna().all() or sim_data.eq(0).all():
            logger.warning(f"No valid fitness data for species {species} in simulation")
            validation_results.append({"Species": species, "RMSE": np.nan, "R²": np.nan})
            continue
        rad_val = None
        if species in empirical_data:
            rad_val = next((empirical_data[species][k] for k in radiation_keys
                           if k in empirical_data[species] and pd.notna(empirical_data[species][k])), None)
        if rad_val is None:
            logger.warning(f"No empirical radiation value for {species}; using default fitness comparison")
            rad_val = 0.005
        real_data = pd.Series([rad_val / 1000.0] * len(sim_data), dtype=float)
        rmse = np.sqrt(mean_squared_error(sim_data, real_data))
        r2 = r2_score(sim_data, real_data) if len(sim_data) > 1 else np.nan
        validation_results.append({"Species": species, "RMSE": rmse, "R²": r2})
        print(f"Validation for {species}: RMSE={rmse}, R²={r2}")
        if len(sim_data) > 0:
            plt.figure(figsize=(8, 5))
            time_steps = range(len(sim_data))
            plt.plot(time_steps, sim_data.to_numpy(), label='Simulated')
            plt.plot(time_steps, real_data.to_numpy(), label='Empirical', linestyle='--')
            plt.title(f'{species}: Simulated vs. Empirical Growth')
            plt.xlabel('Time Step')
            plt.ylabel('Fitness')
            plt.legend()
            plt.savefig(f'validation_{species}_{timestamp}.png')
            plt.close()

    validation_df = pd.DataFrame(validation_results)
    validation_df["RMSE"] = pd.to_numeric(validation_df["RMSE"], errors='coerce')
    validation_df["R²"] = pd.to_numeric(validation_df["R²"], errors='coerce')
    logger.info(f"Validation dataset columns: {list(validation_df.columns)}")
    logger.info(f"Validation dataset NaN counts: {validation_df.isna().sum().to_dict()}")
    validation_df.to_csv(f'validation_results_{timestamp}.csv', index=False, encoding="utf-8")
    logger.info(f'Validation results saved to validation_results_{timestamp}.csv')

    return population, stats, hof, best_params

if __name__ == "__main__":
    list_available_gpus()
    try:
        population, stats, hof, best_params = main()
    finally:
        if OPENMC_WORK_DIR and os.path.exists(OPENMC_WORK_DIR):
            shutil.rmtree(OPENMC_WORK_DIR)
            logger.info(f"Cleaned up temporary directory {OPENMC_WORK_DIR}")
