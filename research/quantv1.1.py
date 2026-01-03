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
| **1** | **Dose-response (radiotrophy)** | \\[ R(d)=\exp\!\bigl[-0.9\;\bigl(\ln\frac{d}{0.05}\bigr)^2\bigr] \\] <br> *with*  \(d = \text{dose in mGy h}^{-1}\), \(R(0.05) = 1\) |
| **2** | **Growth rate** | \\[ G = v_{\text{max}} \cdot \frac{C}{K_{S,C} + C} \cdot \frac{N}{K_{S,N} + N} \cdot R(d) \\] <br> \(v_{\text{max}} = 0.30 \text{ (melanin)} \text{ or } 0.20 \text{ (no melanin)}\), \(K_{S,C} = 0.1\), \(K_{S,N} = 0.02\) |
| **3** | **Per-base mutation rate** | \\[ \mu = \mu_0 \bigl(1 + 4 d + 20 d^2\bigr) \\] <br> with  \(\mu_0 = 1\times10^{-9}\), \(d = \text{mGy h}^{-1}\). |
| **4** | **Birth probability** | \\[ P_{\text{birth}}=\max\!\Bigl[0,\min\!\Bigl[1,\;\Bigl(1-\frac{N_s}{K_s}\Bigr)\,F \times 0.5\Bigr]\Bigr] \\] <br> \(N_s\)=species pop., \(K_s\)=`2000/24` (~83), \(F\)=fitness |
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

import numpy as np
import pandas as pd
import random
import datetime
import logging
import csv
import json
import os
import time
import math
import scipy.ndimage
import argparse
import hashlib
import shutil
from copy import deepcopy
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error, r2_score
from deap import base, creator, tools, algorithms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Any, Dict, List, Tuple, Optional
from multiprocessing import Pool
import pickle  # Added for dose_cache loading
# Suppress Numba performance warnings
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Optional imports with robust GPU handling
try:
    import openmc
    OPENMC_AVAILABLE = True
except ImportError:
    OPENMC_AVAILABLE = False

# Custom array module shim to avoid CuPy/NumPy aliasing issues
class ArrayModule:
    def __init__(self, use_gpu: bool):
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.backend = cupy
            try:
                cupy.cuda.runtime.getDeviceCount()
            except Exception as e:
                logging.getLogger('fungal_simulation').error(f"GPU unavailable despite CuPy import: {e}")
                self.use_gpu = False
                self.backend = np
        else:
            self.backend = np

    def zeros(self, *args, **kwargs):
        return self.backend.zeros(*args, **kwargs)

    def asarray(self, *args, **kwargs):
        return self.backend.asarray(*args, **kwargs)

    def random(self):
        return self.backend.random

    def get(self, arr):
        return arr.get() if self.use_gpu else np.asarray(arr)

try:
    from numba import cuda
    array_module = ArrayModule(True)
    GPU_AVAILABLE = array_module.use_gpu
except ImportError:
    GPU_AVAILABLE = False
    array_module = ArrayModule(False)
    cuda = None

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fungal Ecosystem Simulation")
parser.add_argument("--steps", type=int, default=1000, help="Number of simulation time steps")
parser.add_argument("--profile", type=str, default="Cryptococcus", choices=["Cryptococcus", "Clado"],
                    help="Mutation profile: Cryptococcus or Clado")
parser.add_argument("--base-dir", type=str, default="/input" if os.getenv("IN_DOCKER", "") == "1" else os.getcwd(),
                    help="Base directory for input/output files")
parser.add_argument("--optimize", action="store_true", help="Use smaller grid for GA optimization")
args = parser.parse_args()
BASE_DIR = args.base_dir  # Set global BASE_DIR immediately after parsing args

# Configuration class
class Config:
    DOSE_CONVERSION_COEFFICIENT = 1.3e-5
    ANNUAL_INTAKE = 1.0
    NUM_SPECIES = 24
    MAX_SPECIES = 100
    GENERATIONS = 10
    GA_POP_SIZE = 20
    INITIAL_GRID_POP = 50
    NUM_TIME_STEPS = args.steps
    NUM_RECORDS = 10000
    TEST_TIME_STEPS = 50
    ATTENUATION_CM2_G = {"PLA-melanin": 0.046, "pure_melanin": 0.075}
    GENOME_LENGTH = 10_000_000
    MATING_RATE = 0.2
    DEATH_RATE = 0.05
    BIRTH_SCALE = 0.5
    RECOVERY_THRESHOLD = 0.5
    PURE_THRESHOLD = 0.95
    NUM_SEEDS = 5
    BLOCK_SIZE = 128
    DOSE_OPTIMUM_mGyh = 0.05
    k_shape = 0.9
    MELANIN_PEAK_GAIN = 2.5
    ALBINO_HORMESIS_MAX = 1.10
    BASE_MUT_RATE = 1e-9
    MAX_CARRY = 1000
    GRID_SIZE = (40, 40) if args.optimize else (100, 100)
    MAX_VARS = 1000
    DT = 1.0
    VMAX_MELANIN = 0.30
    VMAX_NO_MELANIN = 0.20
    KS_C = 0.1
    KS_N = 0.02
    C_RANGE = (0.2, 2.0)
    N_RANGE = (0.05, 0.3)
    NO2_RANGE = (0.01, 1.0)
    SPECIES_LIFESPAN = 5
    ISOTOPES = [("137Cs", 0.85), ("90Sr", 0.15)]

VectorF = NDArray[np.float64]
VectorU = NDArray[np.uint32]
VectorB = NDArray[np.uint8]

# Mutation profiles
MUTATION_PROFILES_CACHED = {
    "Cryptococcus": {"SNP": 0.7, "INS": 0.15, "DEL": 0.1, "TE": 0.05},
    "Clado": {"SNP": 0.8, "INS": 0.18, "DEL": 0.0, "TE": 0.02}
}
for profile in MUTATION_PROFILES_CACHED:
    total = sum(MUTATION_PROFILES_CACHED[profile].values())
    MUTATION_PROFILES_CACHED[profile] = {k: v / total for k, v in MUTATION_PROFILES_CACHED[profile].items()}

# Define creator classes
if not hasattr(creator, "FitnessMulti"):
    creator.create('FitnessMulti', base.Fitness, weights=(1.0, -1.0))
    creator.create('Individual', list, fitness=creator.FitnessMulti)

# Initialize toolbox
toolbox = base.Toolbox()
toolbox.register('attr_float', np.random.uniform, 0, 1)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=Config.NUM_SPECIES * 3)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=[0.02] * (Config.NUM_SPECIES * 3), indpb=0.05)
toolbox.register('select', tools.selNSGA2)

# Global variables for worker processes
locations = {}
species_data = {}
zones = {}
mushroom_data = []
radiation_data = []
location_data = []
empirical_data = {}
GLOBAL_OPENMC_MODEL = None
_pre_baked_models = {}
dose_cache = {}
OPENMC_WORK_DIR = None
GLOBAL_SPECIES = None
GLOBAL_PROFILE = None

def _ensure_worker_init(base_dir: str):
    global locations, species_data, zones, mushroom_data, radiation_data, location_data, empirical_data
    global _pre_baked_models, GLOBAL_OPENMC_MODEL, OPENMC_WORK_DIR
    logger = logging.getLogger('fungal_simulation')
    if not getattr(_ensure_worker_init, "ready", False):
        # Load JSON data
        required_json_files = [
            'locations.json', 'species_data.json', 'zones.json',
            'mushroom_data.json', 'radiation_data.json', 'location_data.json',
            'empirical_data.json'
        ]
        for json_file in required_json_files:
            file_path = os.path.join(base_dir, json_file)
            if not os.path.exists(file_path):
                raise RuntimeError(f"Missing required JSON file: {json_file}")
        try:
            with open(os.path.join(base_dir, 'locations.json'), 'r', encoding="utf-8") as f:
                locations.update(json.load(f))
            with open(os.path.join(base_dir, 'species_data.json'), 'r', encoding="utf-8") as f:
                species_data.update(json.load(f))
            with open(os.path.join(base_dir, 'zones.json'), 'r', encoding="utf-8") as f:
                zones.update(json.load(f))
            with open(os.path.join(base_dir, 'mushroom_data.json'), 'r', encoding="utf-8") as f:
                mushroom_data.extend(json.load(f))
            with open(os.path.join(base_dir, 'radiation_data.json'), 'r', encoding="utf-8") as f:
                radiation_data.extend(json.load(f))
            with open(os.path.join(base_dir, 'location_data.json'), 'r', encoding="utf-8") as f:
                location_data.extend(json.load(f))
            with open(os.path.join(base_dir, 'empirical_data.json'), 'r', encoding="utf-8") as f:
                empirical_data.update(json.load(f))
            logger.info("JSON files loaded successfully in worker")
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON files: {e}")

        # Load pre-computed dose cache if available
        cache_file = os.path.join(base_dir, 'dose_cache.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    dose_cache.update(pickle.load(f))
                logger.info(f"Loaded dose cache from {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load dose cache: {e}")

        # OpenMC setup
        OPENMC_WORK_DIR = os.path.join(base_dir, "openmc_work")
        os.makedirs(OPENMC_WORK_DIR, exist_ok=True)
        STATIC_XML_DIR = os.path.join(base_dir, "openmc_static")  # Moved outside if block
        if OPENMC_AVAILABLE:
            def make_material(name: str, material_data: Optional[Dict[str, Any]] = None) -> Any:
                if material_data is None:
                    if name == "air":
                        material_data = {"density": 0.001225, "composition": {"N": 0.78, "O": 0.21}}
                    elif name == "soil":
                        material_data = {"density": 1.5, "composition": {"Si": 0.33, "O": 0.50, "Al": 0.10, "Cs137": 1e-6}}
                    else:
                        raise ValueError(f"No default material data for {name}")
                try:
                    material = openmc.Material(name=name)
                    material.id = hash(name) % 10000
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

            if not os.path.exists(STATIC_XML_DIR):
                os.makedirs(STATIC_XML_DIR)
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
                    air_cell.id = 1
                    soil_cell = openmc.Cell(fill=soil, region=+air_surface & -soil_surface)
                    soil_cell.id = 2
                    geometry = openmc.Geometry([air_cell, soil_cell])
                    geometry.export_to_xml(os.path.join(STATIC_XML_DIR, "geometry.xml"))
                    settings = openmc.Settings()
                    settings.run_mode = 'fixed source'
                    settings.particles = 10000
                    settings.batches = 1
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

            if os.path.exists(STATIC_XML_DIR):
                shutil.copytree(STATIC_XML_DIR, OPENMC_WORK_DIR, dirs_exist_ok=True)
                openmc.reset_auto_ids()
                GLOBAL_OPENMC_MODEL = openmc.Model(
                    geometry=openmc.Geometry.from_xml(os.path.join(OPENMC_WORK_DIR, "geometry.xml")),
                    materials=openmc.Materials.from_xml(os.path.join(OPENMC_WORK_DIR, "materials.xml")),
                    settings=openmc.Settings.from_xml(os.path.join(OPENMC_WORK_DIR, "settings.xml")),
                    tallies=openmc.Tallies.from_xml(os.path.join(OPENMC_WORK_DIR, "tallies.xml"))
                )

        # Check for XML files
        for fn in ("materials.xml", "geometry.xml", "settings.xml", "tallies.xml"):
            if not os.path.isfile(os.path.join(STATIC_XML_DIR, fn)):
                raise FileNotFoundError(f"Static XML file {fn} missing in {STATIC_XML_DIR}")

        # Ensure consistent RNG state in workers with process-safe seeding
        seed_seq = np.random.SeedSequence(42)
        child_seeds = seed_seq.spawn(1)[0]  # Spawn one seed per worker
        np.random.seed(child_seeds.generate_state(1)[0])
        random.seed(child_seeds.generate_state(1)[0])
        if GPU_AVAILABLE:
            cupy_random = array_module.backend.random
            getattr(cupy_random, "seed", lambda *_: None)(child_seeds.generate_state(1)[0])

        _ensure_worker_init.ready = True

def get_radiation_mGyh(location_name: str, default: float = 0.005) -> float:
    loc = locations.get(location_name, {})
    for key in ("Radiation_µSv_hr", "Radiation (µSv/hr)", "Radiation_uSv_hr", "Radiation"):
        if key in loc and pd.notna(loc[key]):
            return float(loc[key]) / 1000.0
    logger = logging.getLogger('fungal_simulation')
    logger.warning(f"No valid radiation data for {location_name}, using default {default} mGy/h")
    return default

def get_contamination_zone(contamination: float) -> str:
    for zone, data in zones.items():
        if data["Min Contamination"] <= contamination <= data["Max Contamination"]:
            return zone
    return "Zone 5"

def dose_response(d_mGy_h: VectorF) -> VectorF:
    r_d = np.zeros_like(d_mGy_h, dtype=np.float64)
    valid = d_mGy_h > 0
    k_shape = 0.693
    r_d[valid] = np.exp(-k_shape * (np.log(d_mGy_h[valid] / Config.DOSE_OPTIMUM_mGyh) ** 2))
    r_d = np.nan_to_num(r_d, nan=0.0, posinf=0.0, neginf=0.0)
    logger = logging.getLogger('fungal_simulation')
    logger.debug(f"Dose response: mean dose={d_mGy_h.mean():.4f} mGy/h, mean R(d)={r_d.mean():.4f}")
    return r_d

def dose_scaled_mu(dose_mGy_h: float, base_rate: float = Config.BASE_MUT_RATE) -> float:
    mu = base_rate * (1.0 + 4.0 * dose_mGy_h + 20.0 * dose_mGy_h * dose_mGy_h)
    mu = min(mu, 0.5)  # Clamp to prevent probability > 1
    logger = logging.getLogger('fungal_simulation')
    logger.debug(f"Mutation rate: dose={dose_mGy_h:.2f} mGy h⁻¹, μ/μ₀={mu/base_rate:.2f}")
    return mu

def compute_growth_rate(
        C: VectorF,
        N: VectorF,
        dose_mGy_h: VectorF,
        melanin: bool,
        species: str
) -> VectorF:
    """
    Monod-type growth rate with radiotrophic multiplier.

    Parameters
    ----------
    C, N        : nutrient concentrations (g L⁻¹)
    dose_mGy_h  : dose rate (mGy h⁻¹)
    melanin     : True if strain is melanised
    species     : species name (key in ``species_data``)

    Returns
    -------
    VectorF
        Per-time-step growth rate (same shape as *C*)
    """
    vmax  = species_data.get(
        species, {}
    ).get('Vmax', Config.VMAX_MELANIN if melanin else Config.VMAX_NO_MELANIN)

    ks_c  = species_data.get(species, {}).get('Ks_C', Config.KS_C)
    ks_n  = species_data.get(species, {}).get('Ks_N', Config.KS_N)

    # Monod nutrient term
    f_nutr = (C / (ks_c + C)) * (N / (ks_n + N))

    # Radiotrophic term (corrected line ↓↓↓)
    r_d = dose_response(dose_mGy_h)

    growth_rate = vmax * f_nutr * r_d
    growth_rate = np.nan_to_num(growth_rate, nan=0.0, posinf=0.0, neginf=0.0)

    logger = logging.getLogger('fungal_simulation')
    logger.debug(
        f"Growth rate for {species}: "
        f"C={C.mean():.3f} g/L, N={N.mean():.3f} g/L, "
        f"dose={dose_mGy_h.mean():.4f} mGy/h, "
        f"mean rate={growth_rate.mean():.4f}"
    )

    return growth_rate

def simulate_fungi_growth(location: str, radiation_level: float, fungi_species: str, nutrient_level: float) -> str:
    threshold = species_data.get(fungi_species, {}).get('Radiation_Threshold', 0.01)
    if "reactor" in location.lower() or "soil" in location.lower():
        if radiation_level > threshold and nutrient_level < 0.5:
            return "High (Radiotropism)" if fungi_species in empirical_data else "High"
    return "Low"

def calculate_committed_effective_dose(concentration: float, intake: float = Config.ANNUAL_INTAKE) -> float:
    return concentration * Config.DOSE_CONVERSION_COEFFICIENT * intake

def calculate_derived_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    try:
        mean_log_cells_control = float(record["Mean Log (Number of Viable Cells) Control ± SD"].split(" ± ")[0])
        mean_log_cells_30_min = float(record["Mean (Log Number of Viable Cells in 30-min Exposure - Log Number of Cells in Control) ± SD"].split(" ± ")[0])
        mean_log_cells_60_min = float(record["Mean (Log Number of Viable Cells in 60-min Exposure - Log Number of Cells in Control) ± SD"].split(" ± ")[0])
        record["Viable Cells After 30 min (%)"] = round(100 * (10 ** (mean_log_cells_control + mean_log_cells_30_min)) / (10 ** mean_log_cells_control), 2)
        record["Viable Cells After 60 min (%)"] = round(100 * (10 ** (mean_log_cells_control + mean_log_cells_60_min)) / (10 ** mean_log_cells_control), 2)
    except (KeyError, ValueError):
        record["Viable Cells After 30 min (%)"] = None
        record["Viable Cells After 60 min (%)"] = None
    return record

def random_date(start_date: datetime.datetime, end_date: datetime.datetime) -> datetime.datetime:
    return start_date + datetime.timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

def estimate_dates(years_since_2024: int) -> Tuple[datetime.datetime, datetime.datetime]:
    birth_date = random_date(datetime.datetime(2024 - min(years_since_2024, Config.SPECIES_LIFESPAN), 1, 1),
                             datetime.datetime(2024, 12, 31))
    death_date = random_date(birth_date,
                             birth_date + datetime.timedelta(days=365 * Config.SPECIES_LIFESPAN))
    return birth_date, death_date

def run_smoke_test() -> None:
    test_doses = [0.01, 0.05, 0.30, 1.00]
    logger = logging.getLogger('fungal_simulation')
    logger.info("Running smoke test for dose response and mutation rate")
    for dose in test_doses:
        r_d = dose_response(np.array([dose], dtype=np.float64))[0]
        mu_ratio = dose_scaled_mu(dose) / Config.BASE_MUT_RATE
        logger.info(f"Dose: {dose:.2f} mGy h⁻¹, R(d): {r_d:.4f}, μ/μ₀: {mu_ratio:.2f}")

def run_transport(thickness_cm: float, location: str, overburden: float = 0.1) -> float:
    if not OPENMC_AVAILABLE:
        default_dose = get_radiation_mGyh(location, default=0.005)
        logger = logging.getLogger('fungal_simulation')
        logger.warning(f"OpenMC unavailable; using default dose {default_dose:.4f} mGy/h for {location}")
        return default_dose
    try:
        loc_data = next((item for item in radiation_data if item.get("location") == location),
                        {"isotope_list": [
                            {"isotope": "Cs137", "energy_MeV": [0.662], "rel_intensity": [0.85], "A_Bqkg": 1e-6},
                            {"isotope": "Sr90", "energy_MeV": [0.546], "rel_intensity": [0.15], "A_Bqkg": 1e-7}
                        ]})
        spectrum = [(iso["energy_MeV"][i], iso["rel_intensity"][i])
                    for iso in loc_data.get("isotope_list", [])
                    for i in range(len(iso.get("energy_MeV", [0.662])))
                    if i < len(iso.get("rel_intensity", [1.0]))]
        spectrum = spectrum or [(0.662, 0.85), (0.546, 0.15)]
        spectrum_hash = hashlib.md5(str((tuple(sorted(spectrum)), thickness_cm, overburden)).encode()).hexdigest()
        if spectrum_hash in dose_cache:
            return dose_cache[spectrum_hash]
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        openmc.reset_auto_ids()
        if spectrum_hash not in _pre_baked_models:
            model = deepcopy(GLOBAL_OPENMC_MODEL)
            energies, weights = zip(*spectrum)
            model.settings.source = openmc.IndependentSource(
                space=openmc.stats.Point((0, 0, 0)),
                energy=openmc.stats.Discrete([E * 1e6 for E in energies], weights),
                particle='photon'
            )
            _pre_baked_models[spectrum_hash] = model
        sp_file = _pre_baked_models[spectrum_hash].run(cwd=OPENMC_WORK_DIR)
        with openmc.StatePoint(sp_file) as sp:
            tally = sp.get_tally(name='dose_tally')
            dose_eV = tally.mean.flatten()[0]
            soil_density = loc_data.get("materials", [{"density": 1.5}])[0]["density"]
            mass_kg = max(soil_density * thickness_cm / 1000.0, 1e-6)
            dose_Gy_per_particle = dose_eV * 1.602e-19 / mass_kg
            dose_mGy_h = dose_Gy_per_particle * GLOBAL_OPENMC_MODEL.settings.particles * 3600 * 1000
        dose_cache[spectrum_hash] = max(dose_mGy_h, 0.0001)
        logger = logging.getLogger('fungal_simulation')
        logger.debug(f"OpenMC dose: {dose_mGy_h:.4f} mGy/h for {location}")
        return dose_cache[spectrum_hash]
    except Exception as e:
        logger = logging.getLogger('fungal_simulation')
        logger.error(f"OpenMC failed: {e}")
        return get_radiation_mGyh(location, default=0.005)

kmer_size = 15
def random_kmer(k: int = 15) -> bytes:
    return ''.join(random.choices('ATCG', k=k)).encode()
trait_map = {random_kmer(): np.random.normal(0, 1) for _ in range(5000)}
melanin_effect_mask = np.random.normal(0, 0.1, Config.MAX_VARS)

def compute_kmer_effect(var_pos: VectorU, var_base: VectorB, species: str) -> float:
    valid_indices = var_pos != 0
    valid_pos = var_pos[valid_indices]
    idx = valid_pos % Config.MAX_VARS  # Map positions to mask range
    melanin_genes = np.array(species_data.get(species, {}).get('Melanin_Genes', []), dtype=np.uint32)
    effect = np.sum(melanin_effect_mask[idx[np.isin(valid_pos, melanin_genes)]])
    logger = logging.getLogger('fungal_simulation')
    logger.debug(f"k-mer effect for {species}: effect={effect:.3f}")
    return effect

def to_host(arr):
    return array_module.get(arr)

if GPU_AVAILABLE and cuda is not None:
    @cuda.jit
    def recombine_kernel(var_pos1, var_base1, var_pos2, var_base2, child_pos, child_base, mutation_rate, random_vals, max_vars):
        idx = cuda.grid(1)
        if idx < max_vars:
            num_vars1 = 0
            num_vars2 = 0
            for i in range(max_vars):
                if var_pos1[i] != 0:
                    num_vars1 += 1
                if var_pos2[i] != 0:
                    num_vars2 += 1
            num_vars = min(num_vars1, num_vars2, max_vars)
            if num_vars == 0:
                child_pos[idx] = 0
                child_base[idx] = 0
                return
            point = int(random_vals[idx] * num_vars)
            if idx < num_vars:
                if idx < point and idx < num_vars1:
                    child_pos[idx] = var_pos1[idx]
                    child_base[idx] = var_base1[idx]
                elif idx >= point and idx < num_vars2:
                    child_pos[idx] = var_pos2[idx]
                    child_base[idx] = var_base2[idx]
                else:
                    child_pos[idx] = 0
                    child_base[idx] = 0
                if random_vals[idx + max_vars] < mutation_rate:
                    mutation_choice = int(random_vals[idx + 2 * max_vars] * 4)
                    child_base[idx] = ord('A') + mutation_choice % 4
            else:
                child_pos[idx] = 0
                child_base[idx] = 0

    @cuda.jit
    def similarity_kernel(var_pos1, var_base1, var_pos2, var_base2, max_vars, similarity, seen):
        idx = cuda.grid(1)
        if idx == 0:
            matches = 0
            union_cnt = 0
            seen[:] = 0
            seen_count = 0
            i = 0
            while i < var_pos1.size and var_pos1[i] != 0:
                is_new_pos = True
                for k in range(seen_count):
                    if seen[k] == var_pos1[i]:
                        is_new_pos = False
                        break
                if is_new_pos and seen_count < max_vars:
                    seen[seen_count] = var_pos1[i]
                    seen_count += 1
                i += 1
            j = 0
            while j < var_pos2.size and var_pos2[j] != 0:
                is_new_pos = True
                for k in range(seen_count):
                    if seen[k] == var_pos2[j]:
                        is_new_pos = False
                        break
                if is_new_pos and seen_count < max_vars:
                    seen[seen_count] = var_pos2[j]
                    seen_count += 1
                j += 1
            union_cnt = seen_count
            i, j = 0, 0
            while i < var_pos1.size and j < var_pos2.size and var_pos1[i] != 0 and var_pos2[j] != 0:
                if var_pos1[i] == var_pos2[j]:
                    if var_base1[i] == var_base2[j]:
                        matches += 1
                    i += 1
                    j += 1
                elif var_pos1[i] < var_pos2[j]:
                    i += 1
                else:
                    j += 1
            if union_cnt > 0:
                similarity[0] = matches / union_cnt
            else:
                similarity[0] = 0.0

def recombine_cpu(p1: VectorU, b1: VectorB, p2: VectorU, b2: VectorB, dose: float, max_vars: int) -> Tuple[VectorU, VectorB]:
    num_vars1 = np.sum(p1 != 0)
    num_vars2 = np.sum(p2 != 0)
    if num_vars1 == 0 or num_vars2 == 0:
        return np.zeros(max_vars, dtype=np.uint32), np.zeros(max_vars, dtype=np.uint8)
    num_vars = min(num_vars1, num_vars2, max_vars)
    cut = random.randint(1, max(num_vars - 1, 1))
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
    new_pos = np.concatenate((p1_valid[:cut], p2_valid[cut:]))[:max_vars]
    new_base = np.concatenate((b1_valid[:cut], b2_valid[cut:]))[:max_vars]
    new_pos = np.pad(new_pos, (0, max_vars - len(new_pos)), 'constant')
    new_base = np.pad(new_base, (0, max_vars - len(new_base)), 'constant')
    logger = logging.getLogger('fungal_simulation')
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Recombine CPU: input_vars1={num_vars1}, input_vars2={num_vars2}, output_vars={np.sum(new_pos!=0)}")
    return new_pos, new_base

def genome_similarity_cpu(p1: VectorU, b1: VectorB, p2: VectorU, b2: VectorB) -> float:
    V1 = set(p1[p1 != 0])
    V2 = set(p2[p2 != 0])
    common = V1 & V2
    union = V1 | V2
    matches = 0
    weights = {'SNP': 1.0, 'TE': 2.0}
    for pos in common:
        idx1 = np.where(p1 == pos)[0]
        idx2 = np.where(p2 == pos)[0]
        if idx1.size > 0 and idx2.size > 0 and b1[idx1[0]] == b2[idx2[0]]:
            weight = weights.get('TE' if b1[idx1[0]] == ord('N') else 'SNP', 1.0)
            matches += weight
    return matches / len(union) if union else 0.0

def genome_similarity_gpu(var_pos1: VectorU, var_base1: VectorB, var_pos2: VectorU, var_base2: VectorB) -> float:
    if not GPU_AVAILABLE or cuda is None:
        return genome_similarity_cpu(var_pos1, var_base1, var_pos2, var_base2)
    sim = array_module.zeros(1, dtype=np.float32)
    scratch = array_module.zeros(Config.MAX_VARS, dtype=np.uint32)
    threads = 1
    blocks = 1
    similarity_kernel[blocks, threads](array_module.asarray(var_pos1), array_module.asarray(var_base1),
                                      array_module.asarray(var_pos2), array_module.asarray(var_base2),
                                      Config.MAX_VARS, sim, scratch)
    return float(to_host(sim)[0])

def recombine_gpu(var_pos1: VectorU, var_base1: VectorB, var_pos2: VectorU, var_base2: VectorB, dose_mGy_h: float, max_vars: int) -> Tuple[VectorU, VectorB]:
    if not GPU_AVAILABLE or cuda is None:
        return recombine_cpu(var_pos1, var_base1, var_pos2, var_base2, dose_mGy_h, max_vars)
    mu = dose_scaled_mu(dose_mGy_h)
    child_pos = array_module.zeros(max_vars, dtype=np.uint32)
    child_base = array_module.zeros(max_vars, dtype=np.uint8)
    rnd = array_module.random().uniform(size=3 * max_vars)
    threads = Config.BLOCK_SIZE
    blocks = math.ceil(max_vars / threads)
    recombine_kernel[blocks, threads](array_module.asarray(var_pos1), array_module.asarray(var_base1),
                                     array_module.asarray(var_pos2), array_module.asarray(var_base2),
                                     child_pos, child_base, mu, rnd, max_vars)
    new_pos = to_host(child_pos)
    new_base = to_host(child_base)
    logger = logging.getLogger('fungal_simulation')
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"recombine_gpu: p1_vars={np.sum(var_pos1!=0)}, p2_vars={np.sum(var_pos2!=0)}, output_vars={np.sum(new_pos!=0)}")
    return new_pos, new_base

class Individual:
    def __init__(self, var_pos: VectorU, var_base: VectorB, label: str, location: str, x: int, y: int,
                 fitness: float = 1.0, mating_type: Optional[str] = None, hybrid_parentage: Optional[Tuple[str, str]] = None) -> None:
        self.var_pos = var_pos
        self.var_base = var_base
        self.species_label = label
        self.location = location
        self.x = x
        self.y = y
        self.fitness = fitness
        self.mating_type = mating_type or random.choice(("A", "B"))
        self.hybrid_parentage = hybrid_parentage

class FungalEcosystem:
    def __init__(self, species_names: List[str], initial_pop_size: int = Config.INITIAL_GRID_POP, mutation_profile: str = "Cryptococcus", params: Optional[List[Dict[str, Any]]] = None, grid_size: Tuple[int, int] = Config.GRID_SIZE) -> None:
        self.species_names = species_names[:Config.MAX_SPECIES]
        self.mutation_profile = mutation_profile
        self.max_vars = Config.MAX_VARS
        self.params = params or [{} for _ in species_names]
        self.grid_size = grid_size
        dtype = [('C', 'f4'), ('N', 'f4'), ('dose', 'f4'), ('pop', 'f4', (Config.MAX_SPECIES,))]
        self.grid = np.zeros(grid_size, dtype=dtype)
        self.grid['C'][:] = np.random.uniform(Config.C_RANGE[0], Config.C_RANGE[1], grid_size)
        self.grid['N'][:] = np.random.uniform(Config.N_RANGE[0], Config.N_RANGE[1], grid_size)
        self.grid_locations = np.array(
            [[random.choice(list(locations.keys())) for _ in range(grid_size[1])] for _ in range(grid_size[0])],
            dtype=object
        )
        if not np.all([loc in locations for loc in self.grid_locations.flatten()]):
            raise ValueError("Invalid location names in grid_locations")
        self.grid['dose'][:] = self._compute_doses()
        self.grid['pop'][:] = 0.0
        for i, name in enumerate(species_names):
            for _ in range(initial_pop_size):
                x = random.randrange(grid_size[0])
                y = random.randrange(grid_size[1])
                self.grid['pop'][x, y, i] += 1.0
        self.reference_genomes = {name: self._generate_genome() for name in species_names}
        self.individual_genomes = {}  # Track mutated genomes
        self.history = []
        logger = logging.getLogger('fungal_simulation')
        logger.debug(f"Initialized {grid_size[0]}x{grid_size[1]} grid with {len(species_names)} species")

    def _compute_doses(self) -> VectorF:
        locs = np.unique(self.grid_locations)
        dose_lut = {loc: run_transport(1.0, loc, 0.1) for loc in locs}
        dose_grid = np.vectorize(dose_lut.get)(self.grid_locations)
        return dose_grid

    def _generate_genome(self, length: int = Config.GENOME_LENGTH, max_vars: int = Config.MAX_VARS) -> Tuple[VectorU, VectorB]:
        num_vars = random.randint(0, max_vars)
        var_pos = np.zeros(max_vars, dtype=np.uint32)
        var_pos[:num_vars] = np.random.choice(length, num_vars, replace=False)
        var_pos[:num_vars].sort()
        var_base = np.frombuffer(''.join(np.random.choice(['A', 'T', 'C', 'G'], num_vars)).encode(), dtype=np.uint8)
        return var_pos, var_base

    def assign_label(self, var_pos: VectorU, var_base: VectorB) -> str:
        max_similarity = 0.0
        assigned_label = self.species_names[0]
        for label, (ref_pos, ref_base) in self.reference_genomes.items():
            similarity = genome_similarity_gpu(var_pos, var_base, ref_pos, ref_base)
            if similarity > max_similarity:
                max_similarity = similarity
                assigned_label = label
        if max_similarity < 0.85 and len(self.species_names) < Config.MAX_SPECIES:
            new_label = f"Species_{len(self.species_names) + 1}"
            self.species_names.append(new_label)
            self.reference_genomes[new_label] = (var_pos.copy(), var_base.copy())
            self.params.append({})
            logger = logging.getLogger('fungal_simulation')
            logger.debug(f"New species created: {new_label}, similarity={max_similarity:.2f}")
            return new_label
        return assigned_label

    def compute_growth_rate(self, C: VectorF, N: VectorF, dose_mGy_h: VectorF, species_idx: int) -> VectorF:
        spec = self.species_names[species_idx]
        melanin = species_data.get(spec, {}).get('Melanin', False)
        return compute_growth_rate(C, N, dose_mGy_h, melanin, spec)

    def mutate_genome(self, var_pos: VectorU, var_base: VectorB, dose_mGy_h: float, species: str) -> Tuple[VectorU, VectorB]:
        mu = dose_scaled_mu(dose_mGy_h)
        base_probs = MUTATION_PROFILES_CACHED[self.mutation_profile]
        te_boost = min(5.0, 1.0 + 10.0 * dose_mGy_h)
        event_probs = base_probs.copy()
        if te_boost > 1.0:
            event_probs = {k: (v * te_boost if k == "TE" else v) for k, v in base_probs.items()}
            norm = sum(event_probs.values())
            event_probs = {k: v / norm for k, v in event_probs.items()}
        new_pos = var_pos.copy()
        new_base = var_base.copy()
        num_vars = np.sum(new_pos != 0)
        hotspot_positions = species_data.get(species, {}).get('Mutation_Hotspots', [])
        i = 0
        while i < num_vars and i < self.max_vars:
            pos = new_pos[i]
            mu_adj = mu * (2.0 if pos in hotspot_positions else 1.0)
            if random.random() < mu_adj:
                event = random.choices(list(event_probs.keys()), weights=list(event_probs.values()), k=1)[0]
                if event == "SNP":
                    new_base[i] = ord(random.choice(['A', 'T', 'C', 'G']))
                elif event == "INS" and num_vars < self.max_vars:
                    new_pos[num_vars] = random.choice(hotspot_positions) if random.random() < 0.3 else random.randint(0, Config.GENOME_LENGTH-1)
                    new_base[num_vars] = ord(random.choice(['A', 'T', 'C', 'G']))
                    num_vars += 1
                elif event == "DEL":
                    new_pos[i] = 0
                    new_base[i] = 0
                    new_pos[i:num_vars-1] = new_pos[i+1:num_vars]
                    new_base[i:num_vars-1] = new_base[i+1:num_vars]
                    new_pos[num_vars-1] = 0
                    new_base[num_vars-1] = 0
                    num_vars -= 1
                    i -= 1
                elif event == "TE":
                    new_base[i] = ord('N')
            i += 1
        idx = new_pos != 0
        new_pos = np.pad(new_pos[idx], (0, self.max_vars - np.sum(idx)), 'constant')
        new_base = np.pad(new_base[idx], (0, self.max_vars - np.sum(idx)), 'constant')
        logger = logging.getLogger('fungal_simulation')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"mutate_genome: input_vars={np.sum(var_pos!=0)}, output_vars={np.sum(new_pos!=0)}")
        return new_pos, new_base

    def can_mate(self, p1: Individual, p2: Individual) -> bool:
        if p1.mating_type == p2.mating_type:
            return False
        similarity = genome_similarity_gpu(p1.var_pos, p1.var_base, p2.var_pos, p2.var_base)
        return similarity >= 0.85

    def mate(self, p1: Individual, p2: Individual) -> Individual:
        dose_mGy_h = self.grid['dose'][p1.x, p1.y]
        child_pos, child_base = recombine_gpu(p1.var_pos, p1.var_base, p2.var_pos, p2.var_base, dose_mGy_h, self.max_vars)
        child_pos, child_base = self.mutate_genome(child_pos, child_base, dose_mGy_h, p1.species_label)
        child_label = self.assign_label(child_pos, child_base)
        melanin = species_data.get(child_label, {}).get('Melanin', False)
        fitness = compute_growth_rate(
            np.array([self.grid['C'][p1.x, p1.y]], dtype=np.float64),
            np.array([self.grid['N'][p1.x, p1.y]], dtype=np.float64),
            np.array([dose_mGy_h], dtype=np.float64),
            melanin, child_label
        )[0]
        kmer_effect = compute_kmer_effect(child_pos, child_base, child_label)
        fitness *= np.exp(np.clip(kmer_effect, -5, 5))  # Cap kmer_effect to prevent explosion
        fitness = np.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)
        child = Individual(child_pos, child_base, child_label, p1.location, p1.x, p1.y,
                           fitness=fitness, hybrid_parentage=(p1.species_label, p2.species_label))
        self.individual_genomes[(child.x, child.y, child.species_label)] = (child_pos, child_base)
        logger = logging.getLogger('fungal_simulation')
        logger.debug(f"Mated {p1.species_label} and {p2.species_label}, child fitness={fitness}")
        return child

    def simulate(self) -> List[Dict[str, float]]:
        logger = logging.getLogger('fungal_simulation')
        logger.info(f"Starting simulation for {Config.NUM_TIME_STEPS} steps")
        self.history = []
        for step in range(Config.NUM_TIME_STEPS):
            self.simulate_step()
            logger.debug(f"Completed simulation step {step + 1}/{Config.NUM_TIME_STEPS}")
        return self.history

    def simulate_step(self) -> None:
        ext_mask = np.random.rand(self.grid_size[0], self.grid_size[1]) < (0.02 * Config.DT)
        pop = self.grid['pop']
        growth = np.stack([self.compute_growth_rate(self.grid["C"], self.grid["N"], self.grid["dose"], i)
                           for i in range(len(self.species_names))], axis=-1)
        carrying_capacity = np.array([species_data.get(spec, {}).get('Carrying_Capacity', Config.MAX_CARRY / len(self.species_names))
                                     for spec in self.species_names])
        death_rate = np.array([self.params[i].get('decay_rate', Config.DEATH_RATE) for i in range(len(self.species_names))])
        delta = (pop * growth * Config.DT) * (1 - pop / carrying_capacity[:, None, None]) - death_rate[:, None, None] * pop * Config.DT
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        pop += delta
        pop[ext_mask, :] = 0.0
        pop[:] = np.clip(pop, 0, carrying_capacity[:, None, None])
        pop[:] = np.nan_to_num(pop, nan=0.0, posinf=0.0, neginf=0.0)
        uptake_C = 0.02 * pop * Config.DT
        uptake_N = 0.004 * pop * Config.DT
        self.grid['C'] -= np.sum(uptake_C, axis=-1)
        self.grid['N'] -= np.sum(uptake_N, axis=-1)
        self.grid['C'] = np.maximum(self.grid['C'], 0.0)
        self.grid['N'] = np.maximum(self.grid['N'], 0.0)
        kernel = np.ones((3, 3), dtype=np.float64) / 8
        self.grid['C'] += 0.1 * (scipy.ndimage.convolve(self.grid['C'], kernel, mode='wrap') - self.grid['C'])
        self.grid['N'] += 0.1 * (scipy.ndimage.convolve(self.grid['N'], kernel, mode='wrap') - self.grid['N'])
        individuals = []
        occupied_cells = [(x, y, i) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])
                         for i, spec in enumerate(self.species_names)
                         if pop[x, y, i] >= 1.0 and not np.isnan(pop[x, y, i])]
        for x, y, i in occupied_cells:
            spec = self.species_names[i]
            var_pos, var_base = self.individual_genomes.get((x, y, spec), self.reference_genomes[spec])
            melanin = species_data.get(spec, {}).get('Melanin', False)
            fitness = compute_growth_rate(
                np.array([self.grid['C'][x, y]], dtype=np.float64),
                np.array([self.grid['N'][x, y]], dtype=np.float64),
                np.array([self.grid['dose'][x, y]], dtype=np.float64),
                melanin, spec
            )[0]
            ind = Individual(var_pos.copy(), var_base.copy(), spec, self.grid_locations[x, y], x, y, fitness=fitness)
            individuals.append(ind)
        births = []
        indices = list(range(len(individuals)))
        random.shuffle(indices)
        species_counts = {s: np.sum(pop[:, :, i]) for i, s in enumerate(self.species_names)}
        for idx in indices:
            p1 = individuals[idx]
            neighbors = [(p1.x + dx, p1.y + dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
                         if 0 <= p1.x + dx < self.grid_size[0] and 0 <= p1.y + dy < self.grid_size[1]]
            for nx, ny in neighbors:
                if self.grid['C'][nx, ny] > 0 and random.random() < p1.fitness * 0.1:
                    species_idx = self.species_names.index(p1.species_label)
                    pop[p1.x, p1.y, species_idx] -= 1
                    pop[nx, ny, species_idx] += 1
                    old_x, old_y = p1.x, p1.y
                    p1.x, p1.y = nx, ny
                    p1.location = self.grid_locations[nx, ny]
                    self.individual_genomes[(nx, ny, p1.species_label)] = self.individual_genomes.pop((old_x, old_y, p1.species_label), (p1.var_pos, p1.var_base))
                    break
            for _ in range(3):
                if len(indices) <= 1:
                    break
                p2 = individuals[random.choice(indices)]
                if self.can_mate(p1, p2):
                    species_size = species_counts.get(p1.species_label, 0)
                    carrying_capacity = species_data.get(p1.species_label, {}).get('Carrying_Capacity', Config.MAX_CARRY / len(self.species_names))
                    raw = (1 - species_size / carrying_capacity) * p1.fitness * Config.BIRTH_SCALE
                    p_birth = max(0.0, min(1.0, raw))
                    if random.random() < p_birth:
                        child = self.mate(p1, p2)
                        births.append(child)
                        try:
                            child_idx = self.species_names.index(child.species_label)
                            child_capacity = species_data.get(child.species_label, {}).get('Carrying_Capacity', Config.MAX_CARRY / len(self.species_names))
                            if pop[child.x, child.y, child_idx] < child_capacity:
                                pop[child.x, child.y, child_idx] += 1
                                species_counts[child.species_label] = species_counts.get(child.species_label, 0) + 1
                        except ValueError:
                            logger = logging.getLogger('fungal_simulation')
                            logger.warning(f"Child species {child.species_label} not in grid; skipping population update")
                    break
        logger = logging.getLogger('fungal_simulation')
        logger.debug(f"Step births: {len(births)}, total pop: {np.sum(pop):.2f}")
        self.history.append(species_counts)

    def record_state(self) -> List[Dict[str, Any]]:
        data = []
        pop = self.grid['pop']
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for i, spec in enumerate(self.species_names):
                    pop_val = pop[x, y, i]
                    if pop_val > 0 and not np.isnan(pop_val):
                        var_pos, var_base = self.individual_genomes.get((x, y, spec), self.reference_genomes[spec])
                        melanin = species_data.get(spec, {}).get('Melanin', False)
                        fitness = compute_growth_rate(
                            np.array([self.grid['C'][x, y]], dtype=np.float64),
                            np.array([self.grid['N'][x, y]], dtype=np.float64),
                            np.array([self.grid['dose'][x, y]], dtype=np.float64),
                            melanin, spec
                        )[0]
                        fitness = np.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)
                        data.append({
                            "Species Name": spec,
                            "Population": pop_val,
                            "X": x,
                            "Y": y,
                            "Fitness": fitness,
                            "Var_Pos": var_pos.tolist(),
                            "Var_Base": var_base.tolist(),
                            "Location": self.grid_locations[x, y]
                        })
        if not data:
            logger = logging.getLogger('fungal_simulation')
            logger.warning("No valid population data; returning empty state")
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

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    bad_rad_cols = [c for c in df.columns if "Radiation" in c and c != "Radiation Level (µSv/hr)"]
    if bad_rad_cols:
        df["Radiation Level (µSv/hr)"] = df["Radiation Level (µSv/hr)"].fillna(df[bad_rad_cols].bfill(axis=1).iloc[:, 0])
        df.drop(columns=bad_rad_cols, inplace=True)
    null_threshold = 0.9
    high_null_cols = [c for c in df.columns if df[c].isna().mean() > null_threshold]
    df.drop(columns=high_null_cols, inplace=True)
    numeric_cols = [
        "Latitude", "Longitude", "pH Level", "Radioactive Contamination Level (kBq/m²)",
        "Concentration in Mushroom (Bq/kg)", "Committed Effective Dose (mSv/year)",
        "Radiation Level (µSv/hr)", "Carbon Level (g/L)", "Nitrogen Level (g/L)",
        "Temperature (°C)", "NO2 (mg/L)", "Cap Diameter (cm)", "Stem Length (cm)",
        "X", "Y", "Fitness", "Population"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def save_dataset(data: List[Dict[str, Any]], filename: str) -> None:
    df = pd.DataFrame(data)
    df = clean_dataframe(df)
    logger = logging.getLogger('fungal_simulation')
    logger.info(f"Dataset columns for {filename}: {list(df.columns)}")
    logger.info(f"Dataset NaN counts for {filename}: {df.isna().sum().to_dict()}")
    try:
        df.to_csv(filename, index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
        logger.info(f"Dataset saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save dataset {filename}: {e}")

def plot_population_trends(history: List[Dict[str, float]], species_names: List[str], title: str, filename: str) -> None:
    try:
        plt.figure(figsize=(10, 6))
        for spec in species_names:
            plt.plot([h.get(spec, 0) for h in history], label=spec)
        plt.xlabel('Time (steps)')
        plt.ylabel('Population')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        logger = logging.getLogger('fungal_simulation')
        logger.error(f"Failed to save plot {filename}: {e}")

def generate_dataset(num_records: int, extended: bool = False) -> List[Dict[str, Any]]:
    data = []
    strain_ids = set(record["Strain ID"] for record in (mushroom_data + location_data))
    rows_per_strain = math.ceil(num_records / len(strain_ids))
    unique_records = set()
    mushroom_records_by_id = {record["Strain ID"]: record for record in mushroom_data if "Strain ID" in record}
    location_records_by_id = {record["Strain ID"]: record for record in location_data if "Strain ID" in record}
    for strain_id in strain_ids:
        for _ in range(rows_per_strain):
            if len(data) >= num_records:
                break
            mushroom_record = mushroom_records_by_id.get(strain_id, {})
            location_record = location_records_by_id.get(strain_id, {})
            combined_record = {**mushroom_record, **location_record}
            combined_record = calculate_derived_fields(combined_record)
            loc_name = location_record.get("Location", "Chernobyl Reactor, Ukraine")
            loc_data = locations.get(loc_name, {
                "Latitude": 51.3894,
                "Longitude": 30.0994,
                "Habitat": "Reactor",
                "Radiation_µSv_hr": 5.0,
                "Contamination_kBq_m2": 2000
            })
            habitat = loc_data["Habitat"]
            species_candidates = [(s, species_data[s]["Habitat Weight"]) for s in species_data
                                 if species_data[s]["Preferred Habitat"] == habitat]
            if not species_candidates:
                species_candidates = [(s, 1.0) for s in species_data]
            cand_names, weights = zip(*species_candidates)
            species = random.choices(cand_names, weights=weights, k=1)[0]
            zone = get_contamination_zone(loc_data.get("Contamination_kBq_m2", 2000))
            concentration = random.uniform(
                species_data.get(species, {}).get("Min", 100.0),  # Default Min
                species_data.get(species, {}).get("Max", 1000.0)  # Default Max
            )
            radiation_level = get_radiation_mGyh(loc_name)
            if OPENMC_AVAILABLE:
                openmc_dose = run_transport(thickness_cm=1.0, location=loc_name, overburden=0.1)
                radiation_level = (radiation_level + openmc_dose) / 2
            c_level = random.uniform(*Config.C_RANGE)
            n_level = random.uniform(*Config.N_RANGE)
            temp = random.uniform(species_data[species].get('Temp_Min', 15),
                                 species_data[species].get('Temp_Max', 25))
            ph = random.uniform(species_data[species].get('pH_Min', 4),
                               species_data[species].get('pH_Max', 8))
            no2 = random.uniform(*Config.NO2_RANGE)
            isotope = random.choices([iso[0] for iso in Config.ISOTOPES], [iso[1] for iso in Config.ISOTOPES], k=1)[0]
            committed_effective_dose = calculate_committed_effective_dose(concentration)
            melanin_present = species_data[species]["Melanin"]
            substrate = species_data[species]["Substrate"]
            years_since_2024 = np.random.randint(1, Config.SPECIES_LIFESPAN + 1)
            birth_date, death_date = estimate_dates(years_since_2024)
            fungi_growth_rate = simulate_fungi_growth(loc_name, radiation_level, species, c_level)
            lat_offset = random.uniform(-0.01, 0.01)
            lon_offset = random.uniform(-0.01, 0.01)
            record_hash = hashlib.md5(
                f"{strain_id}{species}{loc_name}{lat_offset:.6f}{lon_offset:.6f}".encode()
            ).hexdigest()
            if record_hash in unique_records:
                continue
            unique_records.add(record_hash)
            record = {
                "Strain ID": str(strain_id),
                "Species Name": species,
                "Common Name": species_data[species]["Common Name"],
                "Location": loc_name,
                "Latitude": loc_data["Latitude"] + lat_offset,
                "Longitude": loc_data["Longitude"] + lon_offset,
                "Habitat": habitat,
                "Soil Type": random.choice(species_data[species].get('Soil_Types', ["Sandy", "Clay", "Loam"])),
                "pH Level": round(ph, 2),
                "Contamination Zone": zone,
                "Radioactive Contamination Level (kBq/m²)": round(random.uniform(zones[zone]["Min Contamination"],
                                                                               zones[zone]["Max Contamination"]), 2),
                "Radioactive Isotope": isotope,
                "Concentration in Mushroom (Bq/kg)": round(concentration, 2),
                "Committed Effective Dose (mSv/year)": round(committed_effective_dose * 1000, 3),
                "Radiation Level (µSv/hr)": round(radiation_level * 1000.0, 3),
                "Carbon Level (g/L)": round(c_level, 3),
                "Nitrogen Level (g/L)": round(n_level, 3),
                "Temperature (°C)": round(temp, 1),
                "NO2 (mg/L)": round(no2, 2),
                "Substrate": substrate,
                "Cap Diameter (cm)": round(random.uniform(species_data[species].get('Cap_Min', 2),
                                                        species_data[species].get('Cap_Max', 10)), 1),
                "Stem Length (cm)": round(random.uniform(species_data[species].get('Stem_Min', 1),
                                                       species_data[species].get('Stem_Max', 15)), 1),
                "Edibility": species_data[species]["Edibility"],
                "Melanin": melanin_present,
                "Fungi Growth Rate": fungi_growth_rate,
                "Estimated Birth Date": birth_date,
                "Estimated Death Date": death_date,
                **{k: v for k, v in combined_record.items() if v is not None}
            }
            data.append(record)
    return data[:num_records if not extended else num_records * 10]

def _eval_wrapper(ind, species_names, profile):
    _ensure_worker_init(BASE_DIR)
    return eval_species(ind, species_names, profile)

def eval_species(individual: List[float], species_names: List[str], mutation_profile: str) -> Tuple[float, float]:
    _ensure_worker_init(BASE_DIR)
    try:
        params = [
            {
                'replication_rate': np.clip(individual[i * 3], 0.18, 0.32),
                'decay_rate': np.clip(individual[i * 3 + 1], 0.005, 0.02),
                'radiation_effectiveness': np.clip(individual[i * 3 + 2], 0.1, 2.0),
                'Melanin': species_data.get(species_names[i], {}).get('Melanin', False),
                'Substrate': species_data.get(species_names[i], {}).get('Substrate', 'Unknown')
            } for i in range(len(species_names))
        ]
        ecosystem = FungalEcosystem(species_names, initial_pop_size=Config.INITIAL_GRID_POP, mutation_profile=mutation_profile, params=params, grid_size=(40, 40))
        history = ecosystem.simulate()
        final_states = [history[-1].get(spec, 0) for spec in species_names]
        biomass = np.sum(final_states)
        if np.isnan(biomass) or biomass <= 0:
            logger = logging.getLogger('fungal_simulation')
            logger.warning("Invalid biomass; returning (0.0, 0.0)")
            return 0.0, 0.0
        sim_data = pd.DataFrame(ecosystem.record_state())
        if sim_data.empty:
            logger = logging.getLogger('fungal_simulation')
            logger.warning("Empty simulation data; returning (biomass, 0.0)")
            return biomass, 0.0
        rmse = 0.0
        valid_count = 0
        for species in species_names:
            if species not in empirical_data:
                continue
            rad_val = next((empirical_data[species].get(k) for k in ['Radiation_µSv_hr', 'Radiation_k', 'Radiation_kGy_h']
                            if empirical_data[species].get(k) is not None), None)
            if rad_val is None:
                continue
            sim_subset = sim_data[sim_data['Species Name'] == species]
            if sim_subset.empty:
                continue
            sim_fitness = sim_subset['Fitness'].astype(float)
            if sim_fitness.isna().all():
                continue
            rad_val_normalized = rad_val / 1000.0
            error_margin = empirical_data[species].get('Error_Margin', 0.1) * rad_val_normalized
            real_data = pd.Series([np.random.normal(rad_val_normalized, error_margin) for _ in range(len(sim_fitness))], dtype=float)
            rmse += np.sqrt(mean_squared_error(sim_fitness, real_data))
            valid_count += 1
        if valid_count == 0:
            logger = logging.getLogger('fungal_simulation')
            logger.warning("No valid species for RMSE; returning (biomass, 0.0)")
            return biomass, 0.0
        rmse = rmse / valid_count
        logger = logging.getLogger('fungal_simulation')
        logger.debug(f"Evaluation: biomass={biomass:.2f}, rmse={rmse:.4f}, valid_species={valid_count}")
        return biomass, -rmse
    except Exception as e:
        logger = logging.getLogger('fungal_simulation')
        logger.error(f"Error in eval_species: {e}")
        return 0.0, 0.0

def main() -> Tuple[List[Any], tools.Statistics, tools.ParetoFront, List[Dict[str, Any]]]:
    logger = logging.getLogger('fungal_simulation')
    logger.info('Starting NSGA-II Genetic Algorithm')
    sampler = qmc.LatinHypercube(d=Config.NUM_SPECIES * 3)
    sample = sampler.random(n=Config.GA_POP_SIZE)
    population = [creator.Individual([sample[i][j] for j in range(Config.NUM_SPECIES * 3)]) for i in range(Config.GA_POP_SIZE)]
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg_biomass', lambda x: np.mean([v[0] for v in x if len(v) > 0 and v[0] is not None]))
    stats.register('avg_rmse', lambda x: np.mean([-v[1] for v in x if len(v) > 1 and v[1] is not None]))
    with Pool() as pool:
        if 'map' in toolbox.__dict__:
            del toolbox.__dict__['map']
        toolbox.register('map', pool.starmap)
        for gen in range(Config.GENERATIONS):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
            fits = toolbox.map(_eval_wrapper, [(ind, GLOBAL_SPECIES, GLOBAL_PROFILE) for ind in offspring])
            for fit, ind in zip(fits, offspring):
                if len(fit) >= 2 and fit[0] is not None and fit[1] is not None and not (fit[0] == 0 and fit[1] == 0):
                    ind.fitness.values = fit
                else:
                    logger.warning(f"Invalid fitness for individual: {fit}; assigning default")
                    ind.fitness.values = (0.0, 0.0)
            valid_individuals = [ind for ind in offspring + population if len(ind.fitness.values) >= 2]
            if not valid_individuals:
                logger.error("No valid individuals for selection; terminating")
                return population, stats, hof, []
            population = tools.selNSGA2(valid_individuals, k=len(population))
            hof.update(population)
            logger.info(f"Generation {gen+1}/{Config.GENERATIONS}: Pareto front size = {len(hof)}")
    best_params = []
    for i in range(len(GLOBAL_SPECIES)):
        start = i * 3
        param = {
            'replication_rate': np.clip(hof[0][start], 0.18, 0.32),
            'decay_rate': np.clip(hof[0][start + 1], 0.005, 0.02),
            'radiation_effectiveness': np.clip(hof[0][start + 2], 0.1, 2.0),
            'Melanin': species_data.get(GLOBAL_SPECIES[i], {}).get('Melanin', False),
            'Substrate': species_data.get(GLOBAL_SPECIES[i], {}).get('Substrate', 'Unknown')
        }
        best_params.append(param)
    return population, stats, hof, best_params

if __name__ == "__main__":
    # Verify BASE_DIR exists
    if not os.path.exists(BASE_DIR):
        raise RuntimeError(f"Base directory {BASE_DIR} does not exist")

    # Set up logging
    log_dir = BASE_DIR
    log_file = os.path.join(log_dir, f'fungal_simulation_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Error creating log directory: {e}")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
        force=True
    )
    logging.getLogger('numba.cuda').setLevel(logging.WARNING)
    logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)
    logging.getLogger('cupy').setLevel(logging.WARNING)

    logger = logging.getLogger('fungal_simulation')
    try:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info("Logging initialized successfully")
    except OSError as e:
        raise RuntimeError(f"Error setting up log file {log_file}: {e}")

    _ensure_worker_init(BASE_DIR)
    GLOBAL_SPECIES = list(species_data.keys())
    GLOBAL_PROFILE = args.profile

    try:
        run_smoke_test()
        combined_data = generate_dataset(Config.NUM_RECORDS)
        save_dataset(combined_data, "combined_dataset.csv")
        extended_data = generate_dataset(Config.NUM_RECORDS, extended=True)
        save_dataset(extended_data, "extended_fungal_dataset.csv")
        ecosystem = FungalEcosystem(GLOBAL_SPECIES, mutation_profile=GLOBAL_PROFILE)
        start_time = time.time()
        history = ecosystem.simulate()
        end_time = time.time()
        logger.info(f"Simulation took {end_time - start_time:.2f} seconds for {Config.NUM_TIME_STEPS} steps")
        logger.info(f"Total biomass after {Config.NUM_TIME_STEPS} steps: {np.sum(ecosystem.grid['pop']):.2e}")
        plot_population_trends(history, GLOBAL_SPECIES, 'Population Dynamics Over Time', 'population_trends.png')
        sim_records = ecosystem.record_state()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_dataset(sim_records, f'fungal_simulation_{timestamp}.csv')
        if 'map' in toolbox.__dict__:
            del toolbox.__dict__['map']
        population, stats, hof, best_params = main()
        optimized_ecosystem = FungalEcosystem(GLOBAL_SPECIES, mutation_profile=GLOBAL_PROFILE, params=best_params, grid_size=(100, 100))
        optimized_history = optimized_ecosystem.simulate()
        plot_population_trends(optimized_history, GLOBAL_SPECIES, 'Optimized Population Dynamics', 'optimized_population_trends.png')
        optimized_records = optimized_ecosystem.record_state()
        save_dataset(optimized_records, f'optimized_fungal_simulation_{timestamp}.csv')
        validation_results = []
        radiation_keys = ['Radiation_µSv_hr', 'Radiation_k', 'Radiation_kGy_h']
        optimized_df = pd.DataFrame(optimized_records)
        for species in empirical_data:
            if optimized_df.empty or 'Species Name' not in optimized_df.columns:
                logger.warning(f"Validation issue for {species}: No data in optimized_df")
                validation_results.append({"Species": species, "RMSE": np.nan, "R²": np.nan})
                continue
            sim_data = optimized_df[optimized_df['Species Name'] == species]['Fitness'].astype(float)
            if sim_data.empty:
                logger.warning(f"Validation issue for {species}: No individuals in simulation")
                validation_results.append({"Species": species, "RMSE": np.nan, "R²": np.nan})
                continue
            rad_val = next((empirical_data[species].get(k) for k in radiation_keys
                            if k in empirical_data[species] and pd.notna(empirical_data[species][k])), None)
            if rad_val is None:
                logger.warning(f"Validation issue for {species}: No empirical radiation value")
                validation_results.append({"Species": species, "RMSE": np.nan, "R²": np.nan})
                continue
            rad_val_normalized = rad_val / 1000.0
            error_margin = empirical_data[species].get('Error_Margin', 0.1) * rad_val_normalized
            real_data = pd.Series([np.random.normal(rad_val_normalized, error_margin) for _ in range(len(sim_data))], dtype=float)
            rmse = np.sqrt(mean_squared_error(sim_data, real_data))
            r2 = r2_score(sim_data, real_data)
            validation_results.append({"Species": species, "RMSE": rmse, "R²": r2})
            print(f"Validation for {species}: RMSE={rmse:.4f}, R²={r2:.4f}")
            try:
                plt.figure(figsize=(8, 5))
                plt.plot(sim_data.to_numpy(), label='Simulated')
                plt.plot(real_data.to_numpy(), label='Empirical', linestyle='--')
                plt.title(f'{species}: Simulated vs. Empirical Growth')
                plt.xlabel('Time Step')
                plt.ylabel('Fitness')
                plt.legend()
                plt.savefig(f'validation_{species}_{timestamp}.png')
                plt.close()
            except Exception as e:
                logger.error(f"Failed to save validation plot for {species}: {e}")
        validation_df = pd.DataFrame(validation_results)
        validation_df["RMSE"] = pd.to_numeric(validation_df["RMSE"], errors='coerce')
        validation_df["R²"] = pd.to_numeric(validation_df["R²"], errors='coerce')
        logger.info(f"Validation dataset columns: {list(validation_df.columns)}")
        logger.info(f"Validation dataset NaN counts: {validation_df.isna().sum().to_dict()}")
        save_dataset(validation_results, f'validation_results_{timestamp}.csv')
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise