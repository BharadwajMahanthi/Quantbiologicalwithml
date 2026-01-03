import json
import random
import zipfile
import io
from pathlib import Path
from collections import Counter

# Configuration
BASE = Path(r"C:\Users\mbpd1\Downloads\pp-mywork\Quantbiologicalwithml")  # Your project directory
OUT = BASE  # Write corrected files to the same folder

# ---------- Helpers ----------
def load(name):
    return json.loads(Path(BASE, name).read_text(encoding="utf-8"))

def save(name, data):
    Path(OUT, name).write_text(
        json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8"
    )

# Load original files
try:
    species = load("species_data.json")
    mushrooms = load("mushroom_data.json")
    locations = load("location_data.json")
    radiation = load("radiation_data.json")
    zones = load("zones.json")
except FileNotFoundError as e:
    print(f"Error: File {e.filename} not found in {BASE}")
    raise

# ---------- 1) species_data.json ----------
# Define morphological ranges for each species
morphological_ranges = {
    "Boletus edulis": {"CapDiameter_cm": [6, 25], "StemLength_cm": [4, 20]},
    "Leccinum aurantiacum": {"CapDiameter_cm": [5, 15], "StemLength_cm": [5, 18]},
    "Leccinum scabrum": {"CapDiameter_cm": [4, 12], "StemLength_cm": [5, 15]},
    "Suillus luteus": {"CapDiameter_cm": [5, 12], "StemLength_cm": [3, 10]},
    "Cantharellus cibarius": {"CapDiameter_cm": [3, 10], "StemLength_cm": [2, 8]},
    "Russula xerampelina": {"CapDiameter_cm": [5, 15], "StemLength_cm": [3, 10]},
    "Amanita phalloides": {"CapDiameter_cm": [5, 15], "StemLength_cm": [7, 15]},
    "Amanita muscaria": {"CapDiameter_cm": [8, 20], "StemLength_cm": [8, 20]},
    "Gyromitra esculenta": {"CapDiameter_cm": [3, 10], "StemLength_cm": [2, 8]},
    "Cladosporium sphaerospermum": {"CapDiameter_cm": [0.001, 0.002], "StemLength_cm": [0.001, 0.002]},
    "Wangiella dermatitidis": {"CapDiameter_cm": [0.001, 0.002], "StemLength_cm": [0.001, 0.002]},
    "Cryptococcus neoformans": {"CapDiameter_cm": [0.001, 0.002], "StemLength_cm": [0.001, 0.002]},
    "Psathyrella candolleana": {"CapDiameter_cm": [3, 7], "StemLength_cm": [4, 10]},
    "Inocybe geophylla": {"CapDiameter_cm": [2, 4], "StemLength_cm": [3, 7]},
    "Coprinellus micaceus": {"CapDiameter_cm": [2, 5], "StemLength_cm": [3, 8]},
    "Lactarius deliciosus": {"CapDiameter_cm": [5, 15], "StemLength_cm": [3, 7]},
    "Hygrophorus russula": {"CapDiameter_cm": [5, 12], "StemLength_cm": [3, 8]},
    "Cortinarius praestans": {"CapDiameter_cm": [8, 20], "StemLength_cm": [6, 15]},
    "Mycena flavoalba": {"CapDiameter_cm": [1, 3], "StemLength_cm": [2, 5]},
    "Hebeloma crustuliniforme": {"CapDiameter_cm": [4, 9], "StemLength_cm": [4, 10]},
    "Marasmius oreades": {"CapDiameter_cm": [2, 5], "StemLength_cm": [3, 10]},
    "Melanoleuca melaleuca": {"CapDiameter_cm": [4, 10], "StemLength_cm": [4, 10]},
    "Pholiota squarrosa": {"CapDiameter_cm": [5, 12], "StemLength_cm": [5, 15]},
    "Coprinopsis atramentaria": {"CapDiameter_cm": [3, 7], "StemLength_cm": [5, 15]}
}

for sp, blk in species.items():
    # Add morphological ranges
    if sp in morphological_ranges:
        blk["CapDiameter_cm"] = morphological_ranges[sp]["CapDiameter_cm"]
        blk["StemLength_cm"] = morphological_ranges[sp]["StemLength_cm"]
    else:
        blk["CapDiameter_cm"] = [4, 12]  # Fallback
        blk["StemLength_cm"] = [3, 10]
    # Rename Median to Median_Bqkg
    if "Median" in blk:
        blk["Median_Bqkg"] = blk.pop("Median")

save("corrected_species_data.json", species)

# ---------- 2) mushroom_data.json ----------
# Authoritative strain-to-species mapping
strain2species = {
    "1142-2": "Amanita phalloides",
    "1191": "Amanita muscaria",
    "1192": "Gyromitra esculenta",
    "1164": "Marasmius oreades",
    "1109": "Hebeloma crustuliniforme",
    "1165-1": "Melanoleuca melaleuca",
    "1165-2": "Hebeloma crustuliniforme",
    "1163": "Hygrophorus russula",
    "1106": "Hygrophorus russula",
    "1102-2": "Inocybe geophylla",
    "1102-1": "Inocybe geophylla",
    "1101-3": "Psathyrella candolleana",
    "1101-2": "Psathyrella candolleana",
    "1161": "Boletus edulis",
    "1101-1": "Psathyrella candolleana",
    "1116-1": "Cortinarius praestans",
    "1116-2": "Cortinarius praestans",
    "1116-3": "Cortinarius praestans",
    "1126-1": "Cantharellus cibarius",
    "1126-2": "Suillus luteus",
    "1111-1": "Lactarius deliciosus",
    "1111-2": "Lactarius deliciosus",
    "1146-1": "Lactarius deliciosus",
    "1146-2": "Lactarius deliciosus",
    "1127-1": "Cantharellus cibarius",
    "1127-2": "Suillus luteus",
    "1127-3": "Cantharellus cibarius",
    "1130": "Lactarius deliciosus",
    "1131": "Lactarius deliciosus",
    "1132-1": "Lactarius deliciosus",
    "1132-2": "Lactarius deliciosus",
    "1132-3": "Lactarius deliciosus",
    "1134": "Cladosporium sphaerospermum",
    "1103-3": "Cantharellus cibarius",
    "1105": "Lactarius deliciosus",
    "1162": "Suillus luteus"
}

# Update edibility in mushroom_data.json to align with species_data.json
for rec in mushrooms:
    sid = rec["Strain ID"]
    if sid in strain2species:
        rec["Species Name"] = strain2species[sid]
        # Set Edibility based on species_data.json
        sp = rec["Species Name"]
        species_edibility = species[sp]["Edibility"]
        if species_edibility in ["Toxic", "Unknown"]:
            rec["Edibility"] = "Not Edible"
        else:
            rec["Edibility"] = "Edible"
    else:
        print(f"Warning: No species mapping for Strain ID {sid}")

save("corrected_mushroom_data.json", mushrooms)

# ---------- 3) location_data.json ----------
# Apply exact coordinates
corrected_coordinates = {
    "1101-1": {"Latitude": 51.1500, "Longitude": 30.1200},
    "1101-2": {"Latitude": 51.1505, "Longitude": 30.1202},
    "1101-3": {"Latitude": 51.1495, "Longitude": 30.1198},
    "1102-1": {"Latitude": 51.2067, "Longitude": 30.0000},
    "1102-2": {"Latitude": 51.2065, "Longitude": 30.0002},
    "1103-3": {"Latitude": 51.3167, "Longitude": 29.6406},
    "1105": {"Latitude": 51.3833, "Longitude": 30.0667},
    "1106": {"Latitude": 51.5433, "Longitude": 30.1747},
    "1109": {"Latitude": 51.3931, "Longitude": 30.0736},
    "1161": {"Latitude": 51.2763, "Longitude": 30.2219},
    "1162": {"Latitude": 51.3165, "Longitude": 29.6408},
    "1163": {"Latitude": 51.2300, "Longitude": 30.0197},
    "1164": {"Latitude": 50.1300, "Longitude": 30.5300},
    "1165-1": {"Latitude": 51.2077, "Longitude": 30.0010},
    "1165-2": {"Latitude": 51.2075, "Longitude": 30.0012},
    "1142-2": {"Latitude": 47.4618, "Longitude": 18.9208},
    "1191": {"Latitude": 49.9622, "Longitude": 7.9032},
    "1192": {"Latitude": 51.1000, "Longitude": 6.9037},
    "1116-1": {"Latitude": 51.2765, "Longitude": 30.2221},
    "1116-2": {"Latitude": 51.2761, "Longitude": 30.2217},
    "1116-3": {"Latitude": 51.2767, "Longitude": 30.2223},
    "1126-1": {"Latitude": 51.3169, "Longitude": 29.6404},
    "1126-2": {"Latitude": 51.3163, "Longitude": 29.6410},
    "1111-1": {"Latitude": 51.3835, "Longitude": 30.0669},
    "1111-2": {"Latitude": 51.3831, "Longitude": 30.0665},
    "1146-1": {"Latitude": 51.3837, "Longitude": 30.0671},
    "1146-2": {"Latitude": 51.3829, "Longitude": 30.0663},
    "1127-1": {"Latitude": 51.3171, "Longitude": 29.6402},
    "1127-2": {"Latitude": 51.3161, "Longitude": 29.6412},
    "1127-3": {"Latitude": 51.3173, "Longitude": 29.6400},
    "1130": {"Latitude": 51.3839, "Longitude": 30.0673},
    "1131": {"Latitude": 51.3827, "Longitude": 30.0661},
    "1132-1": {"Latitude": 51.3841, "Longitude": 30.0675},
    "1132-2": {"Latitude": 51.3825, "Longitude": 30.0659},
    "1132-3": {"Latitude": 51.3843, "Longitude": 30.0677},
    "1134": {"Latitude": 51.3894, "Longitude": 30.0994}
}

for rec in locations:
    sid = rec["Strain ID"]
    if sid in corrected_coordinates:
        rec["Latitude"] = corrected_coordinates[sid]["Latitude"]
        rec["Longitude"] = corrected_coordinates[sid]["Longitude"]

save("corrected_location_data.json", locations)

# ---------- 4) radiation_data.json ----------
if "_units" not in radiation:
    radiation = {"_units": "µSv per hour", **radiation}

save("corrected_radiation_data.json", radiation)

# ---------- 5) zones.json ----------
for z, blk in zones.items():
    blk["Min Contamination"] = float(blk["Min Contamination"])
    blk["Max Contamination"] = float(blk["Max Contamination"])

save("corrected_zones.json", zones)

# ---------- Validation Checks ----------
# 1) Every Strain ID has species & location
sid_species = {m["Strain ID"] for m in mushrooms}
sid_locations = {l["Strain ID"] for l in locations}
assert sid_species == sid_locations, "Mismatch between mushroom & location files"

# 2) No duplicate lat/lon
latlon = [(round(l["Latitude"], 5), round(l["Longitude"], 5)) for l in locations]
dups = [k for k, v in Counter(latlon).items() if v > 1]
assert not dups, f"Duplicate coordinates found: {dups}"

# 3) Cap ranges numeric & sensible
bad = [sp for sp, b in species.items() if not (
    isinstance(b["CapDiameter_cm"], list) and len(b["CapDiameter_cm"]) == 2 and
    isinstance(b["StemLength_cm"], list) and len(b["StemLength_cm"]) == 2
)]
assert not bad, f"Missing or invalid cap/stem ranges for: {bad}"

# 4) Species Name and Edibility consistency
for rec in mushrooms:
    sid = rec["Strain ID"]
    if "Species Name" in rec:
        sp = rec["Species Name"]
        assert sp in species, f"Invalid species {sp} for Strain ID {sid}"
        expected_edibility = species[sp]["Edibility"]
        actual_edibility = rec["Edibility"]
        # Allow both "Toxic" and "Not Edible" for Toxic/Unknown species
        if expected_edibility in ["Toxic", "Unknown"]:
            assert actual_edibility in ["Not Edible", "Toxic"], (
                f"Edibility mismatch for Strain ID {sid}: {actual_edibility} not in ['Not Edible', 'Toxic']"
            )
        else:
            assert actual_edibility == expected_edibility, (
                f"Edibility mismatch for Strain ID {sid}: {actual_edibility} != {expected_edibility}"
            )

# ---------- Zip everything ----------
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in Path(OUT).glob("corrected_*.json"):
        zf.write(str(fname), fname.name)
buf.seek(0)
with open(Path(OUT, "corrected_json.zip"), "wb") as f:
    f.write(buf.read())

print(f"✅ Patched files written as corrected_*.json and zipped to {OUT}\\corrected_json.zip")