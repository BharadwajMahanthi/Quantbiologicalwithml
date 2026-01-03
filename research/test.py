from pathlib import Path
from collections import Counter
import json, pandas as pd

BASE = Path(r"C:\Users\mbpd1\Downloads\pp-mywork\Quantbiologicalwithml")          # wherever you unpacked the zip
species   = json.loads(Path(BASE, "corrected_species_data.json").read_text())
mushrooms = json.loads(Path(BASE, "corrected_mushroom_data.json").read_text())
locations = json.loads(Path(BASE, "corrected_location_data.json").read_text())

# 1) every Strain ID has species & location
sid_species   = {m["Strain ID"] for m in mushrooms}
sid_locations = {l["Strain ID"] for l in locations}
assert sid_species == sid_locations, "Mismatch between mushroom & location files"

# 2) no duplicate lat/lon
latlon = [(round(l["Latitude"],5), round(l["Longitude"],5)) for l in locations]
dups = [k for k,v in Counter(latlon).items() if v>1]
assert not dups, "Still have duplicate coordinates"

# 3) cap ranges numeric & sensible
bad = [sp for sp,b in species.items() if not (isinstance(b["CapDiameter_cm"], list) and len(b["CapDiameter_cm"])==2)]
assert not bad, f"Missing caps for: {bad}"
