# OpenMC Integration Architecture for QuantumBioSim

## HOW OpenMC is Integrated

### Integration Flow Diagram

```
main.py (--big-bang)
    ↓
src/pygad_engine.py::run_big_bang_pygad()
    ↓
[TWO-LAYER INTEGRATION]
    ↓
┌─────────────────────────────────────────────────┐
│ Layer 1: Static XML Analysis (ALWAYS RUNS)     │
│ src/openmc_adapter.py                           │
├─────────────────────────────────────────────────┤
│ - Parses openmc_static/materials.xml            │
│ - Extracts material properties (density, comp) │
│ - Calculates environmental impacts:             │
│   * radiation_shielding (from density)         │
│   * radiation_source (from Cs137, U isotopes)  │
│   * oxygen_level (from O16 content)            │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Layer 2: Real Monte Carlo (IF AVAILABLE)       │
│ src/real_openmc_model.py                        │
├─────────────────────────────────────────────────┤
│ - Uses official OpenMC Python API               │
│ - Builds full 3D geometry model                │
│ - Runs Monte Carlo neutron transport           │
│ - Returns HEATING tally (eV/source particle)   │
│ - Modifies shielding impact if heating > 1000  │
└─────────────────────────────────────────────────┘
    ↓
Environmental Impact Dictionary
{
  'radiation_shielding': float,  # Higher = better protection
  'radiation_source': float,     # Higher = more radiation
  'oxygen_level': float          # Higher = more O2 available
}
    ↓
Applied to Big Bang Timeline Epochs
(Hadean → Archean → Proterozoic → Paleozoic → Mesozoic → Cenozoic → Anthropocene)
    ↓
Radiation & O2 levels affect species fitness
    ↓
PyGAD Genetic Algorithm
    ↓
Species evolve based on REAL radiation physics
```

## WHY We Integrated OpenMC

### 1. Scientific Accuracy

**Problem:** Generic radiation models are unrealistic
**Solution:** OpenMC provides publication-quality nuclear physics

- Actual neutron/gamma transport equations
- Real cross-section data (444 nuclides)
- Monte Carlo statistical accuracy

### 2. Material-Driven Evolution

**Concept:** Environment shapes biology
**Implementation:**

- Soil composition (Si, O, Al) provides shielding
- Radioactive isotopes (Cs137) create selection pressure
- Oxygen content affects metabolic fitness

### 3. Two-Layer Fallback Design

**Why two layers?**

**Layer 1 (XML Static):**

- Fast: No simulation overhead
- Reliable: Always works
- Basic: Material properties → simple calculations

**Layer 2 (Real OpenMC):**

- Accurate: Full Monte Carlo transport
- Expensive: Requires executable + nuclear data
- Optional: Falls back gracefully if unavailable

### 4. Evolutionary Realism

**Timeline Integration:**

```python
# Hadean Epoch (4.6-4 Ga)
radiation_multiplier = 5.0 / (shield + 0.1)  # High radiation
oxygen_multiplier = 0.0  # No O2 yet

# Modern Era
radiation_multiplier = 1.0 / (shield + 0.1)  # Protected
oxygen_multiplier = oxy_source * 0.5  # Rich O2
```

OpenMC's `radiation_shielding` directly reduces radiation stress
→ Affects fitness function
→ Drives natural selection

### 5. Future Expandability

**Current:** Basic soil shielding model
**Future Potential:**

- Multiple radiation sources (cosmic, decay chains)
- Complex geometries (caves, buildings)
- Time-dependent sources (solar events)
- Dose rate calculations for different organs

## What Triggers OpenMC Integration

**Command:** `python main.py --big-bang`
**Does NOT trigger:** `--run-sim` (uses different simulation path)

**Execution Path:**

1. `main.py` → calls `run_big_bang_pygad()`
2. Lines 108-150 in `pygad_engine.py` execute
3. Layer 1 always runs (static XML)
4. Layer 2 attempts to run (may fail gracefully)
5. Impact values fed into timeline (lines 156-159)

## Current Status

**Your Last Run:**

```
✓ Layer 1: Static XML loaded successfully
✓ Environmental impacts calculated
✓ Layer 2: Model built (XMLs generated)
✗ Layer 2: Execution failed (openmc.exe not in PATH)
✓ Fallback: Using static data
✓ Timeline simulation completed
```

**To Enable Full Integration:**
Add `openmc.exe` to PATH or use Python API's internal runner.

## Key Files

1. `src/pygad_engine.py` (lines 104-151) - Integration logic
2. `src/real_openmc_model.py` - Monte Carlo model
3. `src/openmc_adapter.py` - Static XML parser
4. `openmc_static/materials.xml` - Reference data
5. Generated: `materials.xml`, `geometry.xml`, `settings.xml`, `tallies.xml`

---

**Summary:** OpenMC provides REAL nuclear physics to evolutionary simulation, making radiation a scientifically accurate selection pressure rather than an arbitrary parameter.
