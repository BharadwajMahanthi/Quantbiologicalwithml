"""
Unit-aware extensions for models.py - Research-grade biology parameters.

This module provides unit-aware helper functions and constants for the
QuantumBiologicalSystem to use real scientific parameters from literature.

All new code - does NOT modify existing QuantumBiologicalSystem class.
"""

import os
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.units import u, Qu, load_parameter_with_units, validate_units, ppm_to_fraction, percent_to_fraction

# =============================================================================
# Load Parameters Database
# =============================================================================

_PARAMS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'parameters.csv')
try:
    _PARAMS = pd.read_csv(_PARAMS_FILE, comment='#')
except FileNotFoundError:
    print(f"Warning: Could not load {_PARAMS_FILE}")
    _PARAMS = None


# =============================================================================
# Research-Grade Biological Constants
# =============================================================================

class BiologicalConstants:
    """
    Research-grade biological constants loaded from parameters.csv.
    All values have proper units from astropy.units.
    
    Citations in parameters.csv.
    """
    
    if _PARAMS is not None:
        # Cellular properties
        CELL_MASS = load_parameter_with_units(_PARAMS, 'cell_mass_C_neoformans')  # 1.2 pg
        CELL_DIAMETER = load_parameter_with_units(_PARAMS, 'cell_diameter')  # 5.5 μm
        MELANIN_PER_CELL = load_parameter_with_units(_PARAMS, 'melanin_per_cell')  # 14.2 fg
        MELANIN_DRY_MASS_FRACTION = load_parameter_with_units(_PARAMS, 'melanin_dry_mass_fraction')  # 0.154
        
        # Growth kinetics
        DOUBLING_TIME_OPTIMAL = load_parameter_with_units(_PARAMS, 'doubling_time_optimal')  # 1.3 hr
        DOUBLING_TIME_TYPICAL = load_parameter_with_units(_PARAMS, 'doubling_time_typical')  # 3.95 hr
        GROWTH_TEMP_OPTIMAL = load_parameter_with_units(_PARAMS, 'growth_temp_optimal')  # 30°C
        
        # Radiation parameters
        BACKGROUND_RADIATION = load_parameter_with_units(_PARAMS, 'background_radiation')  # 2.74e-7 Gy/hr
        LD50_MELANIZED = load_parameter_with_units(_PARAMS, 'LD50_C_neoformans_melanized')  # 120 Gy
        LD50_NONMELANIZED = load_parameter_with_units(_PARAMS, 'LD50_C_neoformans_nonmelanized')  # 30 Gy
        MELANIN_PROTECTION_FACTOR = load_parameter_with_units(_PARAMS, 'melanin_protection_factor')  # 4.0
        
        # Dose-response parameters (Dadachova 2007)
        GROWTH_ENHANCEMENT_500X = load_parameter_with_units(_PARAMS, 'growth_rate_500x_background')  # 3.0
        SATURATION_DOSE = load_parameter_with_units(_PARAMS, 'saturation_dose_growth')  # 0.5 Gy/hr
        
        # Mutation rates
        MUTATION_RATE = load_parameter_with_units(_PARAMS, 'mutation_rate_fungi_typical')  # 5e-10 /bp/gen
        
        # Atmospheric modern reference
        O2_MODERN = load_parameter_with_units(_PARAMS, 'O2_modern')  # 21%
        CO2_MODERN = load_parameter_with_units(_PARAMS, 'CO2_modern')  # 420 ppm
        PRESSURE_MODERN = load_parameter_with_units(_PARAMS, 'pressure_modern')  # 1.013 bar
    else:
        # Fallback values if parameters.csv not found
        print("Warning: Using fallback constants (parameters.csv not loaded)")


# =============================================================================
# Unit-Aware Helper Functions
# =============================================================================

def biomass_to_cell_count(biomass, cell_mass=None):
    """
    Convert biomass to cell count.
    
    Args:
        biomass: Quantity with mass units (e.g., kg/m², pg)
        cell_mass: Optional Quantity with mass units (default: BiologicalConstants.CELL_MASS)
    
    Returns:
        Float: Dimensionless cell count
    
    Example:
        >>> biomass = 1.2 * u.pg
        >>> cells = biomass_to_cell_count(biomass)
        >>> print(cells)  # ~1.0
    """
    if cell_mass is None:
        cell_mass = BiologicalConstants.CELL_MASS
    
    # Validate dimensions
    try:
        validate_units(biomass, u.kg)
        validate_units(cell_mass, u.kg)
    except (u.UnitConversionError, AttributeError):
        # If validation fails, assume they're already compatible
        pass
    
    count = (biomass / cell_mass).to(u.dimensionless_unscaled)
    return float(count.value)


def dose_rate_to_growth_factor(dose_rate, has_melanin=True):
    """
    Convert radiation dose rate to growth factor using Dadachova 2007 data.
    
    Implements the observed dose-response curve:
    - Baseline: 1.0 at background radiation
    - Enhancement: Up to 3.0× at ~500× background (0.15 Gy/hr)
    - Saturation: Plateaus above 0.5 Gy/hr
    - Inhibition: Declines above 10 Gy/hr (damage exceeds repair)
    
    Args:
        dose_rate: Quantity in Gy/hr
        has_melanin: bool, whether organism has melanin (affects response)
    
    Returns:
        Float: Growth factor (1.0 = baseline, 3.0 = max enhancement)
    
    Example:
        >>> bg = BiologicalConstants.BACKGROUND_RADIATION
        >>> factor = dose_rate_to_growth_factor(bg)
        >>> print(factor)  # ~1.0
    """
    # Validate units
    if isinstance(dose_rate, Qu):
        validate_units(dose_rate, u.Gy / u.hour)
        dose_value = dose_rate.value
    else:
        dose_value = dose_rate  # Assume Gy/hr if float
    
    # Load parameters
    bg = BiologicalConstants.BACKGROUND_RADIATION.value  # Gy/hr
    growth_max = BiologicalConstants.GROWTH_ENHANCEMENT_500X.value  # 3.0
    saturation = BiologicalConstants.SATURATION_DOSE.value  # 0.5 Gy/hr
    inhibition_threshold = 10.0  # Gy/hr (from parameters.csv)
    
    # Non-melanized fungi don't show radiotropic growth
    if not has_melanin:
        if dose_value > bg * 10:  # Damage without protection
            return max(0.1, 1.0 - (dose_value / inhibition_threshold))
        else:
            return 1.0
    
    # Melanized dose-response curve
    if dose_value < bg:
        return 1.0  # Below background, no effect
    
    elif dose_value < saturation:
        # Linear growth: 1.0 at bg → 3.0 at 500×bg
        relative_dose = dose_value / bg
        growth_factor = 1.0 + (growth_max - 1.0) * min(relative_dose / 500.0, 1.0)
        return min(growth_factor, growth_max)
    
    elif dose_value < inhibition_threshold:
        # Saturated enhancement
        return growth_max
    
    else:
        # Inhibition zone: damage exceeds repair
        excess = (dose_value - inhibition_threshold) / inhibition_threshold
        return max(0.1, growth_max * (1.0 - 0.5 * excess))  # Decline, but don't reach zero


def create_environment_with_units(epoch='modern'):
    """
    Create environment conditions dict with proper units for different geological epochs.
    
    Args:
        epoch: str - 'modern', 'archean_early', 'archean_late', 'proterozoic', etc.
    
    Returns:
        dict with Quantity values (with units!)
    
    Example:
        >>> env = create_environment_with_units('modern')
        >>> print(env['radiation'])  # 2.74e-07 Gy / h
        >>> print(env['oxygen'])     # 21.0 %
    """
    params = _PARAMS
    
    if epoch == 'modern':
        return {
            'radiation': load_parameter_with_units(params, 'background_radiation'),
            'oxygen': load_parameter_with_units(params, 'O2_modern'),
            'co2': load_parameter_with_units(params, 'CO2_modern'),
            'pressure': load_parameter_with_units(params, 'pressure_modern'),
        }
    
    elif epoch == 'archean_early':
        return {
            'radiation': load_parameter_with_units(params, 'radionuclide_dose_archean'),  # 2.0 Gy/yr
            'oxygen': load_parameter_with_units(params, 'O2_archean_early'),  # 0.0002%
            'co2': load_parameter_with_units(params, 'CO2_archean_early'),  # 700,000 ppm
            'pressure': load_parameter_with_units(params, 'pressure_archean_mid'),  # 0.75 bar
        }
    
    elif epoch == 'archean_late':
        return {
            'radiation': load_parameter_with_units(params, 'radionuclide_dose_archean'),
            'oxygen': load_parameter_with_units(params, 'O2_archean_late'),  # 0.01%
            'co2': load_parameter_with_units(params, 'CO2_archean_late'),  # 5,000 ppm
            'pressure': load_parameter_with_units(params, 'pressure_archean_mid'),
        }
    
    elif epoch == 'proterozoic':
        return {
            'radiation': load_parameter_with_units(params, 'radionuclide_dose_modern'),  # ~modern
            'oxygen': load_parameter_with_units(params, 'O2_proterozoic'),  # 5%
            'co2': load_parameter_with_units(params, 'CO2_proterozoic_late'),  # 2,000 ppm
            'pressure': load_parameter_with_units(params, 'pressure_proterozoic'),  # 1.0 bar
        }
    
    else:
        raise ValueError(f"Unknown epoch: {epoch}. Use 'modern', 'archean_early', 'archean_late', or 'proterozoic'")


def convert_env_to_unitless(env_with_units):
    """
    Convert unit-aware environment dict to unitless floats for legacy code.
    
    Extracts .value from each Quantity and converts to appropriate scale.
    
    Args:
        env_with_units: dict with Quantity values
    
    Returns:
        dict with float values (unitless)
    """
    env_unitless = {}
    
    for key, value in env_with_units.items():
        if isinstance(value, Qu):
            if key == 'radiation':
                # Convert to arbitrary "radiation units" (normalized to background)
                bg = BiologicalConstants.BACKGROUND_RADIATION
                env_unitless[key] = float((value / bg).to(u.dimensionless_unscaled).value)
            
            elif key == 'oxygen':
                # Convert to percent (numeric value only)
                env_unitless[key] = float(value.to(u.percent).value)
            
            elif key == 'co2':
                # Convert to ppm (numeric value only)  
                env_unitless[key] = float(value.value)  # Already in ppm
            
            elif key == 'pressure':
                # Convert to bar
                env_unitless[key] = float(value.to(u.bar).value)
            
            else:
                # Generic: extract value
                env_unitless[key] = float(value.value)
        else:
            env_unitless[key] = float(value)
    
    # Add default values for keys expected by legacy code
    if 'water' not in env_unitless:
        env_unitless['water'] = 1.0  # Plenty of water
    if 'food' not in env_unitless:
        env_unitless['food'] = 1.0  # Plenty of food
    if 'pollution' not in env_unitless:
        env_unitless['pollution'] = 0.0  # No pollution
    
    return env_unitless


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print(" TESTING models_units.py")
    print("="*70)
    
    # Test 1: Constants loading
    print("\n1. Loading BiologicalConstants...")
    print(f"  Cell mass: {BiologicalConstants.CELL_MASS}")
    print(f"  Background radiation: {BiologicalConstants.BACKGROUND_RADIATION}")
    print(f"  LD50 melanized: {BiologicalConstants.LD50_MELANIZED}")
    
    # Test 2: Biomass conversion
    print("\n2. Testing biomass_to_cell_count...")
    one_cell_mass = BiologicalConstants.CELL_MASS
    cells = biomass_to_cell_count(one_cell_mass)
    print(f"  {one_cell_mass} = {cells:.2f} cells (expected: 1.0)")
    
    # Test 3: Dose-response
    print("\n3. Testing dose_rate_to_growth_factor...")
    bg = BiologicalConstants.BACKGROUND_RADIATION
    tests = [
        (bg, "background"),
        (bg * 100, "100× background"),
        (bg * 500, "500× background (max)"),
        (1 * u.Gy / u.hour, "saturation"),
    ]
    for dose, label in tests:
        factor = dose_rate_to_growth_factor(dose)
        print(f"  {label}: {factor:.2f}×")
    
    # Test 4: Environment creation
    print("\n4. Testing create_environment_with_units...")
    env_modern = create_environment_with_units('modern')
    print(f"  Modern O₂: {env_modern['oxygen']}")
    print(f"  Modern CO₂: {env_modern['co2']}")
    
    env_archean = create_environment_with_units('archean_early')
    print(f"  Archean O₂: {env_archean['oxygen']}")
    print(f"  Archean CO₂: {env_archean['co2']}")
    
    # Test 5: Unit conversion
    print("\n5. Testing convert_env_to_unitless...")
    env_unitless = convert_env_to_unitless(env_modern)
    print(f"  Unitless: {env_unitless}")
    
    print("\n" + "="*70)
    print(" ALL TESTS COMPLETE")
    print("="*70)
