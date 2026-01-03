"""
Unit definitions and conversions for QuantumBioSim.

This module provides a centralized unit system using astropy.units to ensure
dimensional correctness throughout the simulation. All physical quantities
should use these units.

Usage:
    from src.units import u, Qu
    
    # Define a quantity with units
    dose_rate = 2.74e-7 * u.Gy / u.hour
    
    # Convert to different units
    annual_dose = dose_rate.to(u.Gy / u.year)
    
    # Use custom units
    oxygen = 21 * u.percent
    co2 = 420 * u.ppm
"""

from astropy import units as u
from astropy.units import Quantity as Qu

# ============================================================================
# Custom Unit Definitions
# ============================================================================

# Biological/Ecological Units
PAL = u.def_unit('PAL', doc='Present Atmospheric Level (21% O₂)')
CFU = u.def_unit('CFU', doc='Colony Forming Units')
copies_per_gram = u.def_unit('copies_per_gram', doc='Gene copies per gram')

# Concentration Units (additional to astropy defaults)
# Custom units - using dimensionless_unscaled for proper astropy type checking
ppm = u.def_unit('ppm', doc="Parts per million", represents=1e-6 * u.dimensionless_unscaled)
ppb = u.def_unit('ppb', doc="Parts per billion", represents=1e-9 * u.dimensionless_unscaled)
ppt = u.def_unit('ppt', doc="Parts per trillion", represents=1e-12 * u.dimensionless_unscaled)
# Note: astropy already has u.percent

# Radiation Units (astropy has Gy, Sv, we just alias for clarity)
# Already defined in astropy: u.Gy (Gray), u.Sv (Sievert)

# Rate Units - combining existing units
per_bp_per_generation = u.def_unit('per_bp_per_generation', 
                                   doc='Mutation rate: per base pair per generation')

# Biomass Units
g_C_per_m2 = u.def_unit('g_C_per_m2', 
                        represents=u.gram / u.meter**2,
                        doc='Grams of carbon per square meter')

g_dry_weight_per_L = u.def_unit('g_dry_weight_per_L',
                                represents=u.gram / u.liter,
                                doc='Grams dry weight per liter')

# ============================================================================
# Common Unit Combinations
# ============================================================================

# Time units
hour = u.hour
minute = u.minute
second = u.second
day = u.day
year = u.year
Ga = 1e9 * year  # Billion years (Giga-annum)

# Radiation dose rate units
Gy_per_hour = u.Gy / u.hour
Gy_per_minute = u.Gy / u.minute
Gy_per_year = u.Gy / u.year
mSv_per_year = u.mSv / u.year

# Mass units
picogram = u.pg
femtogram = u.fg
nanogram = u.ng
microgram = u.ug
milligram = u.mg
gram = u.g
kilogram = u.kg

# Length units
nanometer = u.nm
micrometer = u.um
millimeter = u.mm
meter = u.m
kilometer = u.km

# Concentration units
micromolar = u.micromole / u.liter  # μM
millimolar = u.millimole / u.liter  # mM
molar = u.mole / u.liter            # M

# Pressure units
bar = u.bar
pascal = u.Pa

# Temperature units
celsius = u.Celsius
kelvin = u.K

# ============================================================================
# Unit Conversion Utilities
# ============================================================================

def to_gy_per_hour(dose_rate, from_unit='mSv/year'):
    """
    Convert dose rate to Gy/hour.
    
    Args:
        dose_rate: Numerical value or Quantity
        from_unit: String or astropy unit
    
    Returns:
        Quantity in Gy/hour
    
    Example:
        >>> annual_dose = 2.4  # mSv/year
        >>> hourly = to_gy_per_hour(annual_dose, 'mSv/year')
        >>> print(hourly)
        2.74e-7 Gy / hr
    """
    if isinstance(dose_rate, Qu):
        return dose_rate.to(u.Gy / u.hour)
    else:
        if from_unit == 'mSv/year':
            # 1 Sv = 1 Gy for gamma radiation (wR=1)
            # Convert mSv to Gy then divide by hours in year
            return (dose_rate * u.mSv / u.year).to(u.Gy / u.hour)
        elif isinstance(from_unit, str):
            # Parse string unit
            from_u = u.Unit(from_unit)
            return (dose_rate * from_u).to(u.Gy / u.hour)
        else:
            return (dose_rate * from_unit).to(u.Gy / u.hour)


def percent_to_fraction(percent_value):
    """Convert percentage to dimensionless fraction."""
    if isinstance(percent_value, Qu):
        return percent_value.to(u.dimensionless_unscaled)
    else:
        return percent_value / 100.0


def ppm_to_fraction(ppm_value):
    """Convert parts per million to dimensionless fraction."""
    if isinstance(ppm_value, Qu):
        return ppm_value.to(u.dimensionless_unscaled)
    else:
        return ppm_value / 1e6


def validate_units(quantity, expected_unit):
    """
    Validate that a quantity has the expected physical dimension.
    
    Args:
        quantity: Quantity to check
        expected_unit: astropy Unit to compare against
    
    Raises:
        u.UnitConversionError if dimensions don't match
    
    Example:
        >>> dose = 120 * u.Gy
        >>> validate_units(dose, u.Gy)  # OK
        >>> validate_units(dose, u.meter)  # Raises UnitConversionError
    """
    if not isinstance(quantity, Qu):
        raise TypeError(f"Expected Quantity, got {type(quantity)}")
    
    # Try to convert to check compatibility
    try:
        _ = quantity.to(expected_unit)
    except u.UnitConversionError as e:
        raise u.UnitConversionError(
            f"Unit mismatch: {quantity.unit} cannot be converted to {expected_unit}"
        ) from e


# ============================================================================
# Parameter Loading Utilities
# ============================================================================

def load_parameter_with_units(df, param_name):
    """
    Load a parameter from parameters.csv DataFrame with units attached.
    
    Args:
        df: pandas DataFrame from parameters.csv
        param_name: String name of parameter
    
    Returns:
        Quantity with value and unit
    
    Example:
        >>> import pandas as pd
        >>> params = pd.read_csv('data/parameters.csv', comment='#')
        >>> bg_rad = load_parameter_with_units(params, 'background_radiation')
        >>> print(bg_rad)
        2.74e-7 Gy / hr
    """
    row = df[df['parameter'] == param_name]
    if len(row) == 0:
        raise ValueError(f"Parameter '{param_name}' not found in database")
    
    value = row['value'].values[0]
    unit_str = row['unit'].values[0]
    
    # Parse custom units
    unit_map = {
        'Gy_per_hour': u.Gy / u.hour,
        'Gy_per_minute': u.Gy / u.minute,
        'Gy_per_year': u.Gy / u.year,
        'percent_volume': u.percent,
        'ppm': ppm,
        'micromolar': micromolar,
        'millimolar': millimolar,
        'per_bp_per_generation': per_bp_per_generation,
        'femtogram': u.fg,
        'picogram': u.pg,
        'nanometer': u.nm,
        'micrometer': u.um,
        'g_C_per_m2': g_C_per_m2,
        'g_dry_weight_per_L': g_dry_weight_per_L,
        'kg_per_m3': u.kg / u.m**3,
        'copies_per_gram': copies_per_gram,
        'dimensionless': u.dimensionless_unscaled,
        'celsius': u.Celsius,
        'bar': u.bar,
        'Gray': u.Gy,
        'hours': u.hour,
        'minutes': u.minute,
        'degrees': u.degree,
        'count': u.dimensionless_unscaled,
        'fold_change': u.dimensionless_unscaled,
        'mJ_per_cm2': u.mJ / u.cm**2,
        'nmol_per_mg_per_min': u.nmol / u.mg / u.minute,
        'nmol_per_g_per_hour': u.nmol / u.g / u.hour,
        'cubic_micrometer': u.um**3,
    }
    
    if unit_str in unit_map:
        unit = unit_map[unit_str]
    else:
        # Try to parse as astropy unit string
        try:
            unit = u.Unit(unit_str)
        except ValueError:
            raise ValueError(f"Unknown unit: {unit_str} for parameter {param_name}")
    
    return value * unit


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'u', 'Qu',
    'PAL', 'CFU', 'ppm', 'copies_per_gram', 'per_bp_per_generation',
    'g_C_per_m2', 'g_dry_weight_per_L',
    'hour', 'minute', 'second', 'day', 'year', 'Ga',
    'Gy_per_hour', 'Gy_per_minute', 'Gy_per_year', 'mSv_per_year',
    'picogram', 'femtogram', 'nanogram', 'microgram', 'milligram', 'gram', 'kilogram',
    'nanometer', 'micrometer', 'millimeter', 'meter', 'kilometer',
    'micromolar', 'millimolar', 'molar',
    'bar', 'pascal',
    'celsius', 'kelvin',
    'to_gy_per_hour', 'percent_to_fraction', 'ppm_to_fraction',
    'validate_units', 'load_parameter_with_units',
]
