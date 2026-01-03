"""
Unit tests for src/units.py module.

Tests all unit conversions, custom units, and parameter loading to ensure
dimensional correctness before refactoring the codebase.

Run with: python test_units.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.units import *
import pandas as pd
import numpy as np

def test_astropy_import():
    """Test that astropy.units imports correctly."""
    print("TEST: astropy.units import...")
    assert u is not None
    assert Qu is not None
    print("  ✓ astropy.units imported successfully")

def test_custom_units():
    """Test custom unit definitions."""
    print("\nTEST: Custom units...")
    
    # Test ppm
    co2 = 420 * ppm
    assert co2.unit == ppm
    print(f"  ✓ ppm unit: {co2}")
    
    # Test custom biomass units
    biomass = 2.5 * g_C_per_m2
    assert biomass.unit == g_C_per_m2
    print(f"  ✓ g_C_per_m2 unit: {biomass}")
    
    # Test rate units
    mut_rate = 1.67e-10 * per_bp_per_generation
    assert mut_rate.unit == per_bp_per_generation
    print(f"  ✓ per_bp_per_generation unit: {mut_rate}")

def test_radiation_conversions():
    """Test radiation dose conversions (critical for correctness)."""
    print("\nTEST: Radiation dose conversions...")
    
    # Test background radiation conversion
    bg_annual = 2.4 * u.mSv / u.year
    bg_hourly = bg_annual.to(u.Gy / u.hour)
    
    expected = 2.74e-7  # From verified calculation
    actual = bg_hourly.value
    
    assert abs(actual - expected) / expected < 0.01, f"Expected {expected}, got {actual}"
    print(f"  ✓ Background radiation: {bg_annual} = {bg_hourly:.3e}")
    
    # Test conversion utility
    bg_util = to_gy_per_hour(2.4, from_unit='mSv/year')
    assert abs(bg_util.value - expected) / expected < 0.01
    print(f"  ✓ to_gy_per_hour utility: {bg_util:.3e}")
    
    # Test Dadachova experimental dose
    dad_rate = 11.94 * u.Gy / u.minute
    dad_hourly = dad_rate.to(u.Gy / u.hour)
    assert abs(dad_hourly.value - 716.4) < 1
    print(f"  ✓ Dadachova dose rate: {dad_rate} = {dad_hourly:.1f}")

def test_concentration_conversions():
    """Test atmospheric concentration conversions."""
    print("\nTEST: Concentration conversions...")
    
    # Test ppm to fraction
    co2_ppm = 420 * ppm
    co2_frac = ppm_to_fraction(co2_ppm)
    assert abs(co2_frac - 0.000420) < 1e-9
    print(f"  ✓ ppm to fraction: {co2_ppm} = {co2_frac}")
    
    # Test percent to fraction
    o2_pct = 21 * u.percent
    o2_frac = percent_to_fraction(o2_pct)
    assert abs(o2_frac - 0.21) < 1e-9
    print(f"  ✓ percent to fraction: {o2_pct} = {o2_frac}")
    
    # Test micromolar
    km = 12 * micromolar
    km_molar = km.to(u.mole / u.liter)
    assert abs(km_molar.value - 1.2e-5) < 1e-9
    print(f"  ✓ micromolar: {km} = {km_molar:.2e} M")

def test_unit_validation():
    """Test unit validation function."""
    print("\nTEST: Unit validation...")
    
    dose = 120 * u.Gy
    
    # Should pass
    try:
        validate_units(dose, u.Gy)
        print("  ✓ Valid unit (Gy) accepted")
    except:
        raise AssertionError("validate_units rejected valid unit")
    
    # Should fail
    try:
        validate_units(dose, u.meter)
        raise AssertionError("validate_units accepted incompatible unit!")
    except u.UnitConversionError:
        print("  ✓ Invalid unit (meter) rejected correctly")

def test_parameter_loading():
    """Test loading parameters from CSV with units."""
    print("\nTEST: Parameter loading from CSV...")
    
    # Load parameters
    params_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'parameters.csv')
    if not os.path.exists(params_file):
        print(f"  ⚠ Skipping (file not found: {params_file})")
        return
    
    params = pd.read_csv(params_file, comment='#')
    
    # Test background radiation
    bg_rad = load_parameter_with_units(params, 'background_radiation')
    assert bg_rad.unit == u.Gy / u.hour
    assert abs(bg_rad.value - 2.74e-7) / 2.74e-7 < 0.01
    print(f"  ✓ Loaded background_radiation: {bg_rad:.3e}")
    
    # Test melanin content
    melanin = load_parameter_with_units(params, 'melanin_per_cell')
    assert melanin.unit == u.fg
    assert abs(melanin.value - 14.2) < 0.1
    print(f"  ✓ Loaded melanin_per_cell: {melanin}")
    
    # Test mutation rate
    mut_rate = load_parameter_with_units(params, 'mutation_rate_S_cerevisiae')
    assert mut_rate.unit == per_bp_per_generation
    assert abs(mut_rate.value - 1.67e-10) / 1.67e-10 < 0.01
    print(f"  ✓ Loaded mutation_rate_S_cerevisiae: {mut_rate:.3e}")
    
    # Test atmospheric parameter
    o2_modern = load_parameter_with_units(params, 'O2_modern')
    assert o2_modern.unit == u.percent
    assert abs(o2_modern.value - 21.0) < 0.1
    print(f"  ✓ Loaded O2_modern: {o2_modern}")

def test_time_conversions():
    """Test time unit conversions."""
    print("\nTEST: Time conversions...")
    
    # Test doubling time
    dt = 1.3 * u.hour
    dt_min = dt.to(u.minute)
    assert abs(dt_min.value - 78) < 1
    print(f"  ✓ Doubling time: {dt} = {dt_min:.0f}")
    
    # Test geological time
    archean = 2.5 * Ga
    archean_yr = archean.to(u.year)
    assert abs(archean_yr.value - 2.5e9) < 1e6
    print(f"  ✓ Geological time: {archean / u.year:.2e} years")

def test_mass_conversions():
    """Test mass unit conversions."""
    print("\nTEST: Mass conversions...")
    
    # Test melanin mass
    melanin_fg = 14.2 * u.fg
    melanin_ng = melanin_fg.to(u.ng)
    assert abs(melanin_ng.value - 0.0000142) < 0.000001  # 14.2 fg = 0.0000142 ng
    print(f"  ✓ Melanin mass: {melanin_fg} = {melanin_ng:.7f}")
    
    # Test cell mass
    cell_mass = 1.2 * u.pg
    cell_mass_fg = cell_mass.to(u.fg)
    assert abs(cell_mass_fg.value - 1200) < 1
    print(f"  ✓ Cell mass: {cell_mass} = {cell_mass_fg:.0f}")

def test_all():
    """Run all tests."""
    print("="*70)
    print(" UNIT TESTS FOR src/units.py")
    print("="*70)
    
    try:
        test_astropy_import()
        test_custom_units()
        test_radiation_conversions()
        test_concentration_conversions()
        test_unit_validation()
        test_parameter_loading()
        test_time_conversions()
        test_mass_conversions()
        
        print("\n" + "="*70)
        print(" ✓ ALL TESTS PASSED")
        print("="*70)
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(f" ✗ TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
