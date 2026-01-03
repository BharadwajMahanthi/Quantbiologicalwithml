"""
Comprehensive unit tests for models_units.py

Tests all unit-aware infrastructure before integration with main codebase.
Run with: python -m pytest tests/test_models_units.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models_units import *
from src.units import u, Qu

def test_biological_constants_loading():
    """Test that all biological constants load correctly with units."""
    print("\nTEST: BiologicalConstants loading...")
    
    # Check cell properties
    assert BiologicalConstants.CELL_MASS.unit == u.pg
    assert abs(BiologicalConstants.CELL_MASS.value - 1.2) < 0.001
    print(f"  ✓ Cell mass: {BiologicalConstants.CELL_MASS}")
    
    assert BiologicalConstants.CELL_DIAMETER.unit == u.um
    assert abs(BiologicalConstants.CELL_DIAMETER.value - 5.5) < 0.1
    print(f"  ✓ Cell diameter: {BiologicalConstants.CELL_DIAMETER}")
    
    # Check radiation constants
    assert BiologicalConstants.BACKGROUND_RADIATION.unit == u.Gy / u.hour
    assert abs(BiologicalConstants.BACKGROUND_RADIATION.value - 2.74e-7) / 2.74e-7 < 0.01
    print(f"  ✓ Background radiation: {BiologicalConstants.BACKGROUND_RADIATION:.3e}")
    
    assert BiologicalConstants.LD50_MELANIZED.unit == u.Gy
    assert abs(BiologicalConstants.LD50_MELANIZED.value - 120) < 1
    print(f"  ✓ LD50 melanized: {BiologicalConstants.LD50_MELANIZED}")
    
    # Check growth parameters
    assert BiologicalConstants.DOUBLING_TIME_OPTIMAL.unit == u.hour
    assert abs(BiologicalConstants.DOUBLING_TIME_OPTIMAL.value - 1.3) < 0.1
    print(f"  ✓ Doubling time: {BiologicalConstants.DOUBLING_TIME_OPTIMAL}")


def test_biomass_to_cell_count():
    """Test biomass to cell count conversion."""
    print("\nTEST: biomass_to_cell_count...")
    
    # Test 1: One cell
    one_cell = BiologicalConstants.CELL_MASS
    cells = biomass_to_cell_count(one_cell)
    assert abs(cells - 1.0) < 0.01
    print(f"  ✓ {one_cell} = {cells:.2f} cells (expected: 1.0)")
    
    # Test 2: 100 cells
    hundred_cells = 100 * BiologicalConstants.CELL_MASS
    cells = biomass_to_cell_count(hundred_cells)
    assert abs(cells - 100.0) < 0.1
    print(f"  ✓ {hundred_cells:.1f} = {cells:.1f} cells (expected: 100.0)")
    
    # Test 3: Different units (kg)
    biomass_kg = (1.2 * u.pg).to(u.kg)
    cells = biomass_to_cell_count(biomass_kg)
    assert abs(cells - 1.0) < 0.01
    print(f"  ✓ {biomass_kg:.3e} = {cells:.2f} cells (unit conversion works)")


def test_dose_response_curve():
    """Test radiation dose to growth factor conversion."""
    print("\nTEST: dose_rate_to_growth_factor...")
    
    bg = BiologicalConstants.BACKGROUND_RADIATION
    
    # Test baseline (background radiation)
    factor = dose_rate_to_growth_factor(bg, has_melanin=True)
    assert abs(factor - 1.0) < 0.01
    print(f"  ✓ Background: {factor:.2f}× (expected: 1.0×)")
    
    # Test low enhancement (100× background)
    factor = dose_rate_to_growth_factor(bg * 100, has_melanin=True)
    assert 1.0 < factor < 2.0
    print(f"  ✓ 100× background: {factor:.2f}× (expected: 1.0-2.0×)")
    
    # Test max enhancement (500× background)
    factor = dose_rate_to_growth_factor(bg * 500, has_melanin=True)
    assert abs(factor - 3.0) < 0.1
    print(f"  ✓ 500× background: {factor:.2f}× (expected: 3.0×)")
    
    # Test saturation (1 Gy/hr)
    factor = dose_rate_to_growth_factor(1 * u.Gy / u.hour, has_melanin=True)
    assert abs(factor - 3.0) < 0.1
    print(f"  ✓ Saturation (1 Gy/hr): {factor:.2f}× (expected: 3.0×)")
    
    # Test inhibition (100 Gy/hr - extreme dose)
    factor = dose_rate_to_growth_factor(100 * u.Gy / u.hour, has_melanin=True)
    assert factor < 3.0  # Should decline
    print(f"  ✓ Inhibition (100 Gy/hr): {factor:.2f}× (expected: <3.0×)")
    
    # Test non-melanized (no radiotropism)
    factor_nonmelanized = dose_rate_to_growth_factor(bg * 100, has_melanin=False)
    assert abs(factor_nonmelanized - 1.0) < 0.1  # No growth enhancement
    print(f"  ✓ Non-melanized at 100×bg: {factor_nonmelanized:.2f}× (expected: 1.0×)")


def test_environment_creation():
    """Test creating unit-aware environments for different epochs."""
    print("\nTEST: create_environment_with_units...")
    
    # Test modern environment
    env_modern = create_environment_with_units('modern')
    assert env_modern['radiation'].unit == u.Gy / u.hour
    assert env_modern['oxygen'].unit == u.percent
    assert env_modern['co2'].value == 420.0  # ppm
    assert env_modern['pressure'].unit == u.bar
    print(f"  ✓ Modern: O₂={env_modern['oxygen']}, CO₂={env_modern['co2']}")
    
    # Test archean environment
    env_archean = create_environment_with_units('archean_early')
    assert env_archean['oxygen'].value < 0.01  # Very low O₂
    assert env_archean['co2'].value > 100000  # Very high CO₂
    print(f"  ✓ Archean: O₂={env_archean['oxygen']}, CO₂={env_archean['co2']}")
    
    # Test proterozoic environment
    env_proterozoic = create_environment_with_units('proterozoic')
    assert env_proterozoic['oxygen'].value > env_archean['oxygen'].value  # More O₂ than Archean
    assert env_proterozoic['oxygen'].value < env_modern['oxygen'].value  # Less than modern
    print(f"  ✓ Proterozoic: O₂={env_proterozoic['oxygen']}")
    
    # Test invalid epoch
    try:
        create_environment_with_units('invalid_epoch')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Invalid epoch raises ValueError: {e}")


def test_unit_conversion_to_unitless():
    """Test converting unit-aware environment to unitless dict."""
    print("\nTEST: convert_env_to_unitless...")
    
    env_with_units = create_environment_with_units('modern')
    env_unitless = convert_env_to_unitless(env_with_units)
    
    # Check that all values are now floats
    assert isinstance(env_unitless['radiation'], float)
    assert isinstance(env_unitless['oxygen'], float)
    assert isinstance(env_unitless['co2'], float)
    assert isinstance(env_unitless['pressure'], float)
    print(f"  ✓ All values converted to float")
    
    # Check reasonable values
    assert env_unitless['radiation'] == 1.0  # Normalized to background
    assert env_unitless['oxygen'] == 21.0  # Percent
    assert env_unitless['co2'] == 420.0  # ppm
    assert abs(env_unitless['pressure'] - 1.013) < 0.01  # bar
    print(f"  ✓ Values: {env_unitless}")


def test_dimensional_consistency():
    """Test that all operations maintain dimensional consistency."""
    print("\nTEST: Dimensional consistency...")
    
    # Test biomass operation
    mass1 = 1.2 * u.pg
    mass2 = 5.0 * u.fg
    total_mass = mass1 + mass2.to(u.pg)
    cells = biomass_to_cell_count(total_mass)
    assert cells > 1.0
    print(f"  ✓ Mass addition: {mass1} + {mass2} = {total_mass:.2f}")
    
    #Test radiation operation
    dose1 = BiologicalConstants.BACKGROUND_RADIATION
    dose2 = 10 * BiologicalConstants.BACKGROUND_RADIATION  # Use 10× background instead
    total_dose = dose1 + dose2
    factor = dose_rate_to_growth_factor(total_dose)
    assert factor >= 1.0  # May be 1.0 at low doses
    print(f"  ✓ Dose addition: {dose1:.3e} + {dose2:.3e} = {total_dose:.3e}")


def test_integration_with_legacy_code():
    """Test that new code can work alongside legacy unitless code."""
    print("\nTEST: Integration with legacy code...")
    
    # Create unit-aware environment
    env_units = create_environment_with_units('modern')
    
    # Convert to unitless for legacy code
    env_legacy = convert_env_to_unitless(env_units)
    
    # Simulate legacy code using the environment
    radiation_factor = env_legacy['radiation']  # Should be 1.0 (normalized)
    oxygen_level = env_legacy['oxygen']  # Should be 21.0 (percent)
    
    assert radiation_factor == 1.0
    assert oxygen_level == 21.0
    print(f"  ✓ Legacy code can use converted values")
    
    # Test that we can convert back
    radiation_with_units = radiation_factor * BiologicalConstants.BACKGROUND_RADIATION
    growth = dose_rate_to_growth_factor(radiation_with_units)
    assert abs(growth - 1.0) < 0.01
    print(f"  ✓ Can convert back to units for calculations")


def run_all_tests():
    """Run all tests and report results."""
    print("="*70)
    print(" COMPREHENSIVE UNIT TESTS FOR models_units.py")
    print("="*70)
    
    try:
        test_biological_constants_loading()
        test_biomass_to_cell_count()
        test_dose_response_curve()
        test_environment_creation()
        test_unit_conversion_to_unitless()
        test_dimensional_consistency()
        test_integration_with_legacy_code()
        
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
    success = run_all_tests()
    sys.exit(0 if success else 1)
