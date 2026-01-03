"""
COMPREHENSIVE INTEGRATION TEST SUITE

Tests the entire QuantumBioSim system end-to-end to ensure:
1. Backwards compatibility (existing code still works)
2. Integration between all modules
3. Scientific accuracy of results
4. Performance benchmarks
5. Edge case handling

Run with: python tests/test_integration_full.py
Estimated time: ~30 seconds
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all modules
from src.models import QuantumBiologicalSystem, PhysicsConstants
from src.models_units import BiologicalConstants, biomass_to_cell_count, dose_rate_to_growth_factor, create_environment_with_units, convert_env_to_unitless
from src.units import u, Qu, load_parameter_with_units
import pandas as pd

# Test counter
tests_passed = 0
tests_failed = 0
test_details = []

def test_result(name, passed, details=""):
    """Record test result."""
    global tests_passed, tests_failed, test_details
    if passed:
        tests_passed += 1
        status = "✓ PASS"
    else:
        tests_failed += 1
        status = "✗ FAIL"
    test_details.append(f"{status}: {name}")
    if details:
        test_details.append(f"     {details}")
    print(f"  {status}: {name}")
    if details and not passed:
        print(f"       {details}")


print("="*80)
print(" COMPREHENSIVE INTEGRATION TEST SUITE")
print("="*80)

# ============================================================================
# TEST SUITE 1: Module Import and Availability
# ============================================================================

print("\n[1/10] MODULE IMPORTS")
try:
    assert QuantumBiologicalSystem is not None
    test_result("Import QuantumBiologicalSystem", True)
except Exception as e:
    test_result("Import QuantumBiologicalSystem", False, str(e))

try:
    assert BiologicalConstants is not None
    test_result("Import BiologicalConstants", True)
except Exception as e:
    test_result("Import BiologicalConstants", False, str(e))

try:
    assert u is not None
    test_result("Import astropy.units", True)
except Exception as e:
    test_result("Import astropy.units", False, str(e))

# ============================================================================
# TEST SUITE 2: Parameters Database Integrity
# ============================================================================

print("\n[2/10] PARAMETERS DATABASE INTEGRITY")

params_file = 'data/parameters.csv'
try:
    params = pd.read_csv(params_file, comment='#')
    test_result(f"Load {params_file}", True, f"{len(params)} parameters loaded")
except Exception as e:
    test_result(f"Load {params_file}", False, str(e))
    params = None

if params is not None:
    # Check for required parameters
    required_params = [
        'background_radiation',
        'cell_mass_C_neoformans',
        'melanin_per_cell',
        'doubling_time_optimal',
        'LD50_C_neoformans_melanized',
        'O2_modern',
        'CO2_modern',
    ]
    
    for param_name in required_params:
        try:
            param = load_parameter_with_units(params, param_name)
            test_result(f"Parameter '{param_name}' exists", True, f"Value: {param}")
        except Exception as e:
            test_result(f"Parameter '{param_name}' exists", False, str(e))

# ============================================================================
# TEST SUITE 3: Backwards Compatibility with Legacy Code
# ============================================================================

print("\n[3/10] BACKWARDS COMPATIBILITY")

# Test 1: Can create QuantumBiologicalSystem with old-style floats
try:
    initial_states = [100.0, 50.0]
    energies = [0.0, 0.0]
    carrying_capacity = 1000.0
    species_params = [
        {'name': 'species_A', 'radiation_effectiveness': 0.05},
        {'name': 'species_B', 'radiation_effectiveness': 0.03},
    ]
    mutation_rate = 0.01
    
    qbs = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params, mutation_rate)
    test_result("Create QuantumBiologicalSystem (legacy floats)", True, f"States: {qbs.states}")
except Exception as e:
    test_result("Create QuantumBiologicalSystem (legacy floats)", False, str(e))

# Test 2: Legacy environment dict works
try:
    env_legacy = {
        "radiation": 0.1,
        "oxygen": 21.0,
        "co2": 0.04,
        "pollution": 0.0,
        "water": 1.0,
        "food": 1.0
    }
    qbs2 = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params, mutation_rate, env_conditions=env_legacy)
    test_result("Legacy environment dict accepted", True)
except Exception as e:
    test_result("Legacy environment dict accepted", False, str(e))

# Test 3: Can run simulation steps
try:
    deltas = qbs.environmental_effect()
    test_result("environmental_effect() runs", isinstance(deltas, np.ndarray), f"Shape: {deltas.shape}")
except Exception as e:
    test_result("environmental_effect() runs", False, str(e))

# ============================================================================
# TEST SUITE 4: New Unit-Aware Infrastructure
# ============================================================================

print("\n[4/10] UNIT-AWARE INFRASTRUCTURE")

# Test: BiologicalConstants loads correctly
try:
    cell_mass = BiologicalConstants.CELL_MASS
    assert cell_mass.unit == u.pg
    assert cell_mass.value == 1.2
    test_result("BiologicalConstants.CELL_MASS", True, f"{cell_mass}")
except Exception as e:
    test_result("BiologicalConstants.CELL_MASS", False, str(e))

# Test: biomass_to_cell_count
try:
    biomass = 100 * BiologicalConstants.CELL_MASS
    cells = biomass_to_cell_count(biomass)
    assert abs(cells - 100.0) < 0.1
    test_result("biomass_to_cell_count(100 cells)", True, f"{cells:.1f} cells")
except Exception as e:
    test_result("biomass_to_cell_count(100 cells)", False, str(e))

# Test: dose_rate_to_growth_factor
try:
    bg = BiologicalConstants.BACKGROUND_RADIATION
    factor_100x = dose_rate_to_growth_factor(bg * 100)
    factor_500x = dose_rate_to_growth_factor(bg * 500)
    
    assert 1.0 < factor_100x < 3.0
    assert abs(factor_500x - 3.0) < 0.1
    test_result("dose_rate_to_growth_factor", True, f"100×bg: {factor_100x:.2f}, 500×bg: {factor_500x:.2f}")
except Exception as e:
    test_result("dose_rate_to_growth_factor", False, str(e))

# Test: create_environment_with_units
try:
    env_modern = create_environment_with_units('modern')
    env_archean = create_environment_with_units('archean_early')
    
    assert env_modern['oxygen'].value == 21.0
    assert env_archean['oxygen'].value < 0.01
    test_result("create_environment_with_units", True, f"Modern O₂: {env_modern['oxygen']}, Archean O₂: {env_archean['oxygen']}")
except Exception as e:
    test_result("create_environment_with_units", False, str(e))

# Test: convert_env_to_unitless
try:
    env_units = create_environment_with_units('modern')
    env_legacy = convert_env_to_unitless(env_units)
    
    assert isinstance(env_legacy['radiation'], float)
    assert isinstance(env_legacy['oxygen'], float)
    test_result("convert_env_to_unitless", True, f"Converted: {env_legacy}")
except Exception as e:
    test_result("convert_env_to_unitless", False, str(e))

# ============================================================================
# TEST SUITE 5: Integration Between Old and New
# ============================================================================

print("\n[5/10] OLD ↔ NEW INTEGRATION")

# Test: Use unit-aware environment with QuantumBiologicalSystem
try:
    env_units = create_environment_with_units('modern')
    env_legacy = convert_env_to_unitless(env_units)
    
    qbs_new = QuantumBiologicalSystem(
        initial_states=[100.0, 50.0],
        energies=[0.0, 0.0],
        carrying_capacity=1000.0,
        species_params=species_params,
        mutation_rate=0.01,
        env_conditions=env_legacy
    )
    
    deltas = qbs_new.environmental_effect()
    test_result("QuantumBiologicalSystem with converted environment", True, f"Deltas: {deltas}")
except Exception as e:
    test_result("QuantumBiologicalSystem with converted environment", False, str(e))

# Test: Use research parameters in simulation
try:
    # Convert research-grade parameters to simulation values
    doubling_time_hr = BiologicalConstants.DOUBLING_TIME_OPTIMAL.value  # 1.3 hr
    growth_rate_per_step = 1.0 / doubling_time_hr  # per hour
    
    # Create species with realistic parameters
    species_realistic = [
        {
            'name': 'C_neoformans_melanized',
            'radiation_effectiveness': 0.05,  # Will be replaced with dose_rate_to_growth_factor later
            'aging_rate': 0.01,  # Required by QuantumBiologicalSystem
            'phenotype': {
                'rad_sensitivity': 0.02,  # Low (melanin protection)
                'aerobic_efficiency': 0.7,
                'water_efficiency': 0.5,
                'nutrient_efficiency': 0.6,
                'pollution_resistance': 0.3,
            }
        }
    ]
    
    qbs_realistic = QuantumBiologicalSystem(
        initial_states=[100.0],
        energies=[0.0],
        carrying_capacity=1000.0,
        species_params=species_realistic,
        mutation_rate=BiologicalConstants.MUTATION_RATE.value,
        env_conditions=env_legacy
    )
    
    test_result("Simulation with research parameters", True, f"Mutation rate: {BiologicalConstants.MUTATION_RATE}")
except Exception as e:
    test_result("Simulation with research parameters", False, str(e))

# ============================================================================
# TEST SUITE 6: Full Simulation Run
# ============================================================================

print("\n[6/10] FULL SIMULATION RUN")

try:
    # Run simulation for 10 generations
    start_time = time.time()
    
    results = qbs_realistic.update_states(num_time_steps=10)
    
    elapsed = time.time() - start_time
    
    final_state = results[-1] if len(results) > 0 else qbs_realistic.states
    test_result("Simulation runs for 10 generations", True, f"Final state: {final_state}, Time: {elapsed:.3f}s")
except Exception as e:
    test_result("Simulation runs for 10 generations", False, str(e))

# ============================================================================
# TEST SUITE 7: Scientific Validation
# ============================================================================

print("\n[7/10] SCIENTIFIC VALIDATION")

# Test: Dose-response curve matches literature
try:
    bg = BiologicalConstants.BACKGROUND_RADIATION
    
    # Expected from Dadachova 2007:
    # - Background: ~1.0× growth
    # - 500× background: ~3.0× growth
    
    factor_1x = dose_rate_to_growth_factor(bg)
    factor_500x = dose_rate_to_growth_factor(bg * 500)
    
    assert abs(factor_1x - 1.0) < 0.1, f"Background should be 1.0×, got {factor_1x:.2f}×"
    assert abs(factor_500x - 3.0) < 0.1, f"500× bg should be 3.0×, got {factor_500x:.2f}×"
    
    test_result("Dose-response matches Dadachova 2007", True, f"1×: {factor_1x:.2f}, 500×: {factor_500x:.2f}")
except AssertionError as e:
    test_result("Dose-response matches Dadachova 2007", False, str(e))
except Exception as e:
    test_result("Dose-response matches Dadachova 2007", False, str(e))

# Test: LD50 values are reasonable
try:
    ld50_mel = BiologicalConstants.LD50_MELANIZED.value  # 120 Gy
    ld50_non = BiologicalConstants.LD50_NONMELANIZED.value  # 30 Gy
    protection_factor = ld50_mel / ld50_non
    
    assert abs(protection_factor - 4.0) < 0.5, f"Protection factor should be ~4, got {protection_factor:.1f}"
    test_result("Melanin protection factor ~4×", True, f"LD50 ratio: {protection_factor:.1f}×")
except AssertionError as e:
    test_result("Melanin protection factor ~4×", False, str(e))
except Exception as e:
    test_result("Melanin protection factor ~4×", False, str(e))

# Test: Archean atmosphere is anoxic
try:
    env_archean = create_environment_with_units('archean_early')
    o2_archean = env_archean['oxygen'].value
    
    assert o2_archean < 0.01, f"Archean O₂ should be <0.01%, got {o2_archean}%"
    test_result("Archean atmosphere is anoxic", True, f"O₂: {o2_archean}%")
except AssertionError as e:
    test_result("Archean atmosphere is anoxic", False, str(e))
except Exception as e:
    test_result("Archean atmosphere is anoxic", False, str(e))

# ============================================================================
# TEST SUITE 8: Edge Cases and Error Handling
# ============================================================================

print("\n[8/10] EDGE CASES & ERROR HANDLING")

# Test: Invalid epoch name
try:
    try:
        env_invalid = create_environment_with_units('invalid_epoch_name')
        test_result("Invalid epoch raises error", False, "Should have raised ValueError")
    except ValueError:
        test_result("Invalid epoch raises error", True)
except Exception as e:
    test_result("Invalid epoch raises error", False, str(e))

# Test: Extremely high radiation (inhibition zone)
try:
    extreme_dose = 1000 * u.Gy / u.hour
    factor = dose_rate_to_growth_factor(extreme_dose)
    
    assert factor < 3.0, f"Extreme dose should inhibit growth, got {factor:.2f}×"
    test_result("Extreme radiation causes inhibition", True, f"1000 Gy/hr: {factor:.2f}×")
except AssertionError as e:
    test_result("Extreme radiation causes inhibition", False, str(e))
except Exception as e:
    test_result("Extreme radiation causes inhibition", False, str(e))

# Test: Zero biomass
try:
    zero_biomass = 0 * u.pg
    cells = biomass_to_cell_count(zero_biomass)
    assert cells == 0.0
    test_result("Zero biomass = 0 cells", True)
except Exception as e:
    test_result("Zero biomass = 0 cells", False, str(e))

# Test: Non-melanized organism in radiation
try:
    factor_nonmel = dose_rate_to_growth_factor(bg * 100, has_melanin=False)
    assert abs(factor_nonmel - 1.0) < 0.1, f"Non-melanized should not show radiotropism, got {factor_nonmel:.2f}×"
    test_result("Non-melanized organisms no radiotropism", True, f"100×bg: {factor_nonmel:.2f}×")
except AssertionError as e:
    test_result("Non-melanized organisms no radiotropism", False, str(e))
except Exception as e:
    test_result("Non-melanized organisms no radiotropism", False, str(e))

# ============================================================================
# TEST SUITE 9: Performance Benchmarks
# ============================================================================

print("\n[9/10] PERFORMANCE BENCHMARKS")

# Benchmark: Parameter loading
try:
    start = time.time()
    for _ in range(100):
        param = load_parameter_with_units(params, 'background_radiation')
    elapsed = time.time() - start
    
    test_result("Parameter loading (100 iterations)", True, f"{elapsed*1000:.2f}ms total, {elapsed*10:.2f}ms avg")
except Exception as e:
    test_result("Parameter loading (100 iterations)", False, str(e))

# Benchmark: Unit conversions
try:
    start = time.time()
    for _ in range(1000):
        biomass = 100 * BiologicalConstants.CELL_MASS
        cells = biomass_to_cell_count(biomass)
    elapsed = time.time() - start
    
    test_result("Biomass conversion (1000 iterations)", True, f"{elapsed*1000:.2f}ms total, {elapsed:.3f}ms avg")
except Exception as e:
    test_result("Biomass conversion (1000 iterations)", False, str(e))

# Benchmark: Dose-response calculation
try:
    start = time.time()
    for i in range(1000):
        dose = BiologicalConstants.BACKGROUND_RADIATION * (i + 1)
        factor = dose_rate_to_growth_factor(dose)
    elapsed = time.time() - start
    
    test_result("Dose-response calc (1000 iterations)", True, f"{elapsed*1000:.2f}ms total, {elapsed:.3f}ms avg")
except Exception as e:
    test_result("Dose-response calc (1000 iterations)", False, str(e))

# ============================================================================
# TEST SUITE 10: Code Quality Checks
# ============================================================================

print("\n[10/10] CODE QUALITY CHECKS")

# Test: All functions have docstrings
try:
    from src.models_units import biomass_to_cell_count, dose_rate_to_growth_factor, create_environment_with_units
    
    funcs = [biomass_to_cell_count, dose_rate_to_growth_factor, create_environment_with_units]
    all_documented = all(func.__doc__ is not None for func in funcs)
    
    test_result("All public functions have docstrings", all_documented)
except Exception as e:
    test_result("All public functions have docstrings", False, str(e))

# Test: No bare excepts
try:
    with open('src/models_units.py', 'r') as f:
        content = f.read()
        bare_excepts = content.count('except:')
        
    test_result("No bare 'except:' statements", bare_excepts == 0, f"Found: {bare_excepts}")
except Exception as e:
    test_result("No bare 'except:' statements", False, str(e))

# ============================================================================
# FINAL REPORT
# ============================================================================

print("\n" + "="*80)
print(" TEST RESULTS SUMMARY")
print("="*80)

total_tests = tests_passed + tests_failed
pass_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {tests_passed} ({pass_rate:.1f}%)")
print(f"Failed: {tests_failed}")

if tests_failed > 0:
    print("\n❌ SOME TESTS FAILED")
    print("\nFailed Tests:")
    for detail in test_details:
        if "✗ FAIL" in detail:
            print(f"  {detail}")
else:
    print("\n✅ ALL TESTS PASSED!")

print("\n" + "="*80)

# Exit with appropriate code
sys.exit(0 if tests_failed == 0 else 1)
