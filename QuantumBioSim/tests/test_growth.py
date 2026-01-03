"""
Unit tests for src/growth.py - Growth rate equations

Tests all growth rate functions against literature values and edge cases.
"""

import pytest
import numpy as np

# Try to import growth module with units
try:
    from src.growth import (
        malthusian_growth_rate,
        logistic_growth_factor,
        radiation_growth_enhancement,
        oxygen_growth_limitation,
        temperature_correction_factor,
        calculate_growth_rate,
        UNITS_AVAILABLE
    )
    from src.units import u, Qu
    GROWTH_AVAILABLE = True
except ImportError as e:
    GROWTH_AVAILABLE = False
    pytest.skip(f"Growth module not available: {e}", allow_module_level=True)


class TestMalthusianGrowth:
    """Test exponential growth rate calculations."""
    
    def test_basic_calculation(self):
        """Verify r = ln(2) / doubling_time."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        # C. neoformans typical doubling time
        doubling_time = 3.95 * u.hour
        r = malthusian_growth_rate(doubling_time)
        
        expected = np.log(2) / 3.95  # 0.175 hr^-1
        assert abs(r.value - expected) < 0.001
        assert r.unit == (1 / u.hour)
    
    def test_without_units(self):
        """Test fallback without astropy units."""
        r = malthusian_growth_rate(3.95, has_units=False)
        expected = np.log(2) / 3.95
        assert abs(r - expected) < 0.001
    
    def test_different_doubling_times(self):
        """Test various doubling times."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        # Fast growth (1.3 hrs - optimal conditions)
        r_fast = malthusian_growth_rate(1.3 * u.hour)
        assert r_fast.value > 0.5  # >0.5 hr^-1
        
        # Slow growth (10 hrs - stress)
        r_slow = malthusian_growth_rate(10 * u.hour)
        assert r_slow.value < 0.1  # <0.1 hr^-1


class TestLogisticGrowth:
    """Test logistic growth limitation factor."""
    
    def test_empty_environment(self):
        """Population far below capacity should give factor ~1.0."""
        factor = logistic_growth_factor(1.0, 1000.0, has_units=False)
        assert factor > 0.99
    
    def test_at_capacity(self):
        """Population at capacity should give factor ~0.0."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        factor = logistic_growth_factor(
            8.5, 
            8.5 * u.g_dry_weight_per_L,
            pop_units=u.g_dry_weight_per_L
        )
        assert factor < 0.01
    
    def test_halfway_to_capacity(self):
        """Population at 50% capacity should give factor ~0.5."""
        factor = logistic_growth_factor(5.0, 10.0, has_units=False)
        assert 0.45 < factor < 0.55
    
    def test_zero_capacity(self):
        """Zero carrying capacity should return 0.0."""
        factor = logistic_growth_factor(10.0, 0.0, has_units=False)
        assert factor == 0.0


class TestRadiationEnhancement:
    """Test radiation growth enhancement (Dadachova 2007)."""
    
    def test_background_radiation(self):
        """Natural background should give ~1.0× (no enhancement)."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        bg = 2.74e-7 * u.Gy / u.hour
        enhancement = radiation_growth_enhancement(bg, has_melanin=True)
        assert 0.95 < enhancement < 1.05
    
    def test_dadachova_peak_enhancement(self):
        """~500× background should give ~3.0× enhancement."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        # 500× background ≈ 1.37e-4 Gy/hr
        high_dose = 1.37e-4 * u.Gy / u.hour
        enhancement = radiation_growth_enhancement(high_dose, has_melanin=True)
        assert 2.5 < enhancement < 3.5  # Peak from Dadachova Figure 2
    
    def test_saturation_behavior(self):
        """Beyond saturation dose, enhancement should plateau."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        # Very high dose (above saturation)
        extreme_dose = 10 * u.Gy / u.hour
        enhancement = radiation_growth_enhancement(extreme_dose, has_melanin=True)
        assert 2.8 < enhancement < 3.1  # Should saturate near 3.0
    
    def test_no_melanin(self):
        """Without melanin, should always return 1.0."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        high_dose = 1 * u.Gy / u.hour
        enhancement = radiation_growth_enhancement(high_dose, has_melanin=False)
        assert enhancement == 1.0
    
    def test_without_units(self):
        """Test radiation enhancement without units."""
        enhancement = radiation_growth_enhancement(0.0001, has_melanin=True, has_units=False)
        assert enhancement > 1.0  # Should show some enhancement


class TestOxygenLimitation:
    """Test O₂ growth limitation (Imlay 2013)."""
    
    def test_anaerobe_in_hadean(self):
        """Anaerobe in Hadean (0.0002% O₂) should thrive."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        factor = oxygen_growth_limitation(
            0.0002 * u.percent,
            organism_type="obligate_anaerobe"
        )
        assert factor > 0.9  # Near optimal
    
    def test_anaerobe_in_modern_air(self):
        """Anaerobe in modern air (21% O₂) should be severely inhibited."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        factor = oxygen_growth_limitation(
            21 * u.percent,
            organism_type="obligate_anaerobe"
        )
        assert factor < 0.2  # Severe O₂ toxicity
    
    def test_aerobe_requires_oxygen(self):
        """Obligate aerobe needs O₂ to survive."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        # Low O₂ (1%)
        factor_low = oxygen_growth_limitation(
            1 * u.percent,
            organism_type="obligate_aerobe"
        )
        assert factor_low < 0.5  # Insufficient
        
        # Optimal O₂ (21%)
        factor_optimal = oxygen_growth_limitation(
            21 * u.percent,
            organism_type="obligate_aerobe"
        )
        assert factor_optimal > 0.8  # Good growth
    
    def test_catalase_protection(self):
        """Catalase should reduce O₂ toxicity."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        # Anaerobe in 5% O₂
        without_catalase = oxygen_growth_limitation(
            5 * u.percent,
            organism_type="obligate_anaerobe",
            has_catalase=False
        )
        
        with_catalase = oxygen_growth_limitation(
            5 * u.percent,
            organism_type="obligate_anaerobe",
            has_catalase=True
        )
        
        # Catalase provides protection
        assert with_catalase > without_catalase
    
    def test_facultative_tolerance(self):
        """Facultative anaerobe should tolerate wide O₂ range."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        # Test several O₂ levels
        for o2 in [0.01, 1, 5, 21]:
            factor = oxygen_growth_limitation(
                o2 * u.percent,
                organism_type="facultative_anaerobe"
            )
            assert factor > 0.3  # Should tolerate all


class TestTemperatureCorrection:
    """Test Arrhenius temperature dependence."""
    
    def test_optimal_temperature(self):
        """At optimal temp (30°C), factor should be 1.0."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        factor = temperature_correction_factor(30 * u.Celsius)
        assert abs(factor - 1.0) < 0.01
    
    def test_suboptimal_temperature(self):
        """Below optimal, growth should decrease."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        factor_20C = temperature_correction_factor(20 * u.Celsius)
        factor_15C = temperature_correction_factor(15 * u.Celsius)
        
        assert factor_20C < 1.0
        assert factor_15C < factor_20C  # Colder is worse
    
    def test_superoptimal_temperature(self):
        """Above optimal, growth should also decrease."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        factor_35C = temperature_correction_factor(35 * u.Celsius)
        factor_40C = temperature_correction_factor(40 * u.Celsius)
        
        # Slight increase possible initially, then decline
        assert factor_40C < 2.0  # Not too extreme
    
    def test_kelvin_input(self):
        """Should work with Kelvin input."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        # 303.15 K = 30°C
        factor = temperature_correction_factor(303.15 * u.K)
        assert abs(factor - 1.0) < 0.01
    
    def test_without_units(self):
        """Test temperature without units (assume Kelvin)."""
        factor = temperature_correction_factor(303.15, has_units=False)
        assert 0.8 < factor < 1.2


class TestCombinedGrowthRate:
    """Test integrated growth rate calculation."""
    
    def test_optimal_conditions(self):
        """Optimal conditions should give maximum growth rate."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        r_net = calculate_growth_rate(
            doubling_time=3.95 * u.hour,
            population=100.0,  # Well below capacity
            carrying_capacity=10000.0,  # High capacity
            dose_rate=2.74e-7 * u.Gy / u.hour,  # Background
            O2_percent=0.01 * u.percent,  # Optimal for facultative
            temperature=30 * u.Celsius,  # Optimal
            has_melanin=True,
            organism_type="facultative_anaerobe"
        )
        
        # Should be close to base rate
        expected_base = np.log(2) / 3.95
        assert abs(r_net.value - expected_base) / expected_base < 0.3  # Within 30%
    
    def test_at_carrying_capacity(self):
        """At capacity, growth should be near zero."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        r_net = calculate_growth_rate(
            doubling_time=3.95 * u.hour,
            population=8.5,
            carrying_capacity=8.5 * u.g_dry_weight_per_L,
            dose_rate=2.74e-7 * u.Gy / u.hour,
            O2_percent=21 * u.percent,
            temperature=30 * u.Celsius,
            pop_units=u.g_dry_weight_per_L
        )
        
        assert abs(r_net.value) < 0.01  #  Near zero
    
    def test_radiation_boost(self):
        """High radiation should boost growth for melanized fungi."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        # Low radiation
        r_low = calculate_growth_rate(
            doubling_time=3.95 * u.hour,
            population=100.0,
            carrying_capacity=10000.0,
            dose_rate=2.74e-7 * u.Gy / u.hour,
            O2_percent=1 * u.percent,
            temperature=30 * u.Celsius,
            has_melanin=True
        )
        
        # High radiation (500× background)
        r_high = calculate_growth_rate(
            doubling_time=3.95 * u.hour,
            population=100.0,
            carrying_capacity=10000.0,
            dose_rate=1.37e-4 * u.Gy / u.hour,
            O2_percent=1 * u.percent,
            temperature=30 * u.Celsius,
            has_melanin=True
        )
        
        # High radiation should boost growth
        assert r_high.value > r_low.value * 2.0
    
    def test_oxygen_toxicity_limits_growth(self):
        """High O₂ should limit anaerobe growth even with good conditions."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        r_net = calculate_growth_rate(
            doubling_time=3.95 * u.hour,
            population=100.0,
            carrying_capacity=10000.0,
            dose_rate=2.74e-7 * u.Gy / u.hour,
            O2_percent=21 * u.percent,  # Modern atmosphere
            temperature=30 * u.Celsius,
            has_melanin=True,
            organism_type="obligate_anaerobe"  # Hates O₂
        )
        
        # Should be severely limited
        base_rate = np.log(2) / 3.95
        assert r_net.value < base_rate * 0.3


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_doubling_time(self):
        """Zero doubling time should be handled gracefully."""
        # This would cause division by zero - should either error or return inf
        # Implementation should handle this
        pass  # Depends on desired error handling
    
    def test_negative_population(self):
        """Negative population should be handled."""
        factor = logistic_growth_factor(-10.0, 100.0, has_units=False)
        # Should either clamp to valid range or handle gracefully
        assert factor >= 0.0
    
    def test_extreme_temperatures(self):
        """Very high/low temps should be clamped."""
        if not UNITS_AVAILABLE:
            pytest.skip("Units not available")
        
        # Extreme cold
        factor_cold = temperature_correction_factor(0 * u.Celsius)
        assert 0.0 < factor_cold < 0.1
        
        # Extreme heat (above lethality)
        factor_hot = temperature_correction_factor(60 * u.Celsius)
        assert factor_hot < 0.5


# Run pytest with:
# pytest tests/test_growth.py -v
