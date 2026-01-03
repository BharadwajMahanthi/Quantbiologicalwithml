"""
Test suite for sensitivity.py - Sobol sensitivity analysis module.

Tests SALib integration, Sobol sampling, and index calculations.
"""

import pytest
import numpy as np
import pandas as pd

# Check if SALib is available
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False

from src.sensitivity import (
    setup_sobol_problem,
    generate_sobol_samples,
    calculate_sobol_indices,
    run_sobol_analysis
)


@pytest.mark.skipif(not SALIB_AVAILABLE, reason="SALib not installed")
def test_setup_sobol_problem():
    """Test creation of SALib problem definition."""
    param_bounds = {
        'doubling_time': (2.0, 6.0),
        'melanin_fraction': (0.05, 0.25),
        'radiation_dose': (1e-8, 1e-5)
    }
    
    problem = setup_sobol_problem(param_bounds)
    
    # Check structure
    assert 'num_vars' in problem
    assert 'names' in problem
    assert 'bounds' in problem
    
    # Check values
    assert problem['num_vars'] == 3
    assert problem['names'] == ['doubling_time', 'melanin_fraction', 'radiation_dose']
    assert len(problem['bounds']) == 3
    assert problem['bounds'][0] == (2.0, 6.0)


@pytest.mark.skipif(not SALIB_AVAILABLE, reason="SALib not installed")
def test_generate_sobol_samples():
    """Test Sobol sequence generation."""
    problem = {
        'num_vars': 2,
        'names': ['param1', 'param2'],
        'bounds': [(0.0, 1.0), (0.0, 1.0)]
    }
    
    samples = generate_sobol_samples(problem, n_samples=128, seed=42)
    
    # Check shape: N * (2*D + 2) = 128 * (2*2 + 2) = 768
    assert samples.shape[1] == 2  # 2 parameters
    assert samples.shape[0] == 128 * (2*2 + 2)  # Saltelli sampling scheme
    
    # Check bounds
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0


@pytest.mark.skipif(not SALIB_AVAILABLE, reason="SALib not installed")
def test_sobol_samples_reproducibility():
    """Test that same seed gives same Sobol samples."""
    problem = {
        'num_vars': 2,
        'names': ['param1', 'param2'],
        'bounds': [(0.0, 1.0), (0.0, 1.0)]
    }
    
    samples1 = generate_sobol_samples(problem, n_samples=64, seed=42)
    samples2 = generate_sobol_samples(problem, n_samples=64, seed=42)
    
    np.testing.assert_array_equal(samples1, samples2)


@pytest.mark.skipif(not SALIB_AVAILABLE, reason="SALib not installed")
def test_calculate_sobol_indices_simple():
    """Test Sobol index calculation with known model."""
    # Test with Ishigami function (standard test function for sensitivity)
    # Y = sin(param1) + 7*sin(param2)^2 + 0.1*param3^4*sin(param1)
    
    problem = {
        'num_vars': 3,
        'names': ['X1', 'X2', 'X3'],
        'bounds': [[-np.pi, np.pi]] * 3
    }
    
    # Generate samples
    samples = generate_sobol_samples(problem, n_samples=512, seed=42)
    
    # Evaluate Ishigami function
    X1 = samples[:, 0]
    X2 = samples[:, 1]
    X3 = samples[:, 2]
    outputs = np.sin(X1) + 7 * np.sin(X2)**2 + 0.1 * X3**4 * np.sin(X1)
    
    # Calculate indices
    results = calculate_sobol_indices(problem, samples, outputs)
    
    # Check structure
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 3
    assert 'parameter' in results.columns
    assert 'S1' in results.columns
    assert 'ST' in results.columns
    
    # For Ishigami function, X2 should be most important
    # (This is a known result, but with finite samples may vary)
    assert all(results['S1'] >= -0.2)  # S1 should be non-negative (allow small numerical error)
    assert all(results['ST'] >= 0.0)  # ST should definitely be non-negative
    assert all(results['ST'] >= results['S1'] - 0.3)  # ST >= S1 (with tolerance)


@pytest.mark.skipif(not SALIB_AVAILABLE, reason="SALib not installed")
def test_sobol_indices_sum():
    """Test that first-order indices sum to approximately 1 for additive model."""
    # Create simple additive model: Y = X1 + 2*X2 + 3*X3
    problem = {
        'num_vars': 3,
        'names': ['X1', 'X2', 'X3'],
        'bounds': [[0, 1], [0, 1], [0, 1]]
    }
    
    samples = generate_sobol_samples(problem, n_samples=512, seed=42)
    
    # Additive model
    outputs = samples[:, 0] + 2*samples[:, 1] + 3*samples[:, 2]
    
    results = calculate_sobol_indices(problem, samples, outputs)
    
    # For purely additive model, sum of S1 should ≈ 1
    s1_sum = results['S1'].sum()
    assert 0.8 < s1_sum < 1.2, f"S1 sum should be ~1 for additive model, got {s1_sum}"


@pytest.mark.skipif(not SALIB_AVAILABLE, reason="SALib not installed")
def test_sobol_sorted_by_importance():
    """Test that results are sorted by total-order index."""
    problem = {
        'num_vars': 3,
        'names': ['X1', 'X2', 'X3'],
        'bounds': [[0, 1]] * 3
    }
    
    samples = generate_sobol_samples(problem, n_samples=256, seed=42)
    # Model where X1 dominant: Y = 10*X1 + X2 + 0.1*X3
    outputs = 10*samples[:, 0] + samples[:, 1] + 0.1*samples[:, 2]
    
    results = calculate_sobol_indices(problem, samples, outputs)
    
    # Results should be sorted descending by ST
    assert results['ST'].is_monotonic_decreasing
    
    # X1 should be first (most important)
    assert results.iloc[0]['parameter'] == 'X1'


def test_sensitivity_without_salib():
    """Test that module gracefully handles missing SALib."""
    if SALIB_AVAILABLE:
        pytest.skip("SALib is installed, cannot test fallback")
    
    # Should raise ImportError with helpful message
    with pytest.raises(ImportError, match="SALib is required"):
        setup_sobol_problem({'param': (0, 1)})


@pytest.mark.skipif(not SALIB_AVAILABLE, reason="SALib not installed")
def test_sobol_with_nan_handling():
    """Test handling of NaN outputs from failed simulations."""
    problem = {
        'num_vars': 2,
        'names': ['X1', 'X2'],
        'bounds': [[0, 1], [0, 1]]
    }
    
    samples = generate_sobol_samples(problem, n_samples=128, seed=42)
    
    # Create outputs with some NaNs
    outputs = samples[:, 0] + samples[:, 1]
    outputs[10:20] = np.nan  # Simulate 10 failed simulations
    
    # Should still work (SALib handles this)
    # Note: Current implementation removes NaNs with a warning
    # In production, should re-run failed sims
    with pytest.warns(UserWarning, match="NaN"):
        results = calculate_sobol_indices(problem, samples, outputs)
    
    # Should still return results
    assert len(results) == 2


# Integration test (slow, marked for optional execution)
@pytest.mark.slow
@pytest.mark.skipif(not SALIB_AVAILABLE, reason="SALib not installed")
def test_run_sobol_analysis_integration():
    """Integration test for complete Sobol analysis workflow."""
    # Use small sample size for speed
    param_bounds = {
        'doubling_time_typical': (2.0, 6.0),
        'melanin_fraction': (0.1, 0.3)
    }
    
    results = run_sobol_analysis(
        param_bounds=param_bounds,
        n_samples=128,  # Small for test speed
        scenario='big_bang',
        seed=42
    )
    
    # Check results
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert 'S1' in results.columns
    assert 'ST' in results.columns
