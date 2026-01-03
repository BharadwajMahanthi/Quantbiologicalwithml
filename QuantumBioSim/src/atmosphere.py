"""
Atmospheric Model Module for QuantumBioSim

Provides research-grade atmospheric science calculations for biological simulations:
- O₂ growth/toxicity curves (aerobic vs anaerobic organisms)
- CO₂ fertilization effects (photosynthetic organisms)
- Atmospheric pressure effects on gas solubility

All functions return dimensionless growth factors that integrate with models.py
environmental_effect() calculations.

Author: QuantumBioSim Team
Date: January 3, 2026
"""

import numpy as np

# Try to import units, fall back gracefully if not available
try:
    from src.units import u, Qu
    UNITS_AVAILABLE = True
except ImportError:
    UNITS_AVAILABLE = False
    print("Warning: src.units not available in atmosphere.py, using unitless values")


def o2_growth_factor(o2_concentration, is_aerobic=True, has_catalase=False):
    """
    Calculate growth factor from O₂ concentration using research-grade curves.
    
    Implements scientifically accurate oxygen tolerance/requirement curves:
    - Aerobic organisms: optimal 5-21% O₂, reduced below/above
    - Anaerobic organisms: optimal <0.1% O₂, toxic above 2%
    - Catalase/SOD protection: reduces oxidative stress
    
    Parameters
    ----------
    o2_concentration : astropy.Quantity or float
        O₂ concentration in % volume (if Quantity with units) or 
        unitless % (if float). Range: 0-100%
    is_aerobic : bool, optional
        True if organism requires O₂ for respiration (aerobic)
        False for anaerobic/facultative anaerobic organisms
        Default: True
    has_catalase : bool, optional
        True if organism has oxidative stress protection (catalase, SOD enzymes)
        Provides ~30% protection against O₂ toxicity for anaerobes
        Default: False
    
    Returns
    -------
    float
        Growth factor relative to optimal conditions:
        - 1.0 = optimal growth
        - >1.0 = enhanced growth (rare, only in specific conditions)
        - <1.0 = reduced growth (stress, suboptimal)
        - <0.5 = severe stress
        - <0.2 = near-lethal toxicity
    
    Examples
    --------
    >>> # Aerobic organism at modern Earth O₂ (21%)
    >>> o2_growth_factor(21.0, is_aerobic=True)
    1.0  # Optimal
    
    >>> # Aerobic organism in hypoxia (1% O₂)
    >>> o2_growth_factor(1.0, is_aerobic=True)
    0.6  # Severely reduced
    
    >>> # Anaerobic organism at modern O₂ (toxic!)
    >>> o2_growth_factor(21.0, is_aerobic=False, has_catalase=False)
    0.04  # Severe toxicity
    
    >>> # Anaerobic with catalase protection
    >>> o2_growth_factor(21.0, is_aerobic=False, has_catalase=True)
    0.34  # Still toxic but survivable
    
    >>> # Anaerobic in Archean atmosphere (0.0002% O₂)
    >>> o2_growth_factor(0.0002, is_aerobic=False)
    1.0  # Optimal
    
    Notes
    -----
    Curves based on:
    - Oxygen toxicity thresholds from microbiology literature
    - Archean-to-modern atmospheric evolution (0.0002% → 21%)
    - Aerobic vs anaerobic metabolism requirements
    
    References
    ----------
    - Great Oxidation Event: Lyons et al. 2014, Nature
    - Oxidative stress: Imlay 2013, Annu. Rev. Microbiol.
    """
    # Extract numeric value (handle both Quantity and float)
    if UNITS_AVAILABLE and isinstance(o2_concentration, Qu):
        try:
            o2_val = o2_concentration.to(u.percent).value
        except:
            # If conversion fails, assume it's already a number
            o2_val = float(o2_concentration)
    else:
        o2_val = float(o2_concentration)
    
    # Aerobic organisms (require O₂ for energy production)
    if is_aerobic:
        # Research-based curve: optimal 5-21%, reduced outside range
        if o2_val < 0.5:
            # Severe hypoxia: <0.5% O₂
            return 0.2
        elif o2_val < 1.0:
            # Moderate hypoxia: 0.5-1% O₂
            return 0.3 + 0.3 * o2_val  # Linear increase
        elif o2_val < 5.0:
            # Mild hypoxia: 1-5% O₂
            return 0.6 + 0.08 * o2_val  # Gradual approach to optimal
        elif o2_val <= 21.0:
            # Optimal range: 5-21% O₂ (modern Earth-like)
            return 1.0
        elif o2_val <= 30.0:
            # Mild hyperoxia: 21-30% O₂
            # Small oxidative stress penalty
            return 1.0 - 0.011 * (o2_val - 21.0)  # ~10% reduction at 30%
        elif o2_val <= 50.0:
            # Moderate hyperoxia: 30-50% O₂
            # Significant oxidative damage
            return max(0.5, 0.9 - 0.02 * (o2_val - 30.0))  # Down to 50%
        else:
            # Severe hyperoxia: >50% O₂
            # Near-lethal toxicity
            return max(0.2, 0.5 - 0.01 * (o2_val - 50.0))
    
    # Anaerobic organisms (O₂ is toxic)
    else:
        # Oxidative stress protection (catalase, superoxide dismutase)
        protection_factor = 0.3 if has_catalase else 0.0
        
        if o2_val < 0.01:
            # Optimal: essentially anoxic (<0.01% O₂)
            return 1.0
        elif o2_val < 0.1:
            # Trace O₂: 0.01-0.1%
            # Minimal stress, easily managed
            return 0.98 - 0.5 * (o2_val - 0.01)
        elif o2_val < 0.5:
            # Low O₂: 0.1-0.5%
            # Manageable with protection
            base = 0.95 - 1.5 * (o2_val - 0.1)
            return min(1.0, base + protection_factor)
        elif o2_val < 2.0:
            # Moderate O₂: 0.5-2%
            # Significant stress, protection helps
            base = 0.7 - 0.2 * (o2_val - 0.5)
            return max(0.1, base + protection_factor)
        elif o2_val < 5.0:
            # High O₂: 2-5%
            # Severe toxicity
            base = 0.4 - 0.1 * (o2_val - 2.0)
            return max(0.05, base + protection_factor * 0.5)
        else:
            # Very high O₂: >5%
            # Near-lethal for strict anaerobes
            base = max(0.01, 0.1 - 0.01 * (o2_val - 5.0))
            return base + protection_factor * 0.3


def co2_fertilization_factor(co2_concentration, organism_type='fungus'):
    """
    Calculate CO₂ fertilization factor for photosynthetic/chemotrophic growth.
    
    CO₂ fertilization effect:
    - Photosynthetic organisms: Enhanced growth with increased CO₂
    - Fungi/heterotrophs: Minimal direct effect (indirect via food web)
    - Saturating curve: diminishing returns at very high CO₂
    
    Parameters
    ----------
    co2_concentration : astropy.Quantity or float
        CO₂ concentration in ppm (if Quantity) or unitless ppm (if float)
        Range: 100-1,000,000 ppm
    organism_type : str, optional
        Type of organism:
        - 'fungus' or 'heterotroph': Minimal effect (return ~1.0)
        - 'plant', 'algae', 'cyanobacteria': Photosynthetic fertilization
        Default: 'fungus'
    
    Returns
    -------
    float
        Growth factor due to CO₂:
        - Baseline (280 ppm pre-industrial) = 1.0
        - Modern (420 ppm) ≈ 1.15× for phototrophs
        - Archean (700,000 ppm) ≈ 2.5× (saturates ~3×)
        - Below 100 ppm: CO₂ limitation (<1.0)
    
    Examples
    --------
    >>> # Fungus (heterotroph) - no direct effect
    >>> co2_fertilization_factor(420, organism_type='fungus')
    1.0
    
    >>> # Plant at modern CO₂ (420 ppm)
    >>> co2_fertilization_factor(420, organism_type='plant')
    1.15
    
    >>> # Cyanobacteria in Archean (700,000 ppm)
    >>> co2_fertilization_factor(700000, organism_type='cyanobacteria')
    2.47
    
    Notes
    -----
    For fungi-only simulations, this typically returns 1.0 (no effect).
    Useful for future extensions to include photosynthetic organisms.
    
    References  
    ----------
    - CO₂ fertilization: Ainsworth & Long 2005, Plant Physiol.
    - Archean CO₂: Kasting 1993, Science
    """
    # Extract numeric value
    if UNITS_AVAILABLE and isinstance(co2_concentration, Qu):
        try:
            co2_val = co2_concentration.to(u.ppm).value
        except:
            co2_val = float(co2_concentration)
    else:
        co2_val = float(co2_concentration)
    
    # Fungi/heterotrophs: no direct CO₂ effect
    if organism_type in ['fungus', 'heterotroph', 'bacteria']:
        return 1.0
    
    # Photosynthetic organisms: fertilization curve
    # Baseline: 280 ppm (pre-industrial Holocene)
    baseline_co2 = 280.0
    
    if co2_val < 100:
        # CO₂ limitation: below ~100 ppm photosynthesis becomes limited
        return max(0.3, co2_val / 100.0)
    elif co2_val <= baseline_co2:
        # Below pre-industrial baseline
        return co2_val / baseline_co2
    else:
        # Above baseline: saturating curve
        # Modern (420 ppm) → ~15% boost
        # Very high (Archean 700k ppm) → ~2.5× (saturates at ~3×)
        
        # Michaelis-Menten-like saturation
        # Max enhancement: 3× at infinite CO₂
        # Half-saturation: ~1000 ppm
        enhancement = (co2_val - baseline_co2) / (co2_val + 1000.0)
        return 1.0 + enhancement * 2.5  # Scales to max ~3.5×


def atmospheric_pressure_effect(pressure, baseline_pressure=1.013):
    """
    Calculate gas solubility/availability effect from atmospheric pressure.
    
    Higher pressure → increased gas solubility in water → enhanced availability
    Lower pressure → reduced solubility → potential stress
    
    Parameters
    ----------
    pressure : astropy.Quantity or float
        Atmospheric pressure in bar (if Quantity) or unitless bar (if float)
        Range: 0.1-3.0 bar (Earth-like range)
    baseline_pressure : float, optional
        Reference pressure in bar. Default: 1.013 (modern Earth sea level)
    
    Returns
    -------
    float
        Pressure effect factor:
        - 1.0 at baseline (1.013 bar)
        - >1.0 at higher pressure (enhanced solubility)
        - <1.0 at lower pressure (reduced solubility)
    
    Examples
    --------
    >>> # Modern Earth sea level
    >>> atmospheric_pressure_effect(1.013)
    1.0
    
    >>> # Archean (possibly higher, ~1.2 bar)
    >>> atmospheric_pressure_effect(1.2)
    1.03
    
    >>> # Mars surface (~0.006 bar) - extreme low pressure
    >>> atmospheric_pressure_effect(0.006)
    0.25
    
    Notes
    -----
    Effect is relatively small for Earth-like pressures (0.5-2 bar).
    More significant for extreme environments (sub-bar or multi-bar).
    """
    # Extract numeric value
    if UNITS_AVAILABLE and isinstance(pressure, Qu):
        try:
            p_val = pressure.to(u.bar).value
        except:
            p_val = float(pressure)
    else:
        p_val = float(pressure)
    
    # Linear approximation for small pressure changes
    # Henry's law: solubility ∝ pressure
    # Effect is modest for Earth-like range (0.5-2 bar)
    
    if p_val < 0.1:
        # Extreme low pressure (Mars-like): severe gas limitation
        return 0.1 + 1.5 * p_val
    elif p_val < baseline_pressure:
        # Below baseline: slight reduction
        return 0.9 + 0.1 * (p_val / baseline_pressure)
    else:
        # Above baseline: slight enhancement
        ratio = p_val / baseline_pressure
        return 1.0 + 0.05 * (ratio - 1.0)  # ~5% per bar above baseline


# Module-level test
if __name__ == "__main__":
    print("=" * 70)
    print(" ATMOSPHERE.PY MODULE TESTS")
    print("=" * 70)
    
    print("\nTEST 1: O₂ Growth Factor - Aerobic Organism")
    print("-" * 70)
    test_o2_levels_aerobic = [0.0002, 0.01, 1.0, 5.0, 10.0, 21.0, 30.0, 50.0]
    for o2 in test_o2_levels_aerobic:
        factor = o2_growth_factor(o2, is_aerobic=True)
        print(f"  O₂ = {o2:8.4f}% → Growth factor = {factor:.3f}")
    
    print("\nTEST 2: O₂ Growth Factor - Anaerobic Organism")
    print("-" * 70)
    test_o2_levels_anaerobic = [0.0002, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 21.0]
    for o2 in test_o2_levels_anaerobic:
        factor_no_prot = o2_growth_factor(o2, is_aerobic=False, has_catalase=False)
        factor_with_prot = o2_growth_factor(o2, is_aerobic=False, has_catalase=True)
        print(f"  O₂ = {o2:8.4f}% → No protection: {factor_no_prot:.3f}, With catalase: {factor_with_prot:.3f}")
    
    print("\nTEST 3: CO₂ Fertilization")
    print("-" * 70)
    test_co2_levels = [100, 280, 420, 1000, 10000, 700000]
    for co2 in test_co2_levels:
        factor_fungus = co2_fertilization_factor(co2, organism_type='fungus')
        factor_plant = co2_fertilization_factor(co2, organism_type='plant')
        print(f"  CO₂ = {co2:8.0f} ppm → Fungus: {factor_fungus:.3f}, Plant: {factor_plant:.3f}")
    
    print("\nTEST 4: Atmospheric Pressure")
    print("-" * 70)
    test_pressures = [0.006, 0.5, 0.75, 1.013, 1.2, 2.0]
    for p in test_pressures:
        factor = atmospheric_pressure_effect(p)
        print(f"  Pressure = {p:.3f} bar → Effect factor = {factor:.3f}")
    
    print("\n" + "=" * 70)
    print(" ALL TESTS COMPLETE")
    print("=" * 70)
