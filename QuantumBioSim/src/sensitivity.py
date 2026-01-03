"""
Sensitivity analysis module for QuantumBioSim.

Uses Sobol indices to identify which parameters have the largest impact
on model outputs.

Usage:
    from src.sensitivity import run_sobol_analysis
    
    # Run Sobol sensitivity analysis
    results = run_sobol_analysis(n_samples=1024, scenario='big_bang')
    
    # Generate plots
    results.plot_tornado()
    results.plot_sobol_indices()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

# Try SALib import
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    logging.warning("SALib not available - sensitivity analysis disabled")

# Try matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


def setup_sobol_problem(param_bounds: Dict[str, Tuple[float, float]]) -> Dict:
    """
    Create SALib problem definition for Sobol analysis.
    
    Parameters:
    -----------
    param_bounds : dict
        Dictionary mapping parameter names to (min, max) bounds
        
    Returns:
    --------
    dict
        SALib problem definition
        
    Example:
    --------
    >>> bounds = {'doubling_time': (2.0, 6.0), 'melanin_fraction': (0.05, 0.25)}
    >>> problem = setup_sobol_problem(bounds)
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib is required for sensitivity analysis. Install with: pip install SALib")
    
    problem = {
        'num_vars': len(param_bounds),
        'names': list(param_bounds.keys()),
        'bounds': list(param_bounds.values())
    }
    
    logger.info(f"Created Sobol problem with {problem['num_vars']} parameters")
    return problem


def generate_sobol_samples(problem: Dict, n_samples: int = 1024, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate Sobol sequence samples for sensitivity analysis.
    
    Sobol sampling is more efficient than random sampling for global
    sensitivity analysis (quasi-Monte Carlo method).
    
    Parameters:
    -----------
    problem : dict
        SALib problem definition
    n_samples : int
        Base sample size (actual samples will be N*(2*D+2) where D=num_vars)
        Should ideally be a power of 2 (512, 1024, 2048)
    seed : int, optional
        Random seed
    
    Returns:
    --------
    ndarray
        Array of shape (N*(2*D+2), D) containing Sobol samples
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib is required")
    
    # Generate samples using Saltelli sampling scheme
    samples = saltelli.sample(problem, n_samples, calc_second_order=False, seed=seed)
    
    logger.info(f"Generated {samples.shape[0]} Sobol samples from base N={n_samples}")
    return samples


def evaluate_model_sobol(samples: np.ndarray, 
                         param_names: List[str],
                         scenario: str = 'big_bang') -> np.ndarray:
    """
    Evaluate model for each Sobol sample.
    
    Parameters:
    -----------
    samples : ndarray
        Sobol samples from generate_sobol_samples()
    param_names : list
        List of parameter names (same order as samples columns)
    scenario : str
        Simulation scenario
    
    Returns:
    --------
    ndarray
        Array of model outputs (one per sample)
    """
    from src.ensemble import run_single_simulation
    
    outputs = []
    
    for i in range(samples.shape[0]):
        # Create parameter dictionary
        params_dict = {name: samples[i, j] for j, name in enumerate(param_names)}
        
        # Run simulation
        result = run_single_simulation(params_dict, scenario)
        
        # Extract output
        output = result.get('final_population', np.nan)
        outputs.append(output)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Evaluated {i+1}/{samples.shape[0]} samples")
    
    return np.array(outputs)


def calculate_sobol_indices(problem: Dict, 
                            samples: np.ndarray,
                            outputs: np.ndarray,
                            calc_second_order: bool = False) -> pd.DataFrame:
    """
    Calculate Sobol sensitivity indices.
    
    S1: First-order indices (direct effect of each parameter)
    ST: Total-order indices (total effect including interactions)
    
    If ST - S1 is large, parameter has strong interactions with others.
    
    Parameters:
    -----------
    problem : dict
        SALib problem definition
    samples : ndarray
        Sobol samples used for evaluation
    outputs : ndarray
        Model outputs corresponding to samples
    calc_second_order : bool
        Calculate second-order interaction indices (expensive)
    
    Returns:
    --------
    DataFrame
        Sobol indices with columns: parameter, S1, S1_conf, ST, ST_conf
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib is required")
    
    # Remove NaN outputs
    valid_mask = ~np.isnan(outputs)
    if not valid_mask.all():
        logger.warning(f"Removing {(~valid_mask).sum()} NaN outputs")
        outputs = outputs[valid_mask]
        # Note: SALib expects full output array, so this may cause issues
        # In production, should re-run failed simulations
    
    # Calculate Sobol indices
    Si = sobol.analyze(problem, outputs, calc_second_order=calc_second_order, print_to_console=False)
    
    # Convert to DataFrame
    results = pd.DataFrame({
        'parameter': problem['names'],
        'S1': Si['S1'],
        'S1_conf': Si['S1_conf'],
        'ST': Si['ST'],
        'ST_conf': Si['ST_conf']
    })
    
    # Sort by total-order index (most important first)
    results = results.sort_values('ST', ascending=False).reset_index(drop=True)
    
    logger.info("Calculated Sobol indices")
    logger.info(f"Top 3 parameters: {results['parameter'].head(3).tolist()}")
    
    return results


def plot_tornado_diagram(sobol_results: pd.DataFrame, 
                         top_n: int = 10,
                         save_path: Optional[str] = None):
    """
    Create tornado diagram showing most sensitive parameters.
    
    Parameters:
    -----------
    sobol_results : DataFrame
        Results from calculate_sobol_indices()
    top_n : int
        Number of top parameters to show
    save_path : str, optional
        Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available")
        return
    
    # Select top N parameters
    df = sobol_results.head(top_n).copy()
    df = df.sort_values('ST')  # Sort ascending for plot
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizontal bars
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['ST'], xerr=df['ST_conf'], 
           alpha=0.7, edgecolor='black', capsize=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['parameter'])
    ax.set_xlabel('Total Sobol Index (ST)')
    ax.set_title(f'Top {top_n} Most Sensitive Parameters (Tornado Plot)')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved tornado plot to {save_path}")
    else:
        plt.show()


def plot_sobol_indices(sobol_results: pd.DataFrame,
                       save_path: Optional[str] = None):
    """
    Plot S1 vs ST for all parameters.
    
    Points above the diagonal (ST > S1) indicate parameters with
    strong interaction effects.
    
    Parameters:
    -----------
    sobol_results : DataFrame
        Results from calculate_sobol_indices()
    save_path : str, optional
        Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot S1 vs ST
    ax.errorbar(sobol_results['S1'], sobol_results['ST'],
               xerr=sobol_results['S1_conf'], yerr=sobol_results['ST_conf'],
               fmt='o', alpha=0.7, capsize=3, label='Parameters')
    
    # Add diagonal line (ST = S1)
    max_val = max(sobol_results['ST'].max(), sobol_results['S1'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='No interactions')
    
    # Label points
    for _, row in sobol_results.iterrows():
        if row['ST'] > 0.05:  # Only label significant parameters
            ax.annotate(row['parameter'], 
                       (row['S1'], row['ST']),
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('First-order Index (S1)')
    ax.set_ylabel('Total-order Index (ST)')
    ax.set_title('Sobol Sensitivity Indices\n(Above diagonal = strong interactions)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Sobol plot to {save_path}")
    else:
        plt.show()


def run_sobol_analysis(param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                       n_samples: int = 1024,
                       scenario: str = 'big_bang',
                       seed: Optional[int] = None) -> pd.DataFrame:
    """
    Run complete Sobol sensitivity analysis.
    
    Convenience function that:
    1. Sets up problem
    2. Generates Sobol samples
    3. Evaluates model
    4. Calculates indices
    
    Parameters:
    -----------
    param_bounds : dict, optional
        Parameter bounds (auto-generated from parameters.csv if None)
    n_samples : int
        Base sample size (actual will be larger)
    scenario : str
        Simulation scenario
    seed : int, optional
        Random seed
    
    Returns:
    --------
    DataFrame
        Sobol indices for all parameters
    """
    logger.info("Starting Sobol sensitivity analysis")
    
    # Generate default bounds if not provided
    if param_bounds is None:
        from src.ensemble import load_parameter_uncertainties
        params = load_parameter_uncertainties()
        
        # Create bounds as mean ± 3*std
        param_bounds = {}
        for name, (mean, std) in params.items():
            lower = max(0, mean - 3*std)  # Keep positive
            upper = mean + 3*std
            param_bounds[name] = (lower, upper)
    
    # Setup problem
    problem = setup_sobol_problem(param_bounds)
    
    # Generate samples
    samples = generate_sobol_samples(problem, n_samples, seed)
    
    # Evaluate model
    logger.info("Evaluating model for Sobol samples...")
    outputs = evaluate_model_sobol(samples, problem['names'], scenario)
    
    # Calculate indices
    results = calculate_sobol_indices(problem, samples, outputs)
    
    logger.info("Sobol analysis complete")
    return results


# Exports
__all__ = [
    'setup_sobol_problem',
    'generate_sobol_samples',
    'evaluate_model_sobol',
    'calculate_sobol_indices',
    'plot_tornado_diagram',
    'plot_sobol_indices',
    'run_sobol_analysis',
]
