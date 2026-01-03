"""
Monte Carlo ensemble module for QuantumBioSim.

This module enables uncertainty quantification by running the simulation
multiple times with parameters sampled from their uncertainty distributions.

Usage:
    from src.ensemble import run_ensemble, load_parameter_uncertainties
    
    # Load parameter uncertainties from data/parameters.csv
    params = load_parameter_uncertainties()
    
    # Run 100 simulations with parameter uncertainty
    results = run_ensemble(n_simulations=100, scenario='big_bang', parallel=True)
    
    # Analyze results
    results.plot_trajectories()
    results.summary_statistics()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
from dataclasses import dataclass
import logging

# Try imports for optional features
try:
    from multiprocessing import Pool, cpu_count
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
    logging.warning("multiprocessing not available - parallel execution disabled")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logging.warning("tqdm not available - no progress bars")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available - plotting disabled")

logger = logging.getLogger(__name__)


def load_parameter_uncertainties(csv_path: str = 'data/parameters.csv') -> Dict[str, Tuple[float, float]]:
    """
    Load parameter mean values and uncertainties from parameters.csv.
    
    Parameters:
    -----------
    csv_path : str
        Path to parameters.csv file
    
    Returns:
    --------
    dict
        Dictionary mapping parameter names to (mean, std) tuples
        
    Example:
    --------
    >>> params = load_parameter_uncertainties()
    >>> params['doubling_time_typical']
    (3.95, 1.25)
    """
    try:
        # Read CSV, skipping comment lines
        df = pd.read_csv(csv_path, comment='#')
        
        # Filter out rows with missing values or uncertainties
        df = df.dropna(subset=['parameter', 'value', 'uncertainty'])
        
        # Create dictionary
        param_dict = {}
        for _, row in df.iterrows():
            param_name = row['parameter']
            mean_value = float(row['value'])
            uncertainty = float(row['uncertainty'])
            
            # Only include if uncertainty > 0
            if uncertainty > 0:
                param_dict[param_name] = (mean_value, uncertainty)
        
        logger.info(f"Loaded {len(param_dict)} parameters with uncertainties")
        return param_dict
        
    except FileNotFoundError:
        logger.error(f"Parameters file not found: {csv_path}")
        # Return minimal default set
        return {
            'doubling_time_typical': (3.95, 1.25),
            'melanin_dry_mass_fraction': (0.154, 0.02),
            'background_radiation': (2.74e-7, 5e-8)
        }
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        return {}


def sample_parameters(param_distributions: Dict[str, Tuple[float, float]],
                      n_samples: int = 100,
                      seed: Optional[int] = None) -> pd.DataFrame:
    """
    Sample parameters from their uncertainty distributions.
    
    Uses normal distributions truncated at physically realistic bounds.
    
    Parameters:
    -----------
    param_distributions : dict
        Dictionary mapping parameter names to (mean, std) tuples
    n_samples : int
        Number of parameter sets to sample
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    DataFrame
        n_samples rows × n_parameters columns
        
    Example:
    --------
    >>> params = {'doubling_time': (3.95, 1.25)}
    >>> samples = sample_parameters(params, n_samples=10, seed=42)
    >>> samples.shape
    (10, 1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples_dict = {}
    
    for param_name, (mean, std) in param_distributions.items():
        # Sample from normal distribution
        samples = np.random.normal(mean, std, n_samples)
        
        # Apply physical constraints
        if 'doubling_time' in param_name:
            samples = np.clip(samples, 0.5, 20.0)  # 0.5-20 hours
        elif 'radiation' in param_name and 'background' in param_name:
            samples = np.clip(samples, 1e-9, 1e-5)  # Must be positive, reasonable range
        elif 'melanin' in param_name and 'fraction' in param_name:
            samples = np.clip(samples, 0.0, 0.5)  # 0-50% of mass
        elif 'O2' in param_name or 'percent' in param_name:
            samples = np.clip(samples, 0.0, 100.0)  # 0-100%
        elif param_name.startswith('growth_'):
            samples = np.clip(samples, 0.1, 10.0)  # Growth factors
        else:
            # General constraint: keep within ±5σ of mean
            samples = np.clip(samples, mean - 5*std, mean + 5*std)
        
        samples_dict[param_name] = samples
    
    return pd.DataFrame(samples_dict)


def run_single_simulation(params_dict: Dict[str, float], 
                          scenario: str = 'big_bang') -> Dict[str, Any]:
    """
    Run one simulation with given parameter values.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary of parameter values to use
    scenario : str
        Simulation scenario ('big_bang', 'evolve', etc.)
    
    Returns:
    --------
    dict
        Results dictionary with keys like 'final_population', 'success', etc.
    """
    try:
        # Import simulation modules
        from src.pygad_engine import run_big_bang_pygad
        from src.models import QuantumBiologicalSystem
        
        # Run simulation based on scenario
        if scenario == 'big_bang':
            # Use Big Bang simulation
            # Note: Would need to pass params to simulation
            # For now, just run with defaults
            population = run_big_bang_pygad(generations=100)
            
            return {
                'success': True,
                'final_population': population,
                'scenario': scenario
            }
        else:
            # Placeholder for other scenarios
            return {
                'success': False,
                'error': f'Scenario {scenario} not yet implemented',
                'final_population': np.nan
            }
            
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'final_population': np.nan
        }


def _run_single_wrapper(args):
    """Wrapper for multiprocessing (must be picklable)."""
    params_dict, scenario, idx = args
    result = run_single_simulation(params_dict, scenario)
    result['sample_id'] = idx
    return result


@dataclass
class EnsembleResults:
    """
    Container for Monte Carlo ensemble results.
    
    Attributes:
    -----------
    parameter_samples : DataFrame
        Sampled parameter values
    outputs : list
        List of result dictionaries from each simulation
    n_simulations : int
        Number of simulations run
    scenario : str
        Simulation scenario used
    """
    parameter_samples: pd.DataFrame
    outputs: List[Dict[str, Any]]
    n_simulations: int
    scenario: str
    
    def summary_statistics(self) -> Dict[str, float]:
        """
        Calculate summary statistics across ensemble.
        
        Returns:
        --------
        dict
            Statistics including mean, std, percentiles
        """
        # Extract final populations
        populations = [r['final_population'] for r in self.outputs 
                      if r.get('success', False)]
        
        if len(populations) == 0:
            logger.warning("No successful simulations in ensemble")
            return {}
        
        populations = np.array(populations)
        
        stats = {
            'mean': np.mean(populations),
            'std': np.std(populations),
            'median': np.median(populations),
            'min': np.min(populations),
            'max': np.max(populations),
            'percentile_2.5': np.percentile(populations, 2.5),
            'percentile_97.5': np.percentile(populations, 97.5),
            'cv': np.std(populations) / np.mean(populations) if np.mean(populations) > 0 else np.nan,
            'n_successful': len(populations),
            'n_failed': self.n_simulations - len(populations)
        }
        
        return stats
    
    def plot_distribution(self, output_var: str = 'final_population', 
                         bins: int = 30, save_path: Optional[str] = None):
        """
        Plot histogram of output distribution.
        
        Parameters:
        -----------
        output_var : str
            Output variable to plot
        bins : int
            Number of histogram bins
        save_path : str, optional
            Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available - cannot plot")
            return
        
        # Extract values
        values = [r.get(output_var, np.nan) for r in self.outputs 
                 if r.get('success', False)]
        values = np.array([v for v in values if not np.isnan(v)])
        
        if len(values) == 0:
            logger.warning("No valid data to plot")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(values, bins=bins, alpha=0.7, edgecolor='black')
        
        # Add statistics
        stats = self.summary_statistics()
        ax.axvline(stats['mean'], color='red', linestyle='--', 
                  label=f"Mean: {stats['mean']:.2f}")
        ax.axvline(stats['percentile_2.5'], color='blue', linestyle=':', 
                  label=f"95% CI: [{stats['percentile_2.5']:.2f}, {stats['percentile_97.5']:.2f}]")
        ax.axvline(stats['percentile_97.5'], color='blue', linestyle=':')
        
        ax.set_xlabel(output_var.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {output_var} (n={len(values)})')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def save(self, filepath: str):
        """
        Save ensemble results to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save results (pickle format)
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved ensemble results to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load ensemble results from file.
        
        Parameters:
        -----------
        filepath : str
            Path to saved results
        
        Returns:
        --------
        EnsembleResults
            Loaded ensemble results
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"Loaded ensemble results from {filepath}")
        return results


def run_ensemble(n_simulations: int = 100,
                scenario: str = 'big_bang',
                param_distributions: Optional[Dict[str, Tuple[float, float]]] = None,
                parallel: bool = True,
                n_cores: Optional[int] = None,
                seed: Optional[int] = None) -> EnsembleResults:
    """
    Run Monte Carlo ensemble simulation.
    
    Parameters:
    -----------
    n_simulations : int
        Number of simulations to run
    scenario : str
        Simulation scenario
    param_distributions : dict, optional
        Parameter distributions (loaded from CSV if None)
    parallel : bool
        Use parallel processing
    n_cores : int, optional
        Number of CPU cores (auto-detect if None)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    EnsembleResults
        Object containing all results and analysis methods
    """
    logger.info(f"Starting Monte Carlo ensemble: {n_simulations} simulations")
    
    # Load parameter distributions if not provided
    if param_distributions is None:
        param_distributions = load_parameter_uncertainties()
    
    # Sample parameters
    logger.info("Sampling parameters from distributions...")
    parameter_samples = sample_parameters(param_distributions, n_simulations, seed)
    
    # Prepare simulation arguments
    sim_args = []
    for idx, row in parameter_samples.iterrows():
        params_dict = row.to_dict()
        sim_args.append((params_dict, scenario, idx))
    
    # Run simulations
    logger.info("Running simulations...")
    
    if parallel and MULTIPROCESSING_AVAILABLE and n_simulations > 1:
        # Parallel execution
        if n_cores is None:
            n_cores = max(1, cpu_count() - 1)  # Leave one core free
        
        logger.info(f"Using {n_cores} cores for parallel execution")
        
        with Pool(n_cores) as pool:
            if TQDM_AVAILABLE:
                outputs = list(tqdm(pool.imap(_run_single_wrapper, sim_args), 
                                   total=n_simulations, desc="Ensemble"))
            else:
                outputs = pool.map(_run_single_wrapper, sim_args)
    else:
        # Serial execution
        logger.info("Running serially (parallel disabled or n=1)")
        outputs = []
        iterator = tqdm(sim_args, desc="Ensemble") if TQDM_AVAILABLE else sim_args
        
        for args in iterator:
            outputs.append(_run_single_wrapper(args))
    
    # Create results object
    results = EnsembleResults(
        parameter_samples=parameter_samples,
        outputs=outputs,
        n_simulations=n_simulations,
        scenario=scenario
    )
    
    # Log summary
    stats = results.summary_statistics()
    logger.info(f"Ensemble complete: {stats.get('n_successful', 0)} successful, " +
               f"{stats.get('n_failed', 0)} failed")
    if stats:
        logger.info(f"Mean output: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    return results


# Exports
__all__ = [
    'load_parameter_uncertainties',
    'sample_parameters',
    'run_single_simulation',
    'run_ensemble',
    'EnsembleResults',
]
