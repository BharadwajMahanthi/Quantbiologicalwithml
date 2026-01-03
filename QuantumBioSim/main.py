import sys
import os

# Ensure the current directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import pandas as pd
from src.utils import setup_logging # type: ignore
from src.data_generation import generate_mushroom_dataset # type: ignore
from src.simulation import run_evolution # type: ignore

def main():
    parser = argparse.ArgumentParser(description="QuantumBioSim: Quantum-inspired Biological Simulation")
    parser.add_argument('--generate-data', action='store_true', help="Generate synthetic mushroom dataset")
    parser.add_argument('--run-sim', action='store_true', help="Run genetic algorithm simulation")
    parser.add_argument('--evolve', action='store_true', help="Run comparative evolutionary scenario (Human vs Fungi)")
    parser.add_argument('--big-bang', action='store_true', help="Run full evolutionary timeline (Big Bang -> Modern Era)")
    parser.add_argument('--engine', type=str, choices=['deap', 'pygad'], default='pygad', help="Evolutionary Engine")
    parser.add_argument('--output-dir', type=str, default='data', help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Setup
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    logger = setup_logging(os.path.join(args.output_dir, 'evolution.log'))
    logger.info(f"Starting QuantumBioSim (Engine: {args.engine.upper()})...")
    
    if args.generate_data:
        logger.info("Generating synthetic mushroom dataset...")
        # (existing code...)
        
    if args.run_sim:
        if args.engine == 'pygad':
            from src.pygad_engine import run_pygad_simulation
            run_pygad_simulation()
        else:
            from src.simulation import run_evolution
            run_evolution()
        
    if args.big_bang:
        if args.engine == 'pygad':
            from src.pygad_engine import run_big_bang_pygad
            run_big_bang_pygad()
        else:
            from src.simulation import run_big_bang_simulation 
            run_big_bang_simulation()
            
    if args.evolve:
        from src.data_generation import generate_evolutionary_scenario # type: ignore
        from src.models import QuantumBiologicalSystem # type: ignore
        import numpy as np
        
        # 1. Generate Data
        df = generate_mushroom_dataset(num_records=5000)
        output_path = os.path.join(args.output_dir, 'mushroom_data.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")

        # 2. Run Scenario
        logger.info("Running Comparative Evolutionary Scenario...")
        species_params = generate_evolutionary_scenario()
        
        # Initial states: Equal population for all
        initial_states = [100.0, 100.0, 100.0]
        energies = [1.0, 1.0, 1.0]
        carrying_capacity = 10000
        mutation_rate = 0.01
        
        system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params, mutation_rate) 
        
        # Run for 100 generations
        final_states = system.solve(generations=100)
        
        logger.info("Evolution Complete. Final Populations:")
        for i, params in enumerate(species_params):
            logger.info(f"{params['name']}: {final_states[i]:.2f}")
            
    if not args.generate_data and not args.run_sim and not args.evolve and not args.big_bang:
        print("No action specified. Use --generate-data, --run-sim, --evolve, or --big-bang")
        parser.print_help()

if __name__ == "__main__":
    main()
