import numpy as np
import pygad
import logging
import gc  # For garbage collection and cache clearing
from .models import QuantumBiologicalSystem

logger = logging.getLogger('quantum_bio_system')
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s', force=True)


# Global context for Environment (Updated per Epoch)
SIM_CONTEXT = {
    "env_conditions": {
        "radiation": 1.0,
        "oxygen": 21.0, "co2": 0.04, "pollution": 0.0,
        "water": 1.0, "food": 1.0
    }
}

class QuantumPyGADWrapper:
    def __init__(self, num_species=10, dna_size=100):
        self.num_species = num_species
        self.dna_size = dna_size
        self.genome_length = num_species * dna_size
        
        # DNA Mapping: 0=A, 1=T, 2=C, 3=G
        self.gene_space = [0, 1, 2, 3] # Discrete gene space
        
    def decode_solution(self, solution):
        # solution is a numpy array of ints
        mapping = ['A', 'T', 'C', 'G']
        return "".join([mapping[int(gene)] for gene in solution])

    def fitness_func(self, ga_instance, solution, solution_idx):
        dna_full = self.decode_solution(solution)
        
        # Reconstruct Species Params (Just one species representative for PyGAD optimization usually)
        # But our genome is concatenated params for 'num_species'.
        species_params = []
        for i in range(self.num_species):
            start = i * self.dna_size
            end = start + self.dna_size
            dna_seq = dna_full[start:end]
            
            params = {
                'dna_sequence': dna_seq,
                'replication_rate': 0.1, 'decay_rate': 0.1, 'aging_rate': 0.01,
                'lifespan': 20, 'anti_life_effect': 1.0, 
                'birth_rate': 0.05,
                'prob_same_species': 0.01, 'prob_diff_species': 0.0,
                'interaction_strength': np.zeros(self.num_species)
            }
            species_params.append(params)
            
        # Run Simulation
        env_conditions = SIM_CONTEXT["env_conditions"]
        
        system = QuantumBiologicalSystem(
            initial_states=np.full(self.num_species, 50.0),
            energies=np.zeros(self.num_species),
            carrying_capacity=10000, # Increased for realistic war/growth dynamics
            species_params=species_params,
            mutation_rate=0.0,
            env_conditions=env_conditions
        )
        
        # We catch potential errors in physics to prevent crash
        try:
            # Short run to evaluate fitness in this environment
            # Optimization: 2 gens is enough for derivative check, fast for loop
            final_states = system.solve(generations=2)
            fitness = np.sum(final_states)
            # Prevent negative fitness (PyGAD likes maximization)
            return float(max(float(fitness), 0.0))
        except Exception:
            return 0.0

def run_pygad_simulation(pop_size=50, generations=50, radiation=1.0):
    wrapper = QuantumPyGADWrapper()
    # Simple run defaults to Modern Earth context if not specified beyond radiation
    SIM_CONTEXT["env_conditions"]["radiation"] = radiation
    
    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=int(pop_size/2),
        fitness_func=wrapper.fitness_func,
        sol_per_pop=pop_size,
        num_genes=wrapper.genome_length,
        gene_space=wrapper.gene_space,
        gene_type=int,
        mutation_percent_genes=1, 
        parent_selection_type="rank",
        crossover_type="single_point",
        mutation_type="random",
        keep_parents=1,
        suppress_warnings=True
    )
    
    ga_instance.run()
    
    solution, solution_fitness, _ = ga_instance.best_solution()
    logger.info(f"Best Fitness: {solution_fitness}")
    return solution, solution_fitness

def run_big_bang_pygad():
    # [NEW] OpenMC Integration
    # "Use OpenMC, you can see the materials"
    # We load the static XML files to define the physical substrate of the simulation.
    try:
        from .openmc_adapter import OpenMCAdapter
        import os
        # Get the correct absolute path to openmc_static
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        materials_path = os.path.join(os.path.dirname(project_root), 'openmc_static', 'materials.xml')
        adapter = OpenMCAdapter(materials_path)
        materials = adapter.load_materials()
        impact = adapter.get_environmental_impact()
        
        if materials:
            logger.info("\n=== OpenMC Material Analysis ===")
            for name, props in materials.items():
                logger.info(f"Material: {name} | Density: {props['density']} g/cm3")
                logger.info(f"  Composition: {props['composition']}")
            logger.info("================================\n")
            
            logger.info(f"OpenMC Environmental Modifiers: {impact}")

            logger.info(f"OpenMC Environmental Modifiers: {impact}")

        # [NEW] Real OpenMC Integration
        # Attempt to use the OFFICIAL OpenMC library if installed.
        try:
            from .real_openmc_model import RealOpenMCModel
            model = RealOpenMCModel()
            model.build_model() # Generates the XMLs using official API
            
            # This requires the 'openmc' executable in PATH
            heating = model.run()
            
            if heating:
                logger.info(f"OpenMC Simulation Successful. Heating Load: {heating}")
                # Real Physics Feedback: High heating = High Absorption = Good Shielding
                if heating > 1000.0:
                    impact['radiation_shielding'] *= 1.2
            else:
                 logger.warning("OpenMC binary did not run (Native Build Required). Simulation utilizing static XML data only.")
                 
        except ImportError:
            logger.warning("OpenMC Python module not installed. Skipping Real Monte Carlo step.")
        except Exception as e:
            logger.warning(f"OpenMC Error: {e}")
            
    except Exception as e:
        logger.warning(f"Failed to load OpenMC materials: {e}")
        impact = {'radiation_shielding': 0.0, 'radiation_source': 0.0, 'oxygen_level': 0.0}
    
    # Clear memory after OpenMC operations
    gc.collect()

    # Timeline Definition
    # We apply OpenMC impacts to the base environment
    
    # Extract modifiers
    rad_source = impact.get('radiation_source', 0.0)
    shield = impact.get('radiation_shielding', 0.0)
    
    # CRITICAL FIX: Disable OpenMC oxygen for geological Big Bang simulation
    # OpenMC calculates oxygen from modern material composition (air = 21% O₂)
    # This is inappropriate for geological epochs (Hadean should be 0.0002% O₂, not 7.5%)
    # For Big Bang mode, we use research-grade epoch-specific O₂ from parameters.csv
    oxy_source = 0  # Was: impact.get('oxygen_level', 0.0) - disabled for geological accuracy

    logger.info("Starting Big Bang Timeline (Detailed Physics: Air, Water, Food)...")
    logger.info("Timeline: Hadean -> Anthropocene (1800 Generations Total)")
    
    # [DETAILED EPOCHS]
    # 1. Hadean: Hell on Earth. High Rad, Low O2, Volcanic Pollution.
    # 2. Archean: Cooling. Rising O2 (Cyanobacteria).
    # 3. Great Oxidation: The Oxygen Crisis. Anaerobes die. Heme evolves.
    # 4. Phanerozoic: Cambrian Explosion. High O2, stable.
    # 5. Industrial (1800 AD): Start of Pollution.
    # 6. Anthropocene: High Pollution check.
    
    # Load research-grade O₂ values from parameters.csv
    try:
        import pandas as pd
        _params = pd.read_csv('data/parameters.csv', comment='#')
        def _get_param(name):
            row = _params[_params['parameter'] == name]
            return float(row.iloc[0]['value']) if len(row) > 0 else None
        
        O2_hadean = _get_param('O2_hadean') or 2e-6  # 0.0002%
        O2_archean_late = _get_param('O2_archean_late') or 0.01  # 0.01%
        O2_GOE = _get_param('O2_GOE') or 1.0  # 1%
        O2_phanerozoic = _get_param('O2_phanerozoic') or 18.0  # 18%
        O2_modern = _get_param('O2_modern') or 21.0  # 21%
        CO2_hadean = (_get_param('CO2_hadean') or 120000) / 10000  # Convert ppm to %
        CO2_archean = (_get_param('CO2_archean_late') or 5000) / 10000
        CO2_modern = (_get_param('CO2_modern') or 420) / 10000
    except:
        # Fallback to improved values (still better than old hard-coded)
        O2_hadean = 0.0002
        O2_archean_late = 0.01
        O2_GOE = 1.0
        O2_phanerozoic = 18.0
        O2_modern = 21.0
        CO2_hadean = 12.0  # 120,000 ppm
        CO2_archean = 0.5  # 5,000 ppm
        CO2_modern = 0.042  # 420 ppm
    
    epochs = [
        {
            "name": "Hadean Eon (4.0 Ga)", 
            "env": {
                "radiation": max(0.1, 50.0 + rad_source - shield), 
                "oxygen": O2_hadean + oxy_source,  # Research: 0.0002%
                "co2": CO2_hadean, "pollution": 5.0, 
                "water": 0.5, "food": 1.0,
                "plague_risk": 0.05, "war_risk": 0.01 
            },
            "generations": 300
        },
        {
            "name": "Archean Eon (2.5 Ga)", 
            "env": {
                "radiation": max(0.1, 10.0 + rad_source - shield), 
                "oxygen": O2_archean_late + oxy_source,  # Research: 0.01%
                "co2": CO2_archean, "pollution": 1.0, 
                "water": 1.0, "food": 0.8,
                "plague_risk": 0.01, "war_risk": 0.05
            },
            "generations": 300
        },
        {
            "name": "Great Oxidation (2.0 Ga)", 
            "env": {
                "radiation": max(0.1, 2.0 + rad_source - shield), 
                "oxygen": O2_GOE + oxy_source,  # Research: 1% (GOE rise)
                "co2": 1.5, "pollution": 0.5,  # Post-GOE decline
                "water": 1.0, "food": 0.6,
                "plague_risk": 0.15, "war_risk": 0.0 # Oxygen Crisis Purge
            },
            "generations": 300
        },
        {
            "name": "Phanerozoic (0.5 Ga)", 
            "env": {
                "radiation": max(0.1, 0.5 + rad_source - shield), 
                "oxygen": O2_phanerozoic + oxy_source,  # Research: 18%
                "co2": 0.7, "pollution": 0.0,  # Phanerozoic average
                "water": 1.0, "food": 1.0,
                "plague_risk": 0.02, "war_risk": 0.05
            },
            "generations": 300
        },
        {
            "name": "Industrial Rev (1800 AD)", 
            "env": {
                "radiation": max(0.1, 0.1 + rad_source - shield), 
                "oxygen": O2_modern + oxy_source,  # Research: 21%
                "co2": 0.028, "pollution": 2.0,  # Pre-industrial 280 ppm
                "water": 1.0, "food": 2.0,
                "plague_risk": 0.05, "war_risk": 0.20 # Human Conflict
            },
            "generations": 300
        },
        {
            "name": "Anthropocene (Future)", 
            "env": {
                "radiation": max(0.1, 0.2 + rad_source - shield), 
                "oxygen": O2_modern + oxy_source,  # Research: 21%
                "co2": CO2_modern, "pollution": 20.0,  # Modern 420 ppm
                "water": 0.4, "food": 0.5,
                "plague_risk": 0.10, "war_risk": 0.30 # Resource Wars
            },
            "generations": 300
        }
    ]
    
    wrapper = QuantumPyGADWrapper()
    
    # [SPARK OF LIFE]
    # Create initial population with seed
    initial_pop = np.random.randint(0, 4, size=(100, wrapper.genome_length))
    
    # Inject Spark (Radiotrophic GATA)
    spark_dna_str = "TGATAA" + "CGCT" * 23 + "A"
    full_str = (spark_dna_str[:100] * 10) + "".join(["A"]*1000)
    full_str = full_str[:wrapper.genome_length]
    mapping = {'A':0, 'T':1, 'C':2, 'G':3}
    encoded_spark = [mapping.get(c, 0) for c in full_str]
    initial_pop[0] = encoded_spark
    
    logger.info("Injected 'Spark of Life' (GATA Seed) into PyGAD Population.")
    
    current_pop = initial_pop
    
    for epoch in epochs:
        logger.info(f"\n--- Entering {epoch['name']} ---")
        logger.info(f"Environment: {epoch['env']}")
        
        # Update Global Context (ensure type compatibility)
        env_dict = epoch["env"]
        if isinstance(env_dict, dict):
            SIM_CONTEXT["env_conditions"] = env_dict
        else:
            logger.warning(f"Expected dict for env, got {type(env_dict)}")
        
        ga_instance = pygad.GA(
            num_generations=epoch["generations"],
            num_parents_mating=50,
            fitness_func=wrapper.fitness_func,
            initial_population=current_pop, # Carry over
            num_genes=wrapper.genome_length,
            gene_space=wrapper.gene_space,
            gene_type=int,
            mutation_percent_genes=2, # Higher mutation for faster adaptation to 6 epochs
            suppress_warnings=True
        )
        
        ga_instance.run()
        
        # Update population
        current_pop = ga_instance.population
        
        # Analysis
        solution, fitness, _ = ga_instance.best_solution()
        best_dna = wrapper.decode_solution(solution)
        
        # Analysis: Re-run system to get per-species population counts
        # We need to reconstruct the system exactly as in fitness_func
        species_params = []
        dna_full = wrapper.decode_solution(solution)
        
        for i in range(wrapper.num_species):
            start = i * wrapper.dna_size
            end = start + wrapper.dna_size
            dna_seq = dna_full[start:end]
            params = {
                'dna_sequence': dna_seq,
                'replication_rate': 0.1, 'decay_rate': 0.1, 'aging_rate': 0.01,
                'lifespan': 20, 'anti_life_effect': 1.0, 
                'birth_rate': 0.05,
                'prob_same_species': 0.01, 'prob_diff_species': 0.0,
                'interaction_strength': np.zeros(wrapper.num_species)
            }
            species_params.append(params)

        check_sys = QuantumBiologicalSystem(
            initial_states=np.full(wrapper.num_species, 50.0),
            energies=np.zeros(wrapper.num_species),
            carrying_capacity=10000, # Matched capacity
            species_params=species_params,
            mutation_rate=0.0,
            env_conditions=epoch["env"]
        )
        
        # Get final populations
        # Run for 50 generations to show true ecological equilibrium (Accurate Census)
        final_pops = check_sys.solve(generations=50)
        
        logger.info(f"Detailed Census (Best Ecosystem, Fitness={fitness:.1f}):")
        logger.info(f"ID | Pop   | Motifs | Partial DNA Sequence")
        logger.info(f"---|-------|--------|---------------------")
        
        for i, pop in enumerate(final_pops):
            # Only show if alive or significant
            if pop > 0.1:
                dna = species_params[i]['dna_sequence']
                # Identify Motifs
                motifs = []
                if "TGATAA" in dna: motifs.append("GATA")
                if "CCGG" in dna: motifs.append("HEME")
                if "AAAA" in dna: motifs.append("AQUA")
                if "CCCC" in dna: motifs.append("FEED")
                if "TGCG" in dna: motifs.append("DETOX")
                if "TATAAA" in dna: motifs.append("TATA")
                
                motif_str = " ".join(motifs) if motifs else "-"
                logger.info(f"{i:02d} | {pop:5.1f} | {motif_str:6s} | {dna[:20]}...")
        
        logger.info("-" * 50)
        
    logger.info("Evolutionary Timeline Complete.")
