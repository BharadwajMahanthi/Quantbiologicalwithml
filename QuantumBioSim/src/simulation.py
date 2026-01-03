import numpy as np
from deap import base, creator, tools, algorithms
import logging
from .models import QuantumBiologicalSystem

logger = logging.getLogger('quantum_bio_system')

# Configuration for GA
POP_SIZE = 50
NGEN = 50 # Increased generations to allow motifs to evolve
NUM_SPECIES = 10 
DNA_SIZE = 100
TOTAL_DNA_LENGTH = NUM_SPECIES * DNA_SIZE

def mutate_dna(individual, indpb):
    """Custom mutation for DNA character list."""
    choices = ['A', 'T', 'C', 'G']
    for i in range(len(individual)):
        if np.random.random() < indpb:
            individual[i] = np.random.choice(choices)
    return individual,

# Global DEAP Setup (Must only run once)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # type: ignore
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax) # type: ignore

def setup_ga_toolbox():
    toolbox = base.Toolbox()
    
    # Attribute generator: Random Nucleotide
    toolbox.register("attr_nucleotide", np.random.choice, list("ATCG"))
    
    # Structure initializers: Individual is a list of Nucleotides
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_nucleotide, n=TOTAL_DNA_LENGTH) # type: ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore

    # Genetic Operators
    toolbox.register("mate", tools.cxTwoPoint) # Standard crossover works on lists
    toolbox.register("mutate", mutate_dna, indpb=0.01) # Custom DNA mutation
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_ecosystem)
    
    return toolbox

def evaluate_ecosystem(individual):
    # Convert flat list of chars back to species DNA strings
    species_params = []
    
    # We construct species_params with ONLY DNA. 
    # The models.py GPM will calculate the float parameters (growth, decay) from this DNA.
    
    dna_string_full = "".join(individual)
    
    for i in range(NUM_SPECIES):
        start = i * DNA_SIZE
        end = start + DNA_SIZE
        dna_seq = dna_string_full[start:end]
        
        # Initialize basic params structure (values will be overwritten by GPM)
        params = {
            'dna_sequence': dna_seq,
            # Placeholder values, will be recalculated by update_phenotype_from_genotype
            'replication_rate': 0.1,
            'decay_rate': 0.1,
            'aging_rate': 0.01,
            'lifespan': 20,
            'anti_life_effect': 1.0,
            'radiation_effectiveness': 0.0,
            'prob_same_species': 0.01,
            'prob_diff_species': 0.0,
            'interaction_strength': np.zeros(NUM_SPECIES),
            'birth_rate': 0.05
        }
        species_params.append(params)

    # Initial states for simulation
    initial_states = np.full(NUM_SPECIES, 50.0)
    energies = np.ones(NUM_SPECIES)
    carrying_capacity = 1000
    mutation_rate = 0.0 # Handled by GA, not internal model mutation
    
    # Run system
    # The __init__ of QuantumBiologicalSystem now calls update_phenotype_from_genotype()
    # so the params will be set based on the DNA motifs immediately.
    system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params, mutation_rate)
    
    # Run for evaluation duration
    final_states = system.solve(generations=30)
    
    # Fitness = Total Biomass (Population)
    # This selects for species that can survive/thrive in the radiation (via GATA/Bdr1 motifs)
    fitness = np.sum(final_states)
    return fitness,

def run_evolution():
    toolbox = setup_ga_toolbox()
    population = toolbox.population(n=POP_SIZE) # type: ignore
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    logger.info("Starting Genotypic Evolution (Truth Codons)...")
    logger.info(f"Searching for Motifs: GATA (Radiotropism), Bdr1 (Repair), TATA (Growth)")
    
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)
    
    best_ind = hof[0]
    best_dna_full = "".join(best_ind)
    
    logger.info("Evolution Complete.")
    
    # Analyze Best Individual's Motifs
    def score_motif_print(seq, motif):
        count = seq.count(motif)
        if count > 0: return count
        sub1 = motif[:-1]
        sub2 = motif[1:]
        return (seq.count(sub1) + seq.count(sub2)) * 0.2

    # We need to count matching the scoring logic (Total Credit) or just raw count?
    # Let's show Raw Counts to prove "Truth" appearance.
    
    total_gata = best_dna_full.count("TGATAA")
    total_gata_score = score_motif_print(best_dna_full, "TGATAA")
    
    total_bdr1 = best_dna_full.count("TGACGTCA")
    total_bdr1_score = score_motif_print(best_dna_full, "TGACGTCA")
    
    total_tata = best_dna_full.count("TATAAA")
    total_tata_score = score_motif_print(best_dna_full, "TATAAA")
    
    logger.info(f"Best Ecosystem Phenotype Analysis:")
    logger.info(f"Total Radiotropism Motifs (GATA): {total_gata} (Score: {total_gata_score:.1f})")
    logger.info(f"Total Repair Motifs (Bdr1): {total_bdr1} (Score: {total_bdr1_score:.1f})")
    logger.info(f"Total Growth Motifs (TATA): {total_tata} (Score: {total_tata_score:.1f})")
    logger.info(f"Best Fitness (Biomass): {best_ind.fitness.values[0]}")
    
    return best_ind, log

def run_big_bang_simulation():
    """
    Simulates evolutionary timeline from Big Bang (High Rad) to Modern Day (Low Rad).
    Demonstrates emergence of human-like traits (Metabolism dominance) from fungal-like ancestors.
    """
    logger.info("Starting Big Bang Timeline Simulation...")
    
    # Timeline Configuration (Extended Generations for divergence)
    # Hadean: Extreme selection. Modern: Relaxation.
    epochs = [
        {"name": "Hadean Eon (4.0 Ga)", "radiation": 50.0, "generations": 50}, # 50x Radiation, 50 Gens
        {"name": "Archean Eon (2.5 Ga)", "radiation": 10.0, "generations": 50},
        {"name": "Modern Era (0.0 Ga)",  "radiation": 0.1, "generations": 50}
    ]
    
    toolbox = setup_ga_toolbox()
    
    # [NEW] Seed the Primordial Soup with Pre-Life Potentials
    # Instead of pure random noise, we introduce "chemical possibilities" (DNA fragments)
    # from our data definitions. This represents the diversity of early organic molecules.
    from .data_generation import generate_evolutionary_scenario
    proto_species = generate_evolutionary_scenario()
    
    seeded_pop = []
    # Fill population with copies of these proto-species DNA
    # We mix them so they compete immediately.
    while len(seeded_pop) < POP_SIZE:
        for p in proto_species:
            if len(seeded_pop) >= POP_SIZE: break
            # Convert string DNA to list[char] for DEAP
            dna_list = list(p['dna_sequence'])
            ind = creator.Individual(dna_list) # type: ignore
            seeded_pop.append(ind)
            
    # [CRITICAL FIX] "Spark of Life": Inject a survivor
    # Random DNA dies in Hadean (50x Rad). We need at least one "Lucky Mutation" (GATA) to start evolution.
    # Inserting a sequence with TGATAA (Radiotropism)
    spark_dna = list("TGATAA" + "CGCT" * 23 + "A")[:100] # Simple pattern with GATA
    seeded_pop[0] = creator.Individual(spark_dna) # type: ignore
            
    population = seeded_pop
    
    # We need a custom evaluate function that takes 'radiation' as an argument
    # But DEAP's evaluate is fixed argument. We'll use a closure or global update.
    # To keep it robust, we'll redefine the evaluation wrapper inside the loop or pass context.
    
    # Let's use a mutable context dictionary
    sim_context = {"env_radiation": 1.0}
    
    def evaluate_epoch(individual):
        # Convert DNA to params
        species_params = []
        dna_string_full = "".join(individual)
        for i in range(NUM_SPECIES):
            start = i * DNA_SIZE
            params = {
                'dna_sequence': dna_string_full[start : start + DNA_SIZE],
                # Placeholders, overwritten by GPM
                'replication_rate': 0.1, 'decay_rate': 0.1, 'aging_rate': 0.01,
                'lifespan': 20, 'anti_life_effect': 1.0, 
                'radiation_effectiveness': 0.0, # Vital: calculated by GPM
                'prob_same_species': 0.01, 'prob_diff_species': 0.0,
                'interaction_strength': np.zeros(NUM_SPECIES), 'birth_rate': 0.05
            }
            species_params.append(params)
            
        # Run System with EPOCH SPECIFIC Radiation
        system = QuantumBiologicalSystem(
            initial_states=np.full(NUM_SPECIES, 50.0),
            energies=np.ones(NUM_SPECIES),
            carrying_capacity=1000,
            species_params=species_params,
            mutation_rate=0.0,
            env_conditions={"radiation": sim_context["env_radiation"], "oxygen": 21.0, 
                            "co2": 0.04, "pollution": 0.0, "water": 1.0, "food": 1.0}
        )
        
        final_states = system.solve(generations=30)
        return np.sum(final_states),

    # Register the dynamic evaluator
    toolbox.register("evaluate", evaluate_epoch)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    log = None # Initialize log to avoid unbound error
    # Run Epochs sequentially, carrying over population
    for epoch in epochs:
        logger.info(f"\n--- Entering {epoch['name']} ---")
        logger.info(f"Environmental Radiation Level: {epoch['radiation']}x")
        
        # Update Context
        sim_context["env_radiation"] = float(epoch["radiation"]) # explicit float for type safety
        
        # Run Evolution for this Epoch
        population, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, 
                                              ngen=epoch['generations'], stats=stats, verbose=True)
        
        
        # [VERIFICATION] Re-Simulate Best Individual to checking SURVIVAL
        best_ind = tools.selBest(population, 1)[0]
        # best_val = evaluate_epoch(best_ind) # Re-run to get final state logic? 
        # Actually evaluate_epoch returns sum. We need specific species states.
        # So we replicate the setup logic here.
        
        dna_full = "".join(best_ind)
        # Parse Genotypes
        species_list = []
        for i in range(3): # Hardcoded NUM_SPECIES
            part = dna_full[i*100 : (i+1)*100]
            # Must include all parameters required by models.py
            params = {
                'dna_sequence': part, 
                'name': f'Species_{i}',
                'replication_rate': 0.1, 'decay_rate': 0.1, 'aging_rate': 0.01,
                'lifespan': 20, 'anti_life_effect': 1.0, 
                'radiation_effectiveness': 0.0,
                'prob_same_species': 0.01, 'prob_diff_species': 0.0,
                'interaction_strength': np.zeros(3), 'birth_rate': 0.05
            }
            species_list.append(params)
             
        # Re-Run Dynamics
        check_sys = QuantumBiologicalSystem(
            initial_states=np.full(3, 50.0),
            energies=np.ones(3),
            carrying_capacity=1000,
            species_params=species_list,
            mutation_rate=0.0,
            env_conditions={"radiation": float(epoch['radiation']), "oxygen": 21.0, 
                            "co2": 0.04, "pollution": 0.0, "water": 1.0, "food": 1.0}
        )
        final_pops = check_sys.solve(generations=50) # Match Epoch length
        
        # Weighted Analysis
        w_gata = 0.0
        w_tata = 0.0
        total_bio = np.sum(final_pops) + 1e-9
        
        for i, s_pop in enumerate(final_pops):
            dna = species_list[i]['dna_sequence']
            # Only count if species is effectively alive ( > 1.0 biomass)
            if s_pop > 1.0:
                gata = dna.count("TGATAA")
                tata = dna.count("TATAAA")
                w_gata += gata * s_pop
                w_tata += tata * s_pop
        
        # Normalize to "Average Motifs per Survivor"
        avg_gata = w_gata / total_bio
        avg_tata = w_tata / total_bio
        
        logger.info(f"Epoch Dominant Trait Analysis (Biomass Weighted):")
        logger.info(f"  Radiotropism (GATA - Fungi): {avg_gata:.2f} avg/ind")
        logger.info(f"  Metabolism (TATA - Human):   {avg_tata:.2f} avg/ind")
        
    logger.info("\nTimeline Complete.")
    return population, log
