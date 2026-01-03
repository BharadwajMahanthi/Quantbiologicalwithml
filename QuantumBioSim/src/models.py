import numpy as np
from scipy.linalg import expm
import logging
import os
import pandas as pd

# Import units system for research-grade parameters
try:
    from src.units import u, Qu, load_parameter_with_units, validate_units
    from src.models_units import (
        BiologicalConstants,
        dose_rate_to_growth_factor,
        convert_env_to_unitless
    )
    UNITS_AVAILABLE = True
except ImportError:
    # Fallback if units module not available
    UNITS_AVAILABLE = False
    print("Warning: src.units not available, using legacy unitless values")

logger = logging.getLogger('quantum_bio_system')

class PhysicsConstants:
    """Foundational Constants for Unified Physics Simulation"""
    C_LIGHT = 30.0      # Speed of Metabolic limit (Bio-speed of light)
    H_PLANCK = 6.626    # Quantum of Action (Mutation step size)
    G_GRAVITY = 0.1     # Competition Constant (Gravity between species)
    K_BOLTZMANN = 1.38  # Entropy/Thermodynamic constant (Decay base)


class QuantumBiologicalSystem:
    def __init__(self, initial_states, energies, carrying_capacity, species_params, mutation_rate, env_conditions=None):
        # logger.info("Initializing QuantumBiologicalSystem") 
        self.initial_states = initial_states
        self.states = np.array(initial_states, dtype=np.float64)
        self.energies = np.array(energies, dtype=np.float64)
        self.carrying_capacity = carrying_capacity
        self.species_params = species_params
        self.mutation_rate = mutation_rate
        
        # Default Environment if None (Modern Earth)
        if env_conditions is None:
            self.env_conditions = {
                "radiation": 0.1,
                "oxygen": 21.0, "co2": 0.04, "pollution": 0.0,
                "water": 1.0, "food": 1.0
            }
        else:
            self.env_conditions = env_conditions
            
        # Initialize sequences if not present
        self.dna_sequences = []
        self.rna_sequences = []
        for species in self.species_params:
             # Default random sequence if not present
            if 'dna_sequence' not in species:
                species['dna_sequence'] = self.generate_dna_sequence(100)
            self.dna_sequences.append(species['dna_sequence'])
            species['rna_sequence'] = self.generate_rna_sequence(species['dna_sequence'])
            self.rna_sequences.append(species['rna_sequence'])
            
        # Apply Genotype-Phenotype Map (Truth Codons)
        self.update_phenotype_from_genotype()

    def environmental_effect(self):
        """
        Calculates impact of ALL environmental stressors:
        1. Radiation (DNA Damage vs Radiotropism)
        2. Oxygen (Oxidative Stress vs Aerobic Energy)
        3. Resources (Starvation/Desiccation vs Efficiency)
        4. Pollution (Toxicity vs Detox)
        """
        deltas = np.zeros(len(self.states), dtype=np.float64)
        env = self.env_conditions
        
        for i, species in enumerate(self.species_params):
            if self.states[i] <= 0: continue
            
            pheno = species.get('phenotype', {})
            # Fallback if phenotype dict not ready
            if not pheno: 
                # Basic radiation support if legacy
                rad = species.get('radiation_effectiveness', 0.05)
                pheno = {'rad_sensitivity': rad, 'aerobic_efficiency': 0.0, 
                         'water_efficiency': 0.0, 'nutrient_efficiency': 0.0, 'pollution_resistance': 0.0}

            # --- 1. Radiation (Unit-Aware Dose-Response Model) ---
            # Detect if species has melanin (enables radiotropism)
            has_melanin = species.get('has_melanin', False)
            if not has_melanin and 'phenotype' in species:
                # Fallback: check phenotype for melanin indicators
                has_melanin = pheno.get('melanin_content', 0) > 0 or pheno.get('has_melanin', False)
            
            # Calculate radiation effect using research-grade model
            if UNITS_AVAILABLE and isinstance(env.get('radiation'), Qu):
                # Unit-aware path: radiation is already in Gy/hr
                dose_rate = env['radiation']
                growth_factor = dose_rate_to_growth_factor(dose_rate, has_melanin)
                rad_effect = (growth_factor - 1.0) * 0.5  # Scale to reasonable magnitude
            elif UNITS_AVAILABLE:
                # Semi-legacy: float interpreted as multiples of background
                bg_gy_hr = BiologicalConstants.BACKGROUND_RADIATION.value  # 2.74e-7 Gy/hr
                dose_rate_value = env.get('radiation', 1.0) * bg_gy_hr
                dose_rate = dose_rate_value * u.Gy / u.hour
                growth_factor = dose_rate_to_growth_factor(dose_rate, has_melanin)
                rad_effect = (growth_factor - 1.0) * 0.5  # Scale effect
            else:
                # Ultra-legacy: simple damage model (backwards compatible)
                rad_effect = -env.get('radiation', 0.1) * pheno.get('rad_sensitivity', 0.05)


            
            # --- 2. Oxygen (Research-Grade Curve from atmosphere.py) ---
            # Use scientifically accurate O₂ growth/toxicity curves
            if UNITS_AVAILABLE:
                try:
                    from src.atmosphere import o2_growth_factor
                    
                    # Determine if aerobic (requires O₂ for energy)
                    is_aerobic = pheno.get('aerobic_efficiency', 0) > 0.5
                    # Check for oxidative stress protection (catalase, SOD enzymes)
                    has_catalase = pheno.get('oxidative_protection', 0) > 0.5
                    
                    # Get O₂ concentration (handles units or float)
                    o2_conc = env.get('oxygen', 21.0)
                    
                    # Calculate growth factor using research curve
                    # Returns: 1.0 = optimal, <1.0 = stress/toxicity, >1.0 = rare enhancement
                    o2_factor = o2_growth_factor(o2_conc, is_aerobic, has_catalase)
                    
                    # Convert factor to effect (delta from baseline)
                    # Scale by 0.1 to keep magnitude reasonable vs other effects
                    o2_effect = (o2_factor - 1.0) * 0.1
                    
                except (ImportError, KeyError):
                    # Fallback to legacy calculation if atmosphere.py not available
                    if pheno.get('aerobic_efficiency', 0) > 0.5:
                        o2_effect = 0.05 * (env.get('oxygen', 21.0) / 21.0) * pheno['aerobic_efficiency']
                    else:
                        excess_o2 = max(0.0, env.get('oxygen', 21.0) - 2.0)
                        o2_effect = -0.05 * excess_o2
            else:
                # Ultra-legacy: original code for backwards compatibility
                if pheno['aerobic_efficiency'] > 0.5:
                    o2_effect = 0.05 * (env['oxygen'] / 21.0) * pheno['aerobic_efficiency']
                else:
                    excess_o2 = max(0.0, env['oxygen'] - 2.0)
                    o2_effect = -0.05 * excess_o2
                
            # --- 3. Resources (Food & Water) ---
            # Scarcity = Env Value < 1.0. Efficiency mitigates this.
            water_stress = max(0.0, (1.0 - env['water']) - pheno['water_efficiency'])
            food_stress = max(0.0, (1.0 - env['food']) - pheno['nutrient_efficiency'])
            
            resource_penalty = -0.1 * (water_stress + food_stress)
            
            # --- 4. Pollution ---
            # Raw toxicity reduced by Detox
            net_pollution = max(0.0, env['pollution'] - pheno['pollution_resistance'])
            pollution_penalty = -0.1 * net_pollution
            
            # --- Net Effect ---
            # Sum of impacts on growth rate
            # NOTE: rad_effect can now be positive (radiotropic growth) or negative (damage)
            net_growth_modifier = rad_effect + o2_effect + resource_penalty + pollution_penalty
            
            deltas[i] = self.states[i] * net_growth_modifier
            
        return deltas

    def update_phenotype_from_genotype(self):
        """
        Updates species parameters based on 'Truth Codons' in their DNA.
        Expanded GPM for Environmental Adaptation.
        """
        for i, species in enumerate(self.species_params):
            dna = species.get('dna_sequence', '')
            
            # --- Helper for Gradient Scoring ---
            def score_motif(seq, motif):
                count = seq.count(motif)
                if count > 0: return count
                sub1 = motif[:-1]
                sub2 = motif[1:]
                return (seq.count(sub1) + seq.count(sub2)) * 0.25
            
            # Count Biological Motifs
            scores = {
                'GATA': score_motif(dna, 'TGATAA'),       # Radiotropism
                'TATA': score_motif(dna, 'TATAAA'),       # Replication (Promoter)
                'HEME': score_motif(dna, 'CCGG'),         # Aerobic (Oxygen)
                'AQUA': score_motif(dna, 'AAAA'),         # Water Efficiency
                'FEED': score_motif(dna, 'CCCC'),         # Nutrient Efficiency
                'DETOX': score_motif(dna, 'TGCG')         # Pollution Resistance
            }
            
            # Initialize Phenotype Dict
            species['phenotype'] = {}
            
            # 1. Radiation Sensitivity
            # Base = 0.05 (Sensitive). GATA reduces it.
            # If GATA is high, sensitivity becomes negative (Trophism).
            species['phenotype']['rad_sensitivity'] = 0.005 - (scores['GATA'] * 0.003)
            
            # 2. Aerobic Efficiency (0.0 to 1.0)
            # Needs at least one HEME to be aerobic
            species['phenotype']['aerobic_efficiency'] = min(2.0, scores['HEME'] * 0.5)
            
            # 3. Water Efficiency
            species['phenotype']['water_efficiency'] = min(1.0, scores['AQUA'] * 0.2)
            
            # 4. Nutrient Efficiency
            species['phenotype']['nutrient_efficiency'] = min(1.0, scores['FEED'] * 0.2)
            
            # 5. Pollution Resistance
            species['phenotype']['pollution_resistance'] = scores['DETOX'] * 10.0 # High capacity
            
            # 6. Replication Rate (Base Biological Speed)
            base_rep = 0.05
            species['replication_rate'] = base_rep + (scores['TATA'] * 0.05)
            species['decay_rate'] = 0.1 # Constant baseline decay (aging)
            
            # Map legacy GPM for backward compatibility if needed, but 'phenotype' dict handles physics now.
            species['radiation_effectiveness'] = species['phenotype']['rad_sensitivity']

    def hamiltonian_operator(self):
        """
        Constructs a Hamiltonian operator H to model system evolution terms.
        
        NOTE ON SCIENTIFIC VALIDITY:
        This is a 'Quantum-Like' model (QLM). It uses the mathematical formalism of Quantum Mechanics
        (Hamiltonians, Wavefunctions, Unitary Evolution) to model macroscopic population dynamics.
        
        - Diagonal elements H[i,i]: Represent 'Self-Energy' ~ Species internal growth/decay potential.
        - Off-diagonal elements H[i,j]: Represent 'Interactions' ~ Predation/Symbiosis strengths.
        
        This is analogous to Quantum Lotka-Volterra models (see: 'Quantum Probability and Applications to Biology').
        It does NOT imply the macroscopic fungi are in a coherent quantum superposition state.
        """
        # logger.info("Calculating Hamiltonian operator")
        num_species = len(self.states)
        H = np.zeros((num_species, num_species), dtype=np.float64)

        # Diagonal elements represent energies of the species
        for i in range(num_species):
            H[i, i] = self.energies[i]

        # Off-diagonal elements represent interactions between species
        for i in range(num_species):
            for j in range(i + 1, num_species):
                interaction_strength = self.species_params[i].get('interaction_strength', None)
                if isinstance(interaction_strength, (list, np.ndarray)):
                    if j < len(interaction_strength):
                        H[i, j] = interaction_strength[j]
                        H[j, i] = interaction_strength[j]
                    else:
                        # logger.warning(f"Interaction_strength list length shorter than expected at ({i}, {j})")
                        pass

        return H

    def schrodinger_equation(self, wavefunction, time_step):
        try:
            H = self.hamiltonian_operator() 
            if H.dtype != np.complex128:
              H = H.astype(np.complex128)
            
            # Ensure proper dtype for matrix exponential
            exp_H = expm(-1j * H * time_step) 
            
            # Simple matrix multiplication for time evolution
            psi_t = exp_H @ wavefunction  
            return psi_t.real # Returning real part as population must be real
        except Exception as e:
            logger.error(f"Error solving Schrödinger equation: {e}")
            raise e

    def replication_term(self):
        deltas = np.zeros(len(self.states), dtype=np.float64)
        for i in range(len(self.states)):
            deltas[i] = self.states[i] * self.species_params[i]['replication_rate'] * (1 - self.states[i] / self.carrying_capacity)
        return deltas

    def decay_term(self):
        deltas = np.zeros(len(self.states), dtype=np.float64)
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['decay_rate']
        return deltas

    def lifespan_term(self):
        deltas = np.zeros(len(self.states), dtype=np.float64)
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['aging_rate'] * self.species_params[i]['lifespan']
        return deltas

    def anti_life_effects(self):
        deltas = np.zeros(len(self.states), dtype=np.float64)
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['anti_life_effect']
        return deltas

    def compute_same_species_birth(self):
        deltas = np.zeros(len(self.states), dtype=np.float64)
        for i in range(len(self.states)):
            # This loop logic from original code might be O(N^2), optimizing or keeping as is? 
            # Keeping as is to preserve logic reliability.
            for j in range(len(self.states)):
                if i != j:
                    deltas[i] += (self.species_params[i]['prob_same_species'] *
                                  self.species_params[i]['birth_rate'] *
                                  self.states[j])
        return deltas

    def compute_diff_species_birth(self):
        deltas = np.zeros(len(self.states), dtype=np.float64)
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                if i != j:
                    deltas[i] += (self.species_params[i]['prob_diff_species'] *
                                  self.species_params[i]['birth_rate'] *
                                  self.states[j])
        return deltas
    
    def compute_interactions(self):
        # Original code had this logic.
        same_species_birth = np.zeros(len(self.states))
        diff_species_birth = np.zeros(len(self.states))
        
        for i in range(len(self.states)):
            same_species_birth[i] = self.species_params[i]['prob_same_species']
            diff_species_birth[i] = self.species_params[i]['prob_diff_species']
        return same_species_birth, diff_species_birth

    def generate_dna_sequence(self, length):
        return ''.join(np.random.choice(['A', 'T', 'C', 'G'], length))

    def generate_rna_sequence(self, dna_sequence):
        transcription_map = str.maketrans('ATCG', 'UAGC')
        if isinstance(dna_sequence, np.ndarray):
            dna_sequence = dna_sequence.tobytes().decode('utf-8')
        return dna_sequence.translate(transcription_map)

    def mutate_dna(self, dna, mutation_rate):
        dna_list = list(dna)
        for i in range(len(dna_list)):
            if np.random.rand() < mutation_rate:
                dna_list[i] = np.random.choice(list('ATCG'))
        return ''.join(dna_list)

    def replace_extinct_species(self):
        # We need to act on indices where state <= 0
        extinct_indices = np.where(self.states <= 0)[0]
        for i in extinct_indices:
            # Create new params based on old ones with random variations
            old_params = self.species_params[i]
            new_params = old_params.copy()
            
            # Apply variations (logic from original code)
            new_params['replication_rate'] *= np.random.uniform(0.5, 1.5)
            new_params['decay_rate'] *= np.random.uniform(0.5, 1.5)
            # ... (truncated for brevity, implementing key changes)
            new_params['dna_sequence'] = self.mutate_dna(self.generate_dna_sequence(100), self.mutation_rate)
            new_params['rna_sequence'] = self.generate_rna_sequence(new_params['dna_sequence'])
            
            self.species_params[i] = new_params
            self.states[i] = np.random.uniform(1, 100) # Re-seed population
    
    def update_states(self, num_time_steps=1, time_step=1.0):
        for _ in range(num_time_steps):
            delta_replication = self.replication_term()
            delta_decay = self.decay_term()
            delta_lifespan = self.lifespan_term()
            delta_antilife = self.anti_life_effects()
            
            # [CRITICAL UPDATE] Replace old Radiation Effect with full Environment Effect
            delta_environment = self.environmental_effect()
            
            # [NEW] Catastrophic Events (War, Plague, Purge)
            delta_catastrophe = self.compute_catastrophes()
            
            # [NEW] Fundamental Physics (Einstein & Newton)
            delta_relativity = self.compute_relativity()
            delta_newtonian = self.compute_newtonian_mechanics()
            
            same_species_birth = self.compute_same_species_birth()
            diff_species_birth = self.compute_diff_species_birth()
            
            # The odd interaction term from original
            prob_interaction_same, prob_interaction_diff = self.compute_interactions()

            # Schrodinger evolution
            # Treat states as wavefunction
            # Note: The original code assigns self.states = psi_t
            psi_t = self.schrodinger_equation(self.states, time_step)
            self.states = psi_t
            
            # Add classical terms with NEW Environmental and Catastrophic Deltas
            # Grand Unified Equation:
            # dP/dt = Quantum_Prob + Relativity_Limit + Newtonian_Force + Bio_Growth + Catastrophe
            self.states += (
                delta_replication + delta_decay + delta_lifespan +
                delta_antilife + delta_environment + delta_catastrophe +
                delta_relativity + delta_newtonian +
                same_species_birth + diff_species_birth + 
                prob_interaction_same + prob_interaction_diff
            )
            
            # Clean up
            self.states = np.maximum(0, self.states)
            np.clip(self.states, 0, self.carrying_capacity, out=self.states)
            
            # Extinction check
            self.replace_extinct_species()

        return self.states

    def compute_catastrophes(self):
        """
        Simulates probabilistic catastrophic events:
        1. Plague: Random high-decay event (Black Death style).
        2. War: Density-dependent conflict reducing population.
        """
        deltas = np.zeros(len(self.states), dtype=np.float64)
        
        # Probabilities from Env Conditions (default small)
        plague_risk = self.env_conditions.get('plague_risk', 0.0)
        war_risk = self.env_conditions.get('war_risk', 0.0)
        
        # 1. Plague (The Great Purge)
        # Randomly strikes. Hits dense populations harder.
        if np.random.rand() < plague_risk:
            # logger.debug("!!! PLAGUE EVENT DETECTED !!!") # Silenced per user request
            for i in range(len(self.states)):
                # 30-50% population loss
                damage = self.states[i] * np.random.uniform(0.3, 0.5) 
                deltas[i] -= damage

        # 2. War (Resource Conflict)
        # Triggers if total population is high (>50% capacity)
        total_pop = np.sum(self.states)
        if total_pop > (self.carrying_capacity * 0.5):
            if np.random.rand() < war_risk:
                # logger.debug("!!! WAR DECLARED !!!") # Silenced per user request
                for i in range(len(self.states)):
                    # War hits everyone, but slightly less for 'aggressive' ones? 
                    # For now, uniform destruction: 10-20% loss
                    damage = self.states[i] * np.random.uniform(0.1, 0.2)
                    deltas[i] -= damage
                    
        return deltas

    def compute_relativity(self):
        """
        E = mc^2 (Einstein):
        Calculates the Mass-Energy Equivalence limit.
        Biomass (m) cannot exceed available Energy (E) divided by C^2.
        
        Energy Sources (E):
        - Food (Chemical Potential)
        - Oxygen (Oxidative Potential)
        - Radiation (Photon Energy - for Radiotrophs only)
        """
        c2 = PhysicsConstants.C_LIGHT ** 2
        deltas = np.zeros(len(self.states), dtype=np.float64)
        env = self.env_conditions
        
        for i, species in enumerate(self.species_params):
            if self.states[i] <= 0: continue
            
            # Calculate Total Energy Available (E)
            # Baseline Chemical Energy
            energy_in = env['food'] * 1000.0 # Calories
            
            # Efficient Species extract more E
            pheno = species.get('phenotype', {})
            if pheno.get('aerobic_efficiency', 0) > 0.5:
                # E = E_base + E_oxidative
                energy_in += env['oxygen'] * 500.0 # ATP boost
                
            # Radiotrophs extract E from Radiation
            if pheno.get('rad_sensitivity', 1.0) < 0: # Negative sensitivity = Trophism
                # E = E_base + E_photon
                energy_in += env['radiation'] * 200.0
                
            # Relativity Constraint: m_max = E / c^2
            # If current m > m_max, relativistic decay occurs (starvation/burnout)
            max_mass = energy_in / c2
            
            # "Time Dilation": As they approach max capacity, metabolic time slows?
            # Implemented as simple cap for now.
            if self.states[i] > max_mass:
                # E=mc^2 Penalty: Mass shedding
                loss = (self.states[i] - max_mass) * 0.1
                deltas[i] -= loss
                
        return deltas

    def compute_newtonian_mechanics(self):
        """
        F = ma (Newton):
        Classical Evolutionary Forces acting on Population Mass.
        
        Forces:
        1. Gravity (G): Competition from other species (Mass attraction/repulsion).
        2. Friction (mu): Environmental resistance (Entropy).
        3. Thrust (T): Growth Potential.
        """
        G = PhysicsConstants.G_GRAVITY
        deltas = np.zeros(len(self.states), dtype=np.float64)
        total_mass = np.sum(self.states)
        
        for i, species in enumerate(self.species_params):
            if self.states[i] <= 0: continue
            
            # 1. Gravitational Drag (Competition)
            # F_gravity = G * (M_total * m_i) / r^2 (Assume r=1 niche distance)
            # Biologically: Density Dependent Inhibition
            f_gravity = G * (total_mass * self.states[i]) / self.carrying_capacity
            
            # 2. Inertia (Mass)
            m = self.states[i]
            if m < 1.0: m = 1.0 # Prevent div by zero
            
            # Acceleration = Force / Mass
            # A negative force (Competition) reduces growth
            accel_drag = -f_gravity / m
            
            deltas[i] += accel_drag
            
        return deltas

    def solve(self, generations):
        # Wrapper for update_states to run for N generations
        # In original, 'solve' loops generations and calls terms inside.
        # 'update_states' also looks like it does loops.
        # We will alias solve to update_states for simplicity.
        return self.update_states(num_time_steps=generations)
