# Extracted content from life_equ.ipynb

## Cell 1 (Code)

```python
import math
import random

def combined_equation(Q_0, r, decay_constant, delta, t, lifespan, anti_life_effect, prob_same_species, prob_diff_species):
    growth_term = Q_0 * math.exp((r - decay_constant) * t)
    aging_term = delta * t * lifespan
    anti_life_term = anti_life_effect
    same_species_birth = prob_same_species
    diff_species_birth = prob_diff_species
    
    return growth_term - aging_term - anti_life_term + same_species_birth + diff_species_birth

# Example usage:
Q_0 = 100  # Initial quantity
r = 0.2  # Intrinsic growth rate
decay_constant = 0.1  # Decay constant
delta = 0.05  # Rate of aging
t = 10  # Time
lifespan = 5  # Lifespan
anti_life_effect = 2  # Anti-life effect
prob_same_species = 3  # Probability of same species being born
prob_diff_species = 1  # Probability of different species being born

result = combined_equation(Q_0, r, decay_constant, delta, t, lifespan, anti_life_effect, prob_same_species, prob_diff_species)
print("Result:", result)

```

## Cell 2 (Code)

```python
import random

class BiologicalSystem:
    def __init__(self, species_A_count, species_B_count):
        self.species_A_count = species_A_count
        self.species_B_count = species_B_count

    def decay(self, rate):
        decay_A = random.random() * rate
        decay_B = random.random() * rate
        self.species_A_count -= decay_A
        self.species_B_count -= decay_B

    def replicate(self, rate):
        replicate_A = random.random() * rate
        replicate_B = random.random() * rate
        self.species_A_count += replicate_A
        self.species_B_count += replicate_B

    def lifespan(self, max_count):
        if self.species_A_count > max_count:
            self.species_A_count = max_count
        if self.species_B_count > max_count:
            self.species_B_count = max_count

    def interaction(self, probability_same_species):
        for _ in range(int(self.species_A_count)):
            if random.random() < probability_same_species:
                self.species_A_count += 1
            else:
                self.species_B_count += 1

    def run_simulation(self, steps, decay_rate, replication_rate, max_count, probability_same_species):
        for _ in range(steps):
            self.decay(decay_rate)
            self.replicate(replication_rate)
            self.lifespan(max_count)
            self.interaction(probability_same_species)
            print(f"Species A count: {self.species_A_count}, Species B count: {self.species_B_count}")

# Example usage:
initial_species_A_count = 100
initial_species_B_count = 100
steps = 10
decay_rate = 0.1
replication_rate = 0.2
max_count = 1000
probability_same_species = 0.5

system = BiologicalSystem(initial_species_A_count, initial_species_B_count)
system.run_simulation(steps, decay_rate, replication_rate, max_count, probability_same_species)

```

## Cell 3 (Code)

```python
class QuantumBiologicalSystem:
    def __init__(self, species_A_initial_state, species_B_initial_state, replication_rate):
        self.species_A_state = species_A_initial_state
        self.species_B_state = species_B_initial_state
        self.replication_rate = replication_rate

    def replication_term(self):
        # Calculate the change in population size due to replication
        replication_change_A = self.replication_rate * self.species_A_state
        replication_change_B = self.replication_rate * self.species_B_state

        # Return the replication term
        return replication_change_A, replication_change_B

    # Other methods...

# Example usage:
species_A_initial_state = 100  # Initial population size for species A
species_B_initial_state = 150  # Initial population size for species B
replication_rate = 0.1  # Replication rate (e.g., 10% per unit time)

quantum_system = QuantumBiologicalSystem(species_A_initial_state, species_B_initial_state, replication_rate)
replication_change_A, replication_change_B = quantum_system.replication_term()
print("Change in population size due to replication for species A:", replication_change_A)
print("Change in population size due to replication for species B:", replication_change_B)

```

## Cell 4 (Code)

```python
import numpy as np

class QuantumBiologicalSystem:
    def __init__(self, species_A_initial_state, species_B_initial_state):
        self.species_A_state = species_A_initial_state
        self.species_B_state = species_B_initial_state

    def replication_term(self, replication_rate):
        # Calculate change in population size due to replication
        delta_A = replication_rate * self.species_A_state
        delta_B = replication_rate * self.species_B_state
        return delta_A, delta_B

    def decay_term(self, decay_rate):
        # Calculate change in population size due to decay
        delta_A = -decay_rate * self.species_A_state
        delta_B = -decay_rate * self.species_B_state
        return delta_A, delta_B

    def lifespan_term(self, lifespan_reduction_rate):
        # Calculate change in population size due to lifespan reduction
        delta_A = -lifespan_reduction_rate * self.species_A_state
        delta_B = -lifespan_reduction_rate * self.species_B_state
        return delta_A, delta_B

    def anti_life_effects(self, inhibition_rate):
        # Calculate change in population size due to anti-life effects
        delta_A = -inhibition_rate * self.species_A_state
        delta_B = -inhibition_rate * self.species_B_state
        return delta_A, delta_B

    def probability_of_interactions(self, same_species_prob, diff_species_prob):
        # Define probabilities of species interactions
        P_same = same_species_prob
        P_diff = diff_species_prob
        return P_same, P_diff

    def solve(self, replication_rate, decay_rate, lifespan_reduction_rate, inhibition_rate,
              same_species_prob, diff_species_prob, time_steps):
        # Solve the equations using provided parameters
        for t in range(time_steps):
            # Calculate changes in population size
            delta_rep_A, delta_rep_B = self.replication_term(replication_rate)
            delta_decay_A, delta_decay_B = self.decay_term(decay_rate)
            delta_lifespan_A, delta_lifespan_B = self.lifespan_term(lifespan_reduction_rate)
            delta_anti_A, delta_anti_B = self.anti_life_effects(inhibition_rate)
            
            # Incorporate probability of interactions
            P_same, P_diff = self.probability_of_interactions(same_species_prob, diff_species_prob)
            
            # Update population sizes
            self.species_A_state += delta_rep_A + delta_decay_A + delta_lifespan_A + delta_anti_A
            self.species_B_state += delta_rep_B + delta_decay_B + delta_lifespan_B + delta_anti_B
            
            # Apply interactions
            # For simplicity, let's assume interactions do not change population size

# Example usage:
species_A_initial_state = 100
species_B_initial_state = 50

quantum_system = QuantumBiologicalSystem(species_A_initial_state, species_B_initial_state)
quantum_system.solve(replication_rate=0.1, decay_rate=0.05, lifespan_reduction_rate=0.02, inhibition_rate=0.03,
                     same_species_prob=0.2, diff_species_prob=0.1, time_steps=100)
print("Species A count:", quantum_system.species_A_state)
print("Species B count:", quantum_system.species_B_state)

```

## Cell 5 (Code)

```python
import numpy as np

class QuantumBiologicalSystem:
    def __init__(self, species_A_initial_state, species_B_initial_state):
        self.species_A_state = species_A_initial_state
        self.species_B_state = species_B_initial_state

    def replication_term(self, replication_rate):
        # Change in population size due to replication
        change_A = replication_rate * self.species_A_state
        change_B = replication_rate * self.species_B_state
        return change_A, change_B

    def decay_term(self, decay_rate):
        # Change in population size due to decay
        change_A = -decay_rate * self.species_A_state
        change_B = -decay_rate * self.species_B_state
        return change_A, change_B

    def radiation_term(self, radiation_effect):
        # Change in population size due to radiation effect
        change_A = radiation_effect * self.species_A_state
        change_B = radiation_effect * self.species_B_state
        return change_A, change_B

    def solve(self, replication_rate, decay_rate, radiation_effect, time_steps=100):
        for _ in range(time_steps):
            # Calculate changes due to replication
            change_A_rep, change_B_rep = self.replication_term(replication_rate)
            # Calculate changes due to decay
            change_A_decay, change_B_decay = self.decay_term(decay_rate)
            # Apply radiation effect
            change_A_rad, change_B_rad = self.radiation_term(radiation_effect)
            # Update population counts
            self.species_A_state += change_A_rep + change_A_decay + change_A_rad
            self.species_B_state += change_B_rep + change_B_decay + change_B_rad
            # Ensure population counts are non-negative
            self.species_A_state = max(0, self.species_A_state)
            self.species_B_state = max(0, self.species_B_state)
        return self.species_A_state, self.species_B_state

def combined_equation(Q_0, r, decay_constant, delta, t, lifespan, anti_life_effect, prob_same_species, prob_diff_species):
    growth_term = Q_0 * np.exp((r - decay_constant) * t)
    aging_term = delta * t * lifespan
    anti_life_term = anti_life_effect
    same_species_birth = prob_same_species
    diff_species_birth = prob_diff_species
    
    return growth_term - aging_term - anti_life_term + same_species_birth + diff_species_birth

# Example usage:
species_A_initial_state = 100
species_B_initial_state = 50
quantum_system = QuantumBiologicalSystem(species_A_initial_state, species_B_initial_state)

# Set parameters
replication_rate = 0.1
decay_rate = 0.05
radiation_effect = -0.02  # Negative effect due to radiation
time_steps = 100

# Solve the system
final_A_count, final_B_count = quantum_system.solve(replication_rate, decay_rate, radiation_effect, time_steps)
print("Final Species A count:", final_A_count)
print("Final Species B count:", final_B_count)

# Combined equation example usage
Q_0 = 100  # Initial quantity
r = 0.2  # Intrinsic growth rate
decay_constant = 0.1  # Decay constant
delta = 0.05  # Rate of aging
t = 10  # Time
lifespan = 5  # Lifespan
anti_life_effect = 2  # Anti-life effect
prob_same_species = 3  # Probability of same species being born
prob_diff_species = 1  # Probability of different species being born

result = combined_equation(Q_0, r, decay_constant, delta, t, lifespan, anti_life_effect, prob_same_species, prob_diff_species)
print("Result from combined equation:", result)

```

## Cell 6 (Code)

```python
import numpy as np
import math

class QuantumBiologicalSystem:
    def __init__(self, initial_states, energies, carrying_capacity):
        self.states = np.array(initial_states)
        self.energies = np.array(energies)
        self.carrying_capacity = carrying_capacity

    def hamiltonian_operator(self):
        num_species = len(self.states)
        H = np.zeros((num_species, num_species))  # Initialize a matrix for the Hamiltonian operator

        # Diagonal elements represent the energy of each species
        for i in range(num_species):
            H[i, i] = self.states[i] * self.energies[i]

        # Off-diagonal elements represent interactions between species
        # For simplicity, let's assume no interactions for now
        return H

    def replication_term(self, replication_rates):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = self.states[i] * replication_rates[i] * (1 - self.states[i] / self.carrying_capacity)
        return deltas

    def decay_term(self, decay_rates):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * decay_rates[i]
        return deltas

    def lifespan_term(self, delta, lifespan):
        deltas = np.full(len(self.states), -delta * lifespan)
        return deltas

    def anti_life_effects(self, anti_life_effects):
        return np.array([-effect for effect in anti_life_effects])

    def radiation_effect(self, radiation_effectiveness):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * radiation_effectiveness[i]
        return deltas

    def probability_of_interactions(self, prob_same_species, prob_diff_species):
        return np.array(prob_same_species), np.array(prob_diff_species)

    def schrodinger_equation(self, wavefunction, time):
        H = self.hamiltonian_operator()
        psi_t = np.dot(np.exp(-1j * H * time), wavefunction)
        return psi_t

    def solve(self, replication_rates, decay_rates, delta, lifespan, anti_life_effects, radiation_effectiveness, prob_same_species, prob_diff_species, generations):
        for _ in range(generations):
            delta_replication = self.replication_term(replication_rates)
            delta_decay = self.decay_term(decay_rates)
            delta_lifespan = self.lifespan_term(delta, lifespan)
            delta_antilife = self.anti_life_effects(anti_life_effects)
            delta_radiation = self.radiation_effect(radiation_effectiveness)
            same_species_birth, diff_species_birth = self.probability_of_interactions(prob_same_species, prob_diff_species)

            self.states += (delta_replication + delta_decay + delta_lifespan + delta_antilife + delta_radiation + same_species_birth + diff_species_birth)

            # Ensure populations do not become negative
            self.states = np.maximum(0, self.states)

        return self.states

# Example initial state vector for 20 species
initial_states = [100.0] * 20  # Initial population for each species
energies = [1.0] * 20  # Energy for each species
carrying_capacity = 1000.0  # Carrying capacity for the environment

# Parameters for mushroom and fungal species
replication_rates = [0.1] * 20  # Replication rates for each species
decay_rates = [0.05] * 20  # Decay rates for each species
delta = 0.02  # Rate of aging for mushrooms/fungi
lifespan = 20  # Lifespan for mushrooms/fungi in arbitrary units
anti_life_effects = [1] * 20  # Anti-life effects for each species
radiation_effectiveness = [0.01] * 20  # Effect of radiation on each species
prob_same_species = [0.02] * 20  # Probability of same species interaction
prob_diff_species = [0.01] * 20  # Probability of different species interaction
generations = 1000000  # Number of generations to simulate

# Create QuantumBiologicalSystem instance
quantum_system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity)

# Solve the system for 1 million generations
final_states = quantum_system.solve(replication_rates, decay_rates, delta, lifespan, anti_life_effects, radiation_effectiveness, prob_same_species, prob_diff_species, generations)
print("Final species counts after 1 million generations:", final_states)

```

## Cell 7 (Code)

```python
import numpy as np
import math

class QuantumBiologicalSystem:
    def __init__(self, initial_states, energies, carrying_capacity, species_params):
        self.states = np.array(initial_states)
        self.energies = np.array(energies)
        self.carrying_capacity = carrying_capacity
        self.species_params = species_params

    def hamiltonian_operator(self):
        num_species = len(self.states)
        H = np.zeros((num_species, num_species))  # Initialize a matrix for the Hamiltonian operator

        # Diagonal elements represent the energy of each species
        for i in range(num_species):
            H[i, i] = self.states[i] * self.energies[i]

        # Off-diagonal elements represent interactions between species
        for i in range(num_species):
            for j in range(i+1, num_species):
                interaction_strength = self.species_params[i]['interaction_strength'][j]
                H[i, j] = interaction_strength
                H[j, i] = interaction_strength  # Symmetric interaction matrix

        return H

    def replication_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = self.states[i] * self.species_params[i]['replication_rate'] * (1 - self.states[i] / self.carrying_capacity)
        return deltas

    def decay_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['decay_rate']
        return deltas

    def lifespan_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['aging_rate'] * self.species_params[i]['lifespan']
        return deltas

    def anti_life_effects(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['anti_life_effect']
        return deltas

    def radiation_effect(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['radiation_effectiveness']
        return deltas

    def probability_of_interactions(self):
        same_species_birth = np.zeros(len(self.states))
        diff_species_birth = np.zeros(len(self.states))
        for i in range(len(self.states)):
            same_species_birth[i] = self.species_params[i]['prob_same_species']
            diff_species_birth[i] = self.species_params[i]['prob_diff_species']
        return same_species_birth, diff_species_birth

    def schrodinger_equation(self, wavefunction, time):
        H = self.hamiltonian_operator()
        psi_t = np.dot(np.exp(-1j * H * time), wavefunction)
        return psi_t

    def solve(self, generations):
        for _ in range(generations):
            delta_replication = self.replication_term()
            delta_decay = self.decay_term()
            delta_lifespan = self.lifespan_term()
            delta_antilife = self.anti_life_effects()
            delta_radiation = self.radiation_effect()
            same_species_birth, diff_species_birth = self.probability_of_interactions()

            self.states += (delta_replication + delta_decay + delta_lifespan + delta_antilife + delta_radiation + same_species_birth + diff_species_birth)

            # Ensure populations do not become negative
            self.states = np.maximum(0, self.states)

        return self.states

# Example initial state vector for 20 species
initial_states = np.random.uniform(50, 150, 20)  # Initial population for each species
energies = np.random.uniform(0.5, 1.5, 20)  # Energy for each species
carrying_capacity = 1000.0  # Carrying capacity for the environment

# Parameters for 20 different species with variability and interaction strengths
species_params = [
    {
        'replication_rate': np.random.uniform(0.05, 0.15),
        'decay_rate': np.random.uniform(0.01, 0.1),
        'aging_rate': np.random.uniform(0.01, 0.03),
        'lifespan': np.random.uniform(10, 30),
        'anti_life_effect': np.random.uniform(0.5, 2),
        'radiation_effectiveness': np.random.uniform(0.005, 0.02),
        'prob_same_species': np.random.uniform(0.01, 0.03),
        'prob_diff_species': np.random.uniform(0.005, 0.02),
        'interaction_strength': np.random.uniform(-0.01, 0.01, 20)  # Interaction strength with other species
    } for _ in range(20)
]

# Create QuantumBiologicalSystem instance
quantum_system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params)

# Solve the system for 1 million generations
generations = 1000000
final_states = quantum_system.solve(generations)
print("Final species counts after 1 million generations:", final_states)
```

## Cell 8 (Code)

```python
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

class QuantumBiologicalSystemWithML(QuantumBiologicalSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_models = [AutoReg(self.states[:, i], lags=1).fit() for i in range(len(self.states[0]))]

    def predict_next_generation(self):
        predicted_states = np.zeros_like(self.states)
        for i, model in enumerate(self.predictor_models):
            predicted_states[:, i] = model.predict(start=len(self.states), end=len(self.states)+1)
        return predicted_states


class QuantumBiologicalSystem:
    def __init__(self, initial_states, energies, carrying_capacity, species_params):
        self.states = np.array(initial_states)
        self.energies = np.array(energies)
        self.carrying_capacity = carrying_capacity
        self.species_params = species_params

    def hamiltonian_operator(self):
        num_species = len(self.states)
        H = np.zeros((num_species, num_species))  # Initialize a matrix for the Hamiltonian operator

        # Diagonal elements represent the energy of each species
        for i in range(num_species):
            H[i, i] = self.states[i] * self.energies[i]

        # Off-diagonal elements represent interactions between species
        for i in range(num_species):
            for j in range(i+1, num_species):
                interaction_strength = self.species_params[i]['interaction_strength'][j]
                H[i, j] = interaction_strength
                H[j, i] = interaction_strength  # Symmetric interaction matrix

        return H

    def replication_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = self.states[i] * self.species_params[i]['replication_rate'] * (1 - self.states[i] / self.carrying_capacity)
        return deltas

    def decay_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['decay_rate']
        return deltas

    def lifespan_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['aging_rate'] * self.species_params[i]['lifespan']
        return deltas

    def anti_life_effects(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['anti_life_effect']
        return deltas

    def radiation_effect(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['radiation_effectiveness']
        return deltas

    def probability_of_interactions(self):
        same_species_birth = np.zeros(len(self.states))
        diff_species_birth = np.zeros(len(self.states))
        for i in range(len(self.states)):
            same_species_birth[i] = self.species_params[i]['prob_same_species']
            diff_species_birth[i] = self.species_params[i]['prob_diff_species']
        return same_species_birth, diff_species_birth

    def schrodinger_equation(self, wavefunction, time):
        H = self.hamiltonian_operator()
        psi_t = np.dot(np.exp(-1j * H * time), wavefunction)
        return psi_t

    def solve(self, generations):
        for _ in range(generations):
            delta_replication = self.replication_term()
            delta_decay = self.decay_term()
            delta_lifespan = self.lifespan_term()
            delta_antilife = self.anti_life_effects()
            delta_radiation = self.radiation_effect()
            same_species_birth, diff_species_birth = self.probability_of_interactions()

            self.states += (delta_replication + delta_decay + delta_lifespan + delta_antilife + delta_radiation + same_species_birth + diff_species_birth)

            # Ensure populations do not become negative
            self.states = np.maximum(0, self.states)
            
            # Check for extinct species and replace them with new ones
            extinct_species = np.where(self.states == 0)[0]
            for extinct_species_index in extinct_species:
                # Create a new species based on the parameters of the extinct species
                new_species_params = self.species_params[extinct_species_index].copy()
                new_species_params['replication_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['decay_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['aging_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['lifespan'] *= np.random.uniform(0.5, 1.5)
                new_species_params['anti_life_effect'] *= np.random.uniform(0.5, 1.5)
                new_species_params['radiation_effectiveness'] *= np.random.uniform(0.5, 1.5)
                new_species_params['prob_same_species'] *= np.random.uniform(0.5, 1.5)
                new_species_params['prob_diff_species'] *= np.random.uniform(0.5, 1.5)
                new_species_params['interaction_strength'] = np.random.uniform(-0.01, 0.01, len(self.states))
                # Initialize the new species with a small population
                self.states[extinct_species_index] = np.random.randint(1, 10)

        return self.states

# Example initial state vector for 20 species
initial_states = np.random.uniform(50, 150, 20)  # Initial population for each species
energies = np.random.uniform(0.5, 1.5, 20)  # Energy for each species
carrying_capacity = 1000.0  # Carrying capacity for the environment

# Parameters for 20 different species with variability and interaction strengths
species_params = [
    {
        'replication_rate': np.random.uniform(0.05, 0.15),
        'decay_rate': np.random.uniform(0.01, 0.1),
        'aging_rate': np.random.uniform(0.01, 0.03),
        'lifespan': np.random.uniform(10, 30),
        'anti_life_effect': np.random.uniform(0.5, 2),
        'radiation_effectiveness': np.random.uniform(0.005, 0.02),
        'prob_same_species': np.random.uniform(0.01, 0.03),
        'prob_diff_species': np.random.uniform(0.005, 0.02),
        'interaction_strength': np.random.uniform(-0.01, 0.01, 20)  # Interaction strength with other species
    } for _ in range(20)
]

# Create QuantumBiologicalSystem instance
quantum_system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params)

# Solve the system for 1 million generations
generations = 1000000
final_states = quantum_system.solve(generations)
print("Final species counts after 1 million generations:", final_states)

```

## Cell 9 (Code)

```python
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

class QuantumBiologicalSystemWithML(QuantumBiologicalSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_models = [AutoReg(self.states[:, i], lags=1).fit() for i in range(len(self.states[0]))]

    def predict_next_generation(self):
        predicted_states = np.zeros_like(self.states)
        for i, model in enumerate(self.predictor_models):
            predicted_states[:, i] = model.predict(start=len(self.states), end=len(self.states)+1)
        return predicted_states

class QuantumBiologicalSystem:
    def __init__(self, initial_states, energies, carrying_capacity, species_params):
        self.states = np.array(initial_states)
        self.energies = np.array(energies)
        self.carrying_capacity = carrying_capacity
        self.species_params = species_params

    def hamiltonian_operator(self):
        num_species = len(self.states)
        H = np.zeros((num_species, num_species))  # Initialize a matrix for the Hamiltonian operator

        # Diagonal elements represent the energy of each species
        for i in range(num_species):
            H[i, i] = self.states[i] * self.energies[i]

        # Off-diagonal elements represent interactions between species
        for i in range(num_species):
            for j in range(i+1, num_species):
                interaction_strength = self.species_params[i]['interaction_strength'][j]
                H[i, j] = interaction_strength
                H[j, i] = interaction_strength  # Symmetric interaction matrix

        return H

    def replication_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = self.states[i] * self.species_params[i]['replication_rate'] * (1 - self.states[i] / self.carrying_capacity)
        return deltas

    def decay_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['decay_rate']
        return deltas

    def lifespan_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['aging_rate'] * self.species_params[i]['lifespan']
        return deltas

    def anti_life_effects(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['anti_life_effect']
        return deltas

    def radiation_effect(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['radiation_effectiveness']
        return deltas

    def probability_of_interactions(self):
        same_species_birth = np.zeros(len(self.states))
        diff_species_birth = np.zeros(len(self.states))
        for i in range(len(self.states)):
            same_species_birth[i] = self.species_params[i]['prob_same_species']
            diff_species_birth[i] = self.species_params[i]['prob_diff_species']
        return same_species_birth, diff_species_birth

    def schrodinger_equation(self, wavefunction, time):
        H = self.hamiltonian_operator()
        psi_t = np.dot(np.exp(-1j * H * time), wavefunction)
        return psi_t

    def solve(self, generations):
        for _ in range(generations):
            delta_replication = self.replication_term()
            delta_decay = self.decay_term()
            delta_lifespan = self.lifespan_term()
            delta_antilife = self.anti_life_effects()
            delta_radiation = self.radiation_effect()
            same_species_birth, diff_species_birth = self.probability_of_interactions()

            self.states += (delta_replication + delta_decay + delta_lifespan + delta_antilife + delta_radiation + same_species_birth + diff_species_birth)

            # Ensure populations do not become negative
            self.states = np.maximum(0, self.states)
            
            # Check for extinct species and replace them with new ones
            extinct_species = np.where(self.states == 0)[0]
            for extinct_species_index in extinct_species:
                # Create a new species based on the parameters of the extinct species
                new_species_params = self.species_params[extinct_species_index].copy()
                new_species_params['replication_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['decay_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['aging_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['lifespan'] *= np.random.uniform(0.5, 1.5)
                new_species_params['anti_life_effect'] *= np.random.uniform(0.5, 1.5)
                new_species_params['radiation_effectiveness'] *= np.random.uniform(0.5, 1.5)
                new_species_params['prob_same_species'] *= np.random.uniform(0.5, 1.5)
                new_species_params['prob_diff_species'] *= np.random.uniform(0.5, 1.5)
                new_species_params['interaction_strength'] = np.random.uniform(-0.01, 0.01, len(self.states))
                # Initialize the new species with a small population
                self.states[extinct_species_index] = np.random.randint(1, 10)

        return self.states

class LargeScaleSimulation:
    def __init__(self, num_species):
        self.num_species = num_species
        self.states = np.random.randint(0, 100, size=num_species)

    def replication_term(self):
        return np.random.rand(self.num_species)

    def decay_term(self):
        return np.random.rand(self.num_species)

    def lifespan_term(self):
        return np.random.rand(self.num_species)

    def anti_life_effects(self):
        return np.random.rand(self.num_species)

    def radiation_effect(self):
        return np.random.rand(self.num_species)

    def probability_of_interactions(self):
        return np.random.rand(self.num_species), np.random.rand(self.num_species)

    def update_states(self):
        # Calculate replication term
        delta_replication = self.replication_term()

        # Calculate decay term
        delta_decay = self.decay_term()

        # Calculate lifespan term
        delta_lifespan = self.lifespan_term()

        # Calculate anti-life effects
        delta_antilife = self.anti_life_effects()

        # Calculate radiation effect
        delta_radiation = self.radiation_effect()

        # Calculate probability of interactions
        same_species_birth, diff_species_birth = self.probability_of_interactions()

        # Update states based on changes
        self.states = self.states.astype(float)  # Convert to float
        self.states += (delta_replication + delta_decay + delta_lifespan + delta_antilife
                        + delta_radiation + same_species_birth + diff_species_birth)

        # Ensure populations do not become negative
        self.states = np.maximum(0, self.states.astype(int))  # Convert back to integer

# Example initial state vector for 20 species
initial_states = np.random.uniform(50, 150, 20)  # Initial population for each species
energies = np.random.uniform(0.5, 1.5, 20)  # Energy for each species
carrying_capacity = 1000.0  # Carrying capacity for the environment

# Parameters for 20 different species with variability and interaction strengths
species_params = [
    {
        'replication_rate': np.random.uniform(0.05, 0.15),
        'decay_rate': np.random.uniform(0.01, 0.1),
        'aging_rate': np.random.uniform(0.01, 0.03),
        'lifespan': np.random.uniform(10, 30),
        'anti_life_effect': np.random.uniform(0.5, 2),
        'radiation_effectiveness': np.random.uniform(0.005, 0.02),
        'prob_same_species': np.random.uniform(0.01, 0.03),
        'prob_diff_species': np.random.uniform(0.005, 0.02),
        'interaction_strength': np.random.uniform(-0.01, 0.01, 20)  # Interaction strength with other species
    } for _ in range(20)
]

# Create QuantumBiologicalSystem instance
quantum_system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params)
num_species = 10000
simulation = LargeScaleSimulation(num_species)
# Solve the system for 1 million generations
generations = 1000000

final_states = quantum_system.solve(generations)
print("Final species counts after 1 million generations:", final_states)

```

## Cell 10 (Code)

```python
import numpy as np
from statsmodels.tsa.ar_model import AutoReg


class QuantumBiologicalSystem:
    def __init__(self, initial_states, energies, carrying_capacity, species_params):
        self.states = np.array(initial_states)
        self.energies = np.array(energies)
        self.carrying_capacity = carrying_capacity
        self.species_params = species_params

    def hamiltonian_operator(self):
        num_species = len(self.states)
        H = np.zeros((num_species, num_species))  # Initialize a matrix for the Hamiltonian operator

        # Diagonal elements represent the energy of each species
        for i in range(num_species):
            H[i, i] = self.states[i] * self.energies[i]

        # Off-diagonal elements represent interactions between species
        for i in range(num_species):
            for j in range(i+1, num_species):
                interaction_strength = self.species_params[i]['interaction_strength'][j]
                H[i, j] = interaction_strength
                H[j, i] = interaction_strength  # Symmetric interaction matrix

        return H

    def replication_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = self.states[i] * self.species_params[i]['replication_rate'] * (1 - self.states[i] / self.carrying_capacity)
        return deltas

    def decay_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['decay_rate']
        return deltas

    def lifespan_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['aging_rate'] * self.species_params[i]['lifespan']
        return deltas

    def anti_life_effects(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['anti_life_effect']
        return deltas

    def radiation_effect(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['radiation_effectiveness']
        return deltas

    def probability_of_interactions(self):
        same_species_birth = np.zeros(len(self.states))
        diff_species_birth = np.zeros(len(self.states))
        for i in range(len(self.states)):
            same_species_birth[i] = self.species_params[i]['prob_same_species']
            diff_species_birth[i] = self.species_params[i]['prob_diff_species']
        return same_species_birth, diff_species_birth

    def schrodinger_equation(self, wavefunction, time):
        H = self.hamiltonian_operator()
        psi_t = np.dot(np.exp(-1j * H * time), wavefunction)
        return psi_t

    def solve(self, generations):
        for _ in range(generations):
            delta_replication = self.replication_term()
            delta_decay = self.decay_term()
            delta_lifespan = self.lifespan_term()
            delta_antilife = self.anti_life_effects()
            delta_radiation = self.radiation_effect()
            same_species_birth, diff_species_birth = self.probability_of_interactions()

            self.states += (delta_replication + delta_decay + delta_lifespan + delta_antilife + delta_radiation + same_species_birth + diff_species_birth)

            # Ensure populations do not become negative
            self.states = np.maximum(0, self.states)
            
            # Check for extinct species and replace them with new ones
            extinct_species = np.where(self.states == 0)[0]
            for extinct_species_index in extinct_species:
                # Create a new species based on the parameters of the extinct species
                new_species_params = self.species_params[extinct_species_index].copy()
                new_species_params['replication_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['decay_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['aging_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['lifespan'] *= np.random.uniform(0.5, 1.5)
                new_species_params['anti_life_effect'] *= np.random.uniform(0.5, 1.5)
                new_species_params['radiation_effectiveness'] *= np.random.uniform(0.5, 1.5)
                new_species_params['prob_same_species'] *= np.random.uniform(0.5, 1.5)
                new_species_params['prob_diff_species'] *= np.random.uniform(0.5, 1.5)
                new_species_params['interaction_strength'] = np.random.uniform(-0.01, 0.01, len(self.states))
                # Initialize the new species with a small population
                self.states[extinct_species_index] = np.random.randint(1, 10)

        return self.states
    
    def update_states(self):
        # Calculate replication term
        delta_replication = self.replication_term()

        # Calculate decay term
        delta_decay = self.decay_term()

        # Calculate lifespan term
        delta_lifespan = self.lifespan_term()

        # Calculate anti-life effects
        delta_antilife = self.anti_life_effects()

        # Calculate radiation effect
        delta_radiation = self.radiation_effect()

        # Calculate probability of interactions
        same_species_birth, diff_species_birth = self.probability_of_interactions()

        # Update states based on changes
        self.states = self.states.astype(float)  # Convert to float
        self.states += (delta_replication + delta_decay + delta_lifespan + delta_antilife
                        + delta_radiation + same_species_birth + diff_species_birth)

        # Ensure populations do not become negative
        self.states = np.maximum(0, self.states.astype(int))  # Convert back to integer

class QuantumBiologicalSystemWithML(QuantumBiologicalSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_models = [AutoReg(self.states[:, i], lags=1).fit() for i in range(len(self.states[0]))]

    def predict_next_generation(self):
        predicted_states = np.zeros_like(self.states)
        for i, model in enumerate(self.predictor_models):
            predicted_states[:, i] = model.predict(start=len(self.states), end=len(self.states)+1)
        return predicted_states

class LargeScaleSimulation:
    def __init__(self, num_species):
        self.num_species = num_species
        self.states = np.random.randint(0, 100, size=num_species)

    def replication_term(self):
        return np.random.rand(self.num_species)

    def decay_term(self):
        return np.random.rand(self.num_species)

    def lifespan_term(self):
        return np.random.rand(self.num_species)

    def anti_life_effects(self):
        return np.random.rand(self.num_species)

    def radiation_effect(self):
        return np.random.rand(self.num_species)

    def probability_of_interactions(self):
        return np.random.rand(self.num_species), np.random.rand(self.num_species)

    def update_states(self):
        # Calculate replication term
        delta_replication = self.replication_term()

        # Calculate decay term
        delta_decay = self.decay_term()

        # Calculate lifespan term
        delta_lifespan = self.lifespan_term()

        # Calculate anti-life effects
        delta_antilife = self.anti_life_effects()

        # Calculate radiation effect
        delta_radiation = self.radiation_effect()

        # Calculate probability of interactions
        same_species_birth, diff_species_birth = self.probability_of_interactions()

        # Update states based on changes
        self.states = self.states.astype(float)  # Convert to float
        self.states += (delta_replication + delta_decay + delta_lifespan + delta_antilife
                        + delta_radiation + same_species_birth + diff_species_birth)

        # Ensure populations do not become negative
        self.states = np.maximum(0, self.states.astype(int))  # Convert back to integer

# Example initial state vector for 20 species
initial_states = np.random.uniform(50, 150, 20)  # Initial population for each species
energies = np.random.uniform(0.5, 1.5, 20)  # Energy for each species
carrying_capacity = 1000.0  # Carrying capacity for the environment

# Parameters for 20 different species with variability and interaction strengths
species_params = [
    {
        'replication_rate': np.random.uniform(0.05, 0.15),
        'decay_rate': np.random.uniform(0.01, 0.1),
        'aging_rate': np.random.uniform(0.01, 0.03),
        'lifespan': np.random.uniform(10, 30),
        'anti_life_effect': np.random.uniform(0.5, 2),
        'radiation_effectiveness': np.random.uniform(0.005, 0.02),
        'prob_same_species': np.random.uniform(0.01, 0.03),
        'prob_diff_species': np.random.uniform(0.005, 0.02),
        'interaction_strength': np.random.uniform(-0.01, 0.01, 20)  # Interaction strength with other species
    } for _ in range(20)
]

# Create QuantumBiologicalSystem instance
quantum_system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params)
num_species = 10000
simulation = LargeScaleSimulation(num_species)
# Solve the system for 1 million generations
generations = 1000000

final_states = quantum_system.solve(generations)
print("Final species counts after 1 million generations:", final_states)

data = []  # To store the population counts at each time step

num_steps=1000

for step in range(num_steps):
    # Record current state
    data.append(quantum_system.states.copy())

    # Update state for the next step
    quantum_system.update_states()

# Convert data to numpy array
data = np.array(data)

# Save data to a file (e.g., CSV)
np.savetxt('species_population_data.csv', data, delimiter=',')
```

## Cell 11 (Code)

```python
import numpy as np
import random
import datetime
import numpy as np
from faker import Faker
import csv
import pandas as pd

fake = Faker()

NUM_RECORDS = 50000  # Adjust to create a dataset of around 1GB
DOSE_CONVERSION_COEFFICIENT = 1.3e-2  # µSv/Bq for 137Cs
ANNUAL_INTAKE = 5  # kg/year

# Define mushroom species with radiocesium concentration statistics
species_data = {
    "Boletus edulis": {"Common Name": "Porcini", "Median": 580, "Min": 27, "Max": 1800, "Location": "Korosten, Ukraine", "Edibility": "Edible", "Melanin": False},
    "Leccinum aurantiacum": {"Common Name": "Red-Capped Scaber Stalk", "Median": 250, "Min": 15, "Max": 480, "Location": "Lugine, Ukraine", "Edibility": "Edible", "Melanin": False},
    "Leccinum scabrum": {"Common Name": "Brown Birch Bolete", "Median": 290, "Min": 18, "Max": 1400, "Location": "Chernobyl Exclusion Zone", "Edibility": "Edible", "Melanin": True},
    "Boletus luteus": {"Common Name": "Slippery Jack", "Median": 14500, "Min": 10000, "Max": 20000, "Location": "Veprin, Russia", "Edibility": "Edible", "Melanin": False},
    "Boletus chanteral": {"Common Name": "Chanterelle", "Median": 2550, "Min": 1500, "Max": 4000, "Location": "Veprin, Russia", "Edibility": "Edible", "Melanin": False},
    "Boletus russula": {"Common Name": "Russula", "Median": 8980, "Min": 5000, "Max": 15000, "Location": "Veprin, Russia", "Edibility": "Edible", "Melanin": False},
    "Amanita phalloides": {"Common Name": "Death Cap", "Median": 300, "Min": 200, "Max": 400, "Location": "Various", "Edibility": "Toxic", "Melanin": False},
    "Amanita muscaria": {"Common Name": "Fly Agaric", "Median": 1500, "Min": 1000, "Max": 2000, "Location": "Various", "Edibility": "Toxic", "Melanin": True},
    "Gyromitra esculenta": {"Common Name": "False Morel", "Median": 700, "Min": 500, "Max": 900, "Location": "Various", "Edibility": "Toxic", "Melanin": True},
    "Cladosporium sphaerospermum": {"Common Name": "Radiotrophic Fungus", "Location": "International Space Station", "Min": 0.2, "Max": 25.0, "Edibility": "Unknown", "Melanin": True},
    "Wangiella dermatitidis": {"Common Name": "Black Yeast", "Location": "Worldwide", "Min": 0.3, "Max": 22.0, "Edibility": "Unknown", "Melanin": True},
    "Cryptococcus neoformans": {"Common Name": "Cryptococcus", "Location": "Worldwide", "Min": 0.4, "Max": 24.0, "Edibility": "Unknown", "Melanin": True}
}




# Define contamination zones
zones = {
    "Zone 1": {"Min Contamination": 1480, "Max Contamination": 5000},
    "Zone 2": {"Min Contamination": 555, "Max Contamination": 1480},
    "Zone 3": {"Min Contamination": 185, "Max Contamination": 555},
    "Zone 4": {"Min Contamination": 37, "Max Contamination": 185}
}
# Provided Data
mushroom_data = [
    {"Strain ID": "1142-2", "Radiation (µSv/hr)": 0.00, "Groups of Field Radiation Levels": "Control", "No. of Diseased Plants": 11, "Total Number of Flowering Plants": 31, "Percentage of Diseased Plants": 35.5, "Edibility": "Edible"},
    {"Strain ID": "1191", "Radiation (µSv/hr)": 0.00, "Groups of Field Radiation Levels": "Control", "No. of Diseased Plants": 21, "Total Number of Flowering Plants": 27, "Percentage of Diseased Plants": 77.8, "Edibility": "Not Edible"},
    {"Strain ID": "1192", "Radiation (µSv/hr)": 0.00, "Groups of Field Radiation Levels": "Control", "No. of Diseased Plants": 9, "Total Number of Flowering Plants": 25, "Percentage of Diseased Plants": 36.0, "Edibility": "Edible"},
    {"Strain ID": "1164", "Radiation (µSv/hr)": 0.030, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 6, "Total Number of Flowering Plants": 27, "Percentage of Diseased Plants": 22.2, "Edibility": "Edible"},
    {"Strain ID": "1109", "Radiation (µSv/hr)": 0.050, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 3, "Total Number of Flowering Plants": 26, "Percentage of Diseased Plants": 11.5, "Edibility": "Not Edible"},
    {"Strain ID": "1165-1", "Radiation (µSv/hr)": 0.050, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 13, "Total Number of Flowering Plants": 32, "Percentage of Diseased Plants": 40.6, "Edibility": "Edible"},
    {"Strain ID": "1165-2", "Radiation (µSv/hr)": 0.050, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 12, "Total Number of Flowering Plants": 28, "Percentage of Diseased Plants": 42.9, "Edibility": "Not Edible"},
    {"Strain ID": "1163", "Radiation (µSv/hr)": 0.080, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 9, "Total Number of Flowering Plants": 28, "Percentage of Diseased Plants": 32.1, "Edibility": "Edible"},
    {"Strain ID": "1106", "Radiation (µSv/hr)": 0.080, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 9, "Total Number of Flowering Plants": 29, "Percentage of Diseased Plants": 31.0, "Edibility": "Not Edible"},
    {"Strain ID": "1102-2", "Radiation (µSv/hr)": 0.165, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 4, "Total Number of Flowering Plants": 28, "Percentage of Diseased Plants": 14.3, "Edibility": "Edible"},
    {"Strain ID": "1102-1", "Radiation (µSv/hr)": 0.196, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 4, "Total Number of Flowering Plants": 27, "Percentage of Diseased Plants": 14.8, "Edibility": "Not Edible"},
    {"Strain ID": "1101-3", "Radiation (µSv/hr)": 0.235, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 10, "Total Number of Flowering Plants": 25, "Percentage of Diseased Plants": 40.0, "Edibility": "Edible"},
    {"Strain ID": "1101-2", "Radiation (µSv/hr)": 0.240, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 10, "Total Number of Flowering Plants": 35, "Percentage of Diseased Plants": 28.6, "Edibility": "Not Edible"},
    {"Strain ID": "1161", "Radiation (µSv/hr)": 0.350, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 11, "Total Number of Flowering Plants": 31, "Percentage of Diseased Plants": 35.5, "Edibility": "Edible"},
    {"Strain ID": "1101-1", "Radiation (µSv/hr)": 1.234, "Groups of Field Radiation Levels": "High", "No. of Diseased Plants": 9, "Total Number of Flowering Plants": 32, "Percentage of Diseased Plants": 28.1, "Edibility": "Not Edible"},
    {"Strain ID": "1103-3", "Radiation (µSv/hr)": 3.560, "Groups of Field Radiation Levels": "High", "No. of Diseased Plants": 5, "Total Number of Flowering Plants": 34, "Percentage of Diseased Plants": 14.7, "Edibility": "Edible"},
    {"Strain ID": "1162", "Radiation (µSv/hr)": 8.350, "Groups of Field Radiation Levels": "High", "No. of Diseased Plants": 6, "Total Number of Flowering Plants": 33, "Percentage of Diseased Plants": 18.2, "Edibility": "Not Edible"},
    {"Strain ID": "1105", "Radiation (µSv/hr)": 21.030, "Groups of Field Radiation Levels": "High", "No. of Diseased Plants": 10, "Total Number of Flowering Plants": 36, "Percentage of Diseased Plants": 27.8, "Edibility": "Edible"}
]


radiation_data = [
    {"Strain ID": "1142", "Radiation at Origin": 0, "log (Radiation at Origin + 0.001)": -3.00, "Mean Log (Number of Viable Cells) Control ± SD": "5.45 ± 0.08", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-0.93 ± 0.08", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.87 ± 0.06"},
    {"Strain ID": "1192", "Radiation at Origin": 0, "log (Radiation at Origin + 0.001)": -3.00, "Mean Log (Number of Viable Cells) Control ± SD": "5.48 ± 0.05", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-0.82 ± 0.08", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.03 ± 0.11"},
    {"Strain ID": "1164", "Radiation at Origin": 0.03, "log (Radiation at Origin + 0.001)": -1.51, "Mean Log (Number of Viable Cells) Control ± SD": "5.28 ± 0.13", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.07 ± 0.22", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.62 ± 0.12"},
    {"Strain ID": "1102-1", "Radiation at Origin": 0.20, "log (Radiation at Origin + 0.001)": -0.71, "Mean Log (Number of Viable Cells) Control ± SD": "5.50 ± 0.08", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.08 ± 0.08", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.89 ± 0.09"},
    {"Strain ID": "1101-3", "Radiation at Origin": 0.24, "log (Radiation at Origin + 0.001)": -0.63, "Mean Log (Number of Viable Cells) Control ± SD": "5.57 ± 0.13", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.29 ± 0.13", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.66 ± 0.12"},
    {"Strain ID": "1101-2", "Radiation at Origin": 0.24, "log (Radiation at Origin + 0.001)": -0.62, "Mean Log (Number of Viable Cells) Control ± SD": "5.48 ± 0.02", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.26 ± 0.06", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.43 ± 0.27"},
    {"Strain ID": "1101-1", "Radiation at Origin": 1.23, "log (Radiation at Origin + 0.001)": 0.09, "Mean Log (Number of Viable Cells) Control ± SD": "5.65", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.51 ± 0.16", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.52 ± 0.13"},
    {"Strain ID": "1162", "Radiation at Origin": 8.35, "log (Radiation at Origin + 0.001)": 0.92, "Mean Log (Number of Viable Cells) Control ± SD": "5.41 ± 0.12", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.68 ± 0.12", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-4.36 ± 0.22"},
    {"Strain ID": "1105", "Radiation at Origin": 21.03, "log (Radiation at Origin + 0.001)": 1.32, "Mean Log (Number of Viable Cells) Control ± SD": "5.70 ± 0.07", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.50 ± 0.11", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.46 ± 0.06"},
]

location_data = [
    {"Strain ID": "1101-1", "Location": "Near Zalizia, Ukraine", "GPS Coordinates": "N51°8′60″ E30°7′12″", "Date of Collection": "September 2013", "Radiation (µSv/hr)": 1.234},
    {"Strain ID": "1101-2", "Location": "Near Zalizia, Ukraine", "GPS Coordinates": "N51°8′60″ E30°7′12″", "Date of Collection": "September 2013", "Radiation (µSv/hr)": 0.240},
    {"Strain ID": "1101-3", "Location": "Near Zalizia, Ukraine", "GPS Coordinates": "N51°8′60″ E30°7′12″", "Date of Collection": "September 2013", "Radiation (µSv/hr)": 0.235},
    {"Strain ID": "1102-1", "Location": "EcoCenter, Ukraine", "GPS Coordinates": "N51°12′36″ E30°0′0″", "Date of Collection": "September 2013", "Radiation (µSv/hr)": 0.196},
    {"Strain ID": "1102-2", "Location": "EcoCenter, Ukraine", "GPS Coordinates": "N51°12′36″ E30°0′0″", "Date of Collection": "September 2013", "Radiation (µSv/hr)": 0.165},
    {"Strain ID": "1103-3", "Location": "Vesniane, Ukraine", "GPS Coordinates": "N51°18′600″ E29°38′263″", "Date of Collection": "September 2013", "Radiation (µSv/hr)": 3.560},
    {"Strain ID": "1105", "Location": "Red Forest, Ukraine", "GPS Coordinates": "N51°13′48″ E30°2′23.999″", "Date of Collection": "September 2013", "Radiation (µSv/hr)": 21.030},
    {"Strain ID": "1106", "Location": "Hilton, Ukraine", "GPS Coordinates": "N51°32′378″ E21°10′427″", "Date of Collection": "September 2013", "Radiation (µSv/hr)": 0.080},
    {"Strain ID": "1109", "Location": "Ecopolis, Ukraine", "GPS Coordinates": "N51°23′355″ E30°04′225″", "Date of Collection": "September 2013", "Radiation (µSv/hr)": 0.050},
    {"Strain ID": "1161", "Location": "Chernobyl Village", "GPS Coordinates": "N51°33′000″ E31°11′212″", "Date of Collection": "June 2014", "Radiation (µSv/hr)": 0.350},
    {"Strain ID": "1162", "Location": "Vesniane, Ukraine", "GPS Coordinates": "N51°10′48″ E30°22′12″", "Date of Collection": "June 2014", "Radiation (µSv/hr)": 8.350},
    {"Strain ID": "1163", "Location": "Ivankov, Ukraine", "GPS Coordinates": "N51°13′48″ E30°1′11.999″", "Date of Collection": "June 2014", "Radiation (µSv/hr)": 0.080},
    {"Strain ID": "1164", "Location": "Voronkov, Ukraine", "GPS Coordinates": "N50°7′48″ E30°31′48″", "Date of Collection": "June 2014", "Radiation (µSv/hr)": 0.030},
    {"Strain ID": "1165-1", "Location": "Near Chernobyl, Ukraine", "GPS Coordinates": "N51°12′36″ E30°0′0″", "Date of Collection": "June 2014", "Radiation (µSv/hr)": 0.050},
    {"Strain ID": "1165-2", "Location": "Near Chernobyl, Ukraine", "GPS Coordinates": "N51°12′36″ E30°0′0″", "Date of Collection": "June 2014", "Radiation (µSv/hr)": 0.050},
    {"Strain ID": "1142-2", "Location": "Budaörs, Hungary", "GPS Coordinates": "47°27′46.2″N 18°55′15.0″E", "Date of Collection": "June 2014", "Radiation (µSv/hr)": 0},
    {"Strain ID": "1191", "Location": "Bingen-am-Rhein, Germany", "GPS Coordinates": "49°57′43.9″N 7°54′11.5″E", "Date of Collection": "August 2014", "Radiation (µSv/hr)": 0},
    {"Strain ID": "1192", "Location": "Monheim-am-Rhein, Germany", "GPS Coordinates": "51°05′59.9″N 6°54′13.2″E", "Date of Collection": "August 2014", "Radiation (µSv/hr)": 0},
]


# Function to calculate fungi growth rate based on melanin presence and radiation exposure
def calculate_fungi_growth_rate(melanin_present, radiation_level):
    if melanin_present:
        return 1.5 * radiation_level
    else:
        return 1.0

# Function to simulate fungi growth with radiation effects
def simulate_fungi_growth(location, radiation_level, fungi_species):
    if "reactor" in location.lower() or "soil" in location.lower():
        if radiation_level > 0:
            if fungi_species in ["Cladosporium sphaerospermum", "Wangiella dermatitidis", "Cryptococcus neoformans"]:
                return "High (Radiotropism)"
            else:
                return "High"
        else:
            return "Low"
    else:
        return "Normal"

# Function to calculate previous decaying rate
def calculate_previous_decaying_rate(years_since_2024):
    half_life = 4468e6  # Uranium-238 half-life in years
    return 0.5 ** (years_since_2024 / half_life)

# Function to calculate present decaying rate
def calculate_present_decaying_rate(years_since_2024):
    half_life = 5730  # Carbon-14 half-life in years
    return 0.5 ** (years_since_2024 / half_life)

# Function to estimate life and death cycles based on historical data
def estimate_life_and_death_cycles(previous_decaying_rate, present_decaying_rate):
    decay_rate_difference = previous_decaying_rate - present_decaying_rate
    estimated_cycles = max(int(decay_rate_difference * 10), 1)  # Adjust scaling factor as needed
    return estimated_cycles

# Helper function to calculate committed effective dose
def calculate_committed_effective_dose(concentration):
    return concentration * DOSE_CONVERSION_COEFFICIENT * ANNUAL_INTAKE

# Helper function to calculate derived fields
def calculate_derived_fields(record):
    try:
        mean_log_cells_control = float(record["Mean Log (Number of Viable Cells) Control ± SD"].split(" ± ")[0])
        mean_log_cells_30_min = float(record["Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD"].split(" ± ")[0])
        mean_log_cells_60_min = float(record["Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD"].split(" ± ")[0])
        record["Viable Cells After 30 min (%)"] = round(100 * (10 ** (mean_log_cells_control + mean_log_cells_30_min)) / (10 ** mean_log_cells_control), 2)
        record["Viable Cells After 60 min (%)"] = round(100 * (10 ** (mean_log_cells_control + mean_log_cells_60_min)) / (10 ** mean_log_cells_control), 2)
    except:
        record["Viable Cells After 30 min (%)"] = None
        record["Viable Cells After 60 min (%)"] = None
    return record

# Function to calculate radiation dose
def calculate_dose(radiation, annual_intake):
    return radiation * annual_intake * DOSE_CONVERSION_COEFFICIENT

# Function to generate a random date between two dates
def random_date(start_date, end_date):
    return start_date + datetime.timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
    )

# Function to estimate birth and death dates
def estimate_dates(years_since_2024, previous_decay_rate, present_decay_rate):
    birth_date = random_date(datetime.datetime(2024 - years_since_2024, 1, 1), datetime.datetime(2024, 12, 31))
    death_date = random_date(birth_date, datetime.datetime(2054, 12, 31))
    return birth_date, death_date

# Generate dataset
data = []

for _ in range(NUM_RECORDS):
    mushroom_record = random.choice(mushroom_data)
    radiation_record = random.choice(radiation_data)
    location_record = random.choice(location_data)
    
    # Combine records by Strain ID
    combined_record = {**mushroom_record, **radiation_record, **location_record}
    
    # Calculate derived fields
    combined_record = calculate_derived_fields(combined_record)
    
    # Randomly select mushroom species and contamination zone
    species = random.choice(list(species_data.keys()))
    zone = random.choice(list(zones.keys()))
    
    # Generate random concentration within the specified range for the species
    concentration = random.uniform(species_data[species]["Min"], species_data[species]["Max"])
    
    # Calculate the committed effective dose
    committed_effective_dose = calculate_committed_effective_dose(concentration)

    # Determine if the selected species has melanin
    melanin_present = species_data[species]["Melanin"]

    # Calculate estimated dates of birth and death cycles
    previous_decay_rate = np.random.uniform(0.5, 1.5)
    present_decay_rate = np.random.uniform(0.5, 1.5)
    years_since_2024 = np.random.randint(1, 31)
    birth_date, death_date = estimate_dates(years_since_2024, previous_decay_rate, present_decay_rate)

    # Calculate the growth rate of fungi based on melanin presence and radiation exposure
    growth_rate = calculate_fungi_growth_rate(melanin_present, radiation_record["Radiation at Origin"])

    # Simulate fungi growth based on location and radiation level
    fungi_growth_rate = simulate_fungi_growth(location_record["Location"], combined_record["Radiation (µSv/hr)"], species)

    # Calculate years since 2024
    collection_date = fake.date_between(start_date='-36y', end_date='today')
    years_since_2024 = (datetime.date.today() - collection_date).days / 365.25
    
    # Add mushroom-related data to the combined record
    combined_record.update({
        "Species Name": species,
        "Common Name": species_data[species]["Common Name"],
        "Location": species_data[species]["Location"],
        "Latitude": random.uniform(50.0, 52.0),  # Approx latitude range for Ukraine and Russia locations
        "Longitude": random.uniform(27.0, 30.0),  # Approx longitude range for Ukraine and Russia locations
        "Habitat": random.choice(["Forest", "Meadow", "Urban"]),
        "Soil Type": random.choice(["Sandy", "Clay", "Loam"]),
        "pH Level": round(random.uniform(4.0, 8.0), 1),
        "Contamination Zone": zone,
        "Radioactive Contamination Level (kBq/m²)": round(random.uniform(zones[zone]["Min Contamination"], zones[zone]["Max Contamination"]), 2),
        "Radioactive Isotope": "137Cs",
        "Concentration in Mushroom (Bq/kg)": round(concentration, 2),
        "Committed Effective Dose (mSv/year)": round(committed_effective_dose, 3),
        "Cap Diameter (cm)": round(random.uniform(2, 20), 1),
        "Stem Length (cm)": round(random.uniform(1, 20), 1),
        "Edibility": species_data[species]["Edibility"],
        "Melanin": melanin_present,
        "Fungi Growth Rate": fungi_growth_rate,
        "Previous Decay Rate": previous_decay_rate,
        "Present Decay Rate": present_decay_rate,
        "Years Since 2024": years_since_2024,
        "Estimated Birth Date": birth_date,
        "Estimated Death Date": death_date,
        # Additional columns
        "Previous Decaying Rate": calculate_previous_decaying_rate(years_since_2024),
        "Present Decaying Rate": calculate_present_decaying_rate(years_since_2024),
        "Life and Death Cycles": estimate_life_and_death_cycles(
            calculate_previous_decaying_rate(years_since_2024),
            calculate_present_decaying_rate(years_since_2024)
        )
    })

    # Append to data
    data.append(combined_record)

df = pd.DataFrame(data)

print("start-3")

# Save to CSV
df.to_csv('combined_dataset.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

# QuantumBiologicalSystem class definition
class QuantumBiologicalSystem:
    def __init__(self, initial_states, energies, carrying_capacity, species_params):
        self.states = np.array(initial_states)
        self.energies = np.array(energies)
        self.carrying_capacity = carrying_capacity
        self.species_params = species_params

    def hamiltonian_operator(self):
        num_species = len(self.states)
        H = np.zeros((num_species, num_species))

        # Diagonal elements represent energies of the species
        for i in range(num_species):
            H[i, i] = self.energies[i]

        # Off-diagonal elements represent interactions between species
        for i in range(num_species):
            for j in range(i+1, num_species):
                interaction_strength = self.species_params[i]['interaction_strength'][j]
                H[i, j] = interaction_strength
                H[j, i] = interaction_strength  # Symmetric interaction matrix

        return H

    def replication_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = self.states[i] * self.species_params[i]['replication_rate'] * (1 - self.states[i] / self.carrying_capacity)
        return deltas

    def decay_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['decay_rate']
        return deltas

    def lifespan_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['aging_rate'] * self.species_params[i]['lifespan']
        return deltas

    def anti_life_effects(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['anti_life_effect']
        return deltas

    def radiation_effect(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['radiation_effectiveness']
        return deltas

    def probability_of_interactions(self):
        same_species_birth = np.zeros(len(self.states))
        diff_species_birth = np.zeros(len(self.states))
        for i in range(len(self.states)):
            same_species_birth[i] = self.species_params[i]['prob_same_species']
            diff_species_birth[i] = self.species_params[i]['prob_diff_species']
        return same_species_birth, diff_species_birth

    def schrodinger_equation(self, wavefunction, time):
        H = self.hamiltonian_operator()
        psi_t = np.dot(np.exp(-1j * H * time), wavefunction)
        return psi_t
    
    # Function to initialize the populations of species at the start time
    def initialize_population(start_time, num_species):
        # Initialize populations based on conditions during the Late Heavy Bombardment period
        # You can adjust these initial population values based on geological and environmental factors
        initial_populations = np.random.uniform(low=1, high=100, size=num_species)
        return initial_populations

    # Define the start time of the simulation as the Late Heavy Bombardment period (around 4.1 to 3.8 billion years ago)
    start_time = datetime.datetime(year=4000, month=1, day=1)

    def solve(self, generations):
        for _ in range(generations):
            delta_replication = self.replication_term()
            delta_decay = self.decay_term()
            delta_lifespan = self.lifespan_term()
            delta_antilife = self.anti_life_effects()
            delta_radiation = self.radiation_effect()
            same_species_birth, diff_species_birth = self.probability_of_interactions()

            self.states += (delta_replication + delta_decay + delta_lifespan + delta_antilife + delta_radiation + same_species_birth + diff_species_birth)

            # Ensure populations do not become negative
            self.states = np.maximum(0, self.states)
        
            # Check for extinct species and replace them with new ones
            extinct_species = np.where(self.states == 0)[0]
            for extinct_species_index in extinct_species:
                # Create a new species based on the parameters of the extinct species
                new_species_params = self.species_params[extinct_species_index].copy()
                new_species_params['replication_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['decay_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['aging_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['lifespan'] *= np.random.uniform(0.5, 1.5)
                new_species_params['anti_life_effect'] *= np.random.uniform(0.5, 1.5)
                new_species_params['radiation_effectiveness'] *= np.random.uniform(0.5, 1.5)
                new_species_params['prob_same_species'] *= np.random.uniform(0.5, 1.5)
                new_species_params['prob_diff_species'] *= np.random.uniform(0.5, 1.5)
                new_species_params['interaction_strength'] = np.random.uniform(-0.01, 0.01, len(self.states))
                new_species_params['competitive_strength'] *= np.random.uniform(0.5, 1.5)
                new_species_params['predation_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['birth_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['mutation_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['dispersal_rate'] *= np.random.uniform(0.5, 1.5)
                new_species_params['environmental_tolerance'] *= np.random.uniform(0.5, 1.5)
                new_species_params['carrying_capacity'] *= np.random.uniform(0.5, 1.5)
                new_species_params['optimal_temperature'] *= np.random.uniform(0.5, 1.5)
                new_species_params['optimal_humidity'] *= np.random.uniform(0.5, 1.5)
                new_species_params['optimal_ph'] *= np.random.uniform(0.5, 1.5)
                # Initialize the new species with a small population
                self.states[extinct_species_index] = np.random.randint(1, 10)

        return self.states


    def update_states(self):
        delta_replication = self.replication_term()
        delta_decay = self.decay_term()
        delta_lifespan = self.lifespan_term()
        delta_antilife = self.anti_life_effects()
        delta_radiation = self.radiation_effect()
        same_species_birth, diff_species_birth = self.probability_of_interactions()

        self.states += (delta_replication + delta_decay + delta_lifespan + delta_antilife + delta_radiation + same_species_birth + diff_species_birth)

        # Ensure populations do not become negative
        self.states = np.maximum(0, self.states)
    
        # Check for extinct species and replace them with new ones
        extinct_species = np.where(self.states == 0)[0]
        for extinct_species_index in extinct_species:
        # Create a new species based on the parameters of the extinct species
            new_species_params = self.species_params[extinct_species_index].copy()
            new_species_params['replication_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['decay_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['aging_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['lifespan'] *= np.random.uniform(0.5, 1.5)
            new_species_params['anti_life_effect'] *= np.random.uniform(0.5, 1.5)
            new_species_params['radiation_effectiveness'] *= np.random.uniform(0.5, 1.5)
            new_species_params['prob_same_species'] *= np.random.uniform(0.5, 1.5)
            new_species_params['prob_diff_species'] *= np.random.uniform(0.5, 1.5)
            new_species_params['interaction_strength'] = np.random.uniform(-0.01, 0.01, len(self.states))
            new_species_params['competitive_strength'] *= np.random.uniform(0.5, 1.5)
            new_species_params['predation_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['birth_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['mutation_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['dispersal_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['environmental_tolerance'] *= np.random.uniform(0.5, 1.5)
            new_species_params['carrying_capacity'] *= np.random.uniform(0.5, 1.5)
            new_species_params['optimal_temperature'] *= np.random.uniform(0.5, 1.5)
            new_species_params['optimal_humidity'] *= np.random.uniform(0.5, 1.5)
            new_species_params['optimal_ph'] *= np.random.uniform(0.5, 1.5)
            # Initialize the new species with a small population
            self.states[extinct_species_index] = np.random.randint(1, 10)

        return self.states


# Example initial state vector for 400 species
initial_states = np.random.uniform(50, 150, 50)  # Initial population for each species
energies = np.random.uniform(0.5, 1.5, 500)  # Energy for each species
carrying_capacity = 1000.0  # Carrying capacity for the environment

# Parameters for 20 different species with variability and interaction strengths
species_params = [
    {
        'replication_rate': np.random.uniform(0.03, 0.08),  # Adjusted for longer generation times
        'decay_rate': np.random.uniform(0.005, 0.02),  # Adjusted for longer lifespan and lower mortality rates
        'aging_rate': np.random.uniform(0.001, 0.005),  # Adjusted for slower aging in longer-lived species
        'lifespan': np.random.uniform(20, 50),  # Adjusted for longer average lifespan
        'anti_life_effect': np.random.uniform(0.8, 1.5),  # Adjusted for reduced predation and competition pressure
        'radiation_effectiveness': np.random.uniform(0.001, 0.005),  # Adjusted for historical radiation levels
        'prob_same_species': np.random.uniform(0.02, 0.05),  # Adjusted for social behavior and cooperation
        'prob_diff_species': np.random.uniform(0.01, 0.03),  # Adjusted for historical species interactions
        'interaction_strength': np.random.uniform(-0.005, 0.005, 20),  # Adjusted for competitive or cooperative interactions
        'competitive_strength': np.random.uniform(0.01, 0.1),  # Adjusted for resource competition
        'predation_rate': np.random.uniform(0.005, 0.02),  # Adjusted for historical predation pressure
        'birth_rate': np.random.uniform(0.01, 0.1), # Adjusted for historical birth rates
        'mutation_rate': np.random.uniform(0.0005, 0.001),  # Adjusted for genetic variation rates
        'dispersal_rate': np.random.uniform(0.05, 0.2),  # Adjusted for historical dispersal patterns
        'environmental_tolerance': np.random.uniform(0.05, 0.2),  # Adjusted for habitat diversity
        'carrying_capacity': np.random.uniform(500, 1000),  # Adjusted for historical resource availability
        'optimal_temperature': np.random.uniform(15, 25),  # Adjusted for historical climate data
        'optimal_humidity': np.random.uniform(50, 70),  # Adjusted for historical climate data
        'optimal_ph': np.random.uniform(6, 7.5),  # Adjusted for historical environmental conditions
    } for _ in range(150)
]


# Create QuantumBiologicalSystem instance
quantum_system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params)

# Solve the system for 1 million generations
generations = 4000000
final_states = quantum_system.solve(generations)
print("Final species counts after 1 million generations:", final_states)

# Data to store the population counts at each time step
data = []

num_steps = 50000

for step in range(num_steps):
    # Record current state
    data.append(quantum_system.states.copy())

    # Update state for the next step
    quantum_system.update_states()

# Convert data to numpy array
data = np.array(data)

# Save data to a file (e.g., CSV)
#data.to_csv('species_population_data.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
output_file = 'species_population_data.csv'
np.savetxt(output_file, data, delimiter=',')
print(f"Data has been written to {output_file}")
```

## Cell 12 (Code)

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Load the data from the CSV file without header
data = pd.read_csv('species_population_data.csv', header=None, index_col=0)

# Assign column names based on your understanding
num_species = data.shape[1]
column_names = [f'Species_{i+1}' for i in range(num_species)]
data.columns = column_names

# Display the first few rows of the data to verify column names assignment
print(data.head())

# Plot the population trends of a few selected species
selected_species = ['Species_1', 'Species_2', 'Species_3', 'Species_4', 'Species_5', 'Species_6', 'Species_7', 'Species_8', 'Species_9', 'Species_10',
                    'Species_11', 'Species_12', 'Species_13', 'Species_14', 'Species_15', 'Species_16', 'Species_17', 'Species_18', 'Species_19', 'Species_20']

# Check if all selected species columns exist in the DataFrame
if all(species in data.columns for species in selected_species):
    data[selected_species].plot(figsize=(12, 6))
    plt.title('Population Trends of Selected Species')
    plt.xlabel('Generations')
    plt.ylabel('Population')
    plt.legend(title='Species')
    plt.show()
else:
    print("Specified species columns not found in the DataFrame.")

# Calculate and plot the autocorrelation function for a selected species
species_to_analyze = 'Species_2'
if species_to_analyze in data.columns:
    plot_acf(data[species_to_analyze], lags=50)
    plt.figure(figsize=(10, 8))  # Set figure size here
    plt.title(f'Autocorrelation of {species_to_analyze}')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.show()
else:
    print(f"Species column '{species_to_analyze}' not found in the DataFrame.")

# Plot both ACF and PACF
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
if species_to_analyze in data.columns:
    plot_acf(data[species_to_analyze], lags=50, ax=ax[0])
    ax[0].set_title(f'Autocorrelation of {species_to_analyze}')

    plot_pacf(data[species_to_analyze], lags=50, ax=ax[1])
    ax[1].set_title(f'Partial Autocorrelation of {species_to_analyze}')

    plt.xlabel('Lags')
    plt.tight_layout()
    plt.show()
else:
    print(f"Species column '{species_to_analyze}' not found in the DataFrame.")

# Augmented Dickey-Fuller Test for stationarity
if species_to_analyze in data.columns:
    result = adfuller(data[species_to_analyze])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
else:
    print(f"Species column '{species_to_analyze}' not found in the DataFrame.")

# Save the updated data with column names to a new CSV file
data.to_csv('species_population_data_with_names.csv', index_label='Extinct_Species_Index')

# Plot population data for all species in data
for species in range(data.shape[1]):
    plt.plot(data.iloc[:, species], label=f'Species {species+1}')

plt.xlabel('UM Steps')
plt.ylabel('Population')
plt.title('Population Data Over UM Steps')
plt.legend(loc="upper right")
plt.show()
```

## Cell 13 (Code)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Define the file path and chunk size
file_path = 'species_population_data.csv'
chunk_size = 10**6  # Adjust based on your memory capacity

# Initialize an empty DataFrame to store aggregated results
aggregated_data = []

# Process the file in chunks
for chunk in pd.read_csv(file_path, chunksize=chunk_size, header=None):
    # Assuming the data has 8 generations and UM Steps as rows
    aggregated_data.append(chunk)

# Concatenate all the chunks into a single DataFrame
df = pd.concat(aggregated_data, ignore_index=True)

# Plot the data (example for line plot)
for generation in range(8):
    plt.plot(df[generation], label=f'Generation {generation+1}')

plt.xlabel('UM Steps')
plt.ylabel('Population')
plt.title('Population Data Over Generations')
plt.legend()
plt.show()
```

## Cell 14 (Code)

```python

from deap import base, creator, tools, algorithms
import numpy as np
from scipy.linalg import expm
import csv
from random import uniform
import datetime
import pandas as pd
from random import randint
from faker import Faker
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

# Create a Faker instance
fake = Faker()
data = []
class NoRepetitiveLogs(logging.Filter):
    def __init__(self, max_messages=10000):
        super().__init__()
        self.logged_messages = set()
        self.max_messages = max_messages

    def filter(self, record):
        message = record.getMessage()
        if message in self.logged_messages:
            return False
        self.logged_messages.add(message)
        if len(self.logged_messages) > self.max_messages:
            self.logged_messages.pop()
        return True

# Logger setup
logger = logging.getLogger('quantum_bio_system')
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler
file_handler = logging.FileHandler('evolution.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Custom filter
no_repetitive_logs_filter = NoRepetitiveLogs()
logger.addFilter(no_repetitive_logs_filter)

# QuantumBiologicalSystem class definition
class QuantumBiologicalSystem:
    def __init__(self, initial_states, energies, carrying_capacity, species_params,mutation_rate):
        logger.info("Initializing QuantumBiologicalSystem")
        self.initial_states = initial_states
        self.states = np.array(initial_states, dtype=np.float64)
        self.energies = np.array(energies, dtype=np.float64)
        self.carrying_capacity = carrying_capacity
        self.species_params = species_params
        self.mutation_rate = mutation_rate
        self.rna_sequences = [self.generate_rna_sequence(species['dna_sequence']) for species in self.species_params]
        self.dna_sequences = [species['dna_sequence'] for species in self.species_params]

        # Update species_params with dna and rna sequences
        for i, species in enumerate(self.species_params):
            species['dna_sequence'] = self.dna_sequences[i]
            species['rna_sequence'] = self.rna_sequences[i]

    def hamiltonian_operator(self):
        logger.info("Calculating Hamiltonian operator")
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
                        logger.warning(f"Interaction_strength list length shorter than expected at ({i}, {j})")

        return H

    def schrodinger_equation(self, wavefunction, time_step):
        try:
            H = self.hamiltonian_operator() # Example Hamiltonian construction
            if H.dtype != np.complex128:
              H = H.astype(np.complex128)
            exp_H = expm(-1j * H * time_step) # Ensure proper dtype for matrix exponential
            logger.info(f"Exponential of Hamiltonian exp_H:\n{exp_H}")
            psi_t = exp_H @ wavefunction  # Example time evolution using matrix multiplication
            logger.info(f"Resulting wavefunction psi_t:\n{psi_t}")
            return psi_t
        except Exception as e:
            logger.error(f"Error solving Schrödinger equation: {e}")
            raise e

    def replication_term(self):
        logger.info("Calculating replication term")
        deltas = np.zeros(len(self.states),dtype=np.complex128)
        for i in range(len(self.states)):
            deltas[i] = self.states[i] * self.species_params[i]['replication_rate'] * (1 - self.states[i] / self.carrying_capacity)
        return deltas

    def generate_dna_sequence(self,Length):
        # Example DNA initialization (random sequence of A, T, C, G)
        return ''.join(np.random.choice(['A', 'T', 'C', 'G'], length))

    def generate_rna_sequence(self, dna_sequence):
        transcription_map = str.maketrans('ATCG', 'UAGC')
        if isinstance(dna_sequence, np.ndarray):
            dna_sequence = dna_sequence.tobytes()  # Convert to bytes-like object
        return dna_sequence.translate(transcription_map)

    def transcribe_dna(self, dna):
        # Example RNA transcription (A->U, T->A, C->G, G->C)
        transcription_map = str.maketrans('ATCG', 'UAGC')
        return dna.translate(transcription_map)

    def mutate_dna(self, dna, mutation_rate):
        # Example DNA mutation
        dna_list = list(dna)
        for i in range(len(dna_list)):
            if np.random.rand() < mutation_rate:
                dna_list[i] = np.random.choice(list('ATCG'))
        return ''.join(dna_list)

    def update_species_params(self, index):
        # Example of how DNA/RNA affects species parameters (simplified)
        self.species_params[index]['replication_rate'] *= 1 + (self.rna_sequences[index].count('A') - self.rna_sequences[index].count('G')) * 0.001
        self.species_params[index]['decay_rate'] *= 1 + (self.rna_sequences[index].count('C') - self.rna_sequences[index].count('T')) * 0.001

    def decay_term(self):
        logger.info("Calculating decay term")
        deltas = np.zeros(len(self.states),dtype=np.complex128)
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['decay_rate']
        return deltas

    def lifespan_term(self):
        logger.info("Calculating lifespan term")
        deltas = np.zeros(len(self.states),dtype=np.complex128)
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['aging_rate'] * self.species_params[i]['lifespan']
        return deltas

    def anti_life_effects(self):
        logger.info("Calculating anti-life effects")
        deltas = np.zeros(len(self.states),dtype=np.complex128)
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['anti_life_effect']
        return deltas

    def radiation_effect(self):
        logger.info("Calculating radiation effect")
        deltas = np.zeros(len(self.states),dtype=np.complex128)
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['radiation_effectiveness']
        return deltas

    def compute_same_species_birth(self):
        logger.info("Calculating same species birth")
        deltas = np.zeros(len(self.states),dtype=np.complex128)
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                if i != j:
                    deltas[i] += (self.species_params[i]['prob_same_species'] *
                                  self.species_params[i]['birth_rate'] *
                                  self.states[j])
        return deltas

    def compute_diff_species_birth(self):
        logger.info("Calculating different species birth")
        deltas = np.zeros(len(self.states),dtype=np.complex128)
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                if i != j:
                    deltas[i] += (self.species_params[i]['prob_diff_species'] *
                                  self.species_params[i]['birth_rate'] *
                                  self.states[j])
        return deltas

    def replace_extinct_species(self):
        for i, state in enumerate(self.states):
            if state <= 0:
                new_species_params = {key: np.random.uniform(0, 1) for key in self.species_params[i]}
                self.species_params[i] = new_species_params
                self.states[i] = np.random.uniform(1, 100)
                logger.info(f"Species {i} went extinct and was replaced by a new species")

    def compute_radiation(self):
        deltas = np.zeros(len(self.states),dtype=np.complex128)
        for i in range(len(self.states)):
            try:
                deltas[i] = -self.states[i] * self.species_params[i]['radiation_effectiveness']
            except OverflowError:
                deltas[i] = 0  # Handle overflow gracefully
        return deltas

    def compute_interactions(self):
        logger.info("Calculating probability of interactions")
        same_species_birth = np.zeros(len(self.states))
        diff_species_birth = np.zeros(len(self.states))
        for i in range(len(self.states)):
            same_species_birth[i] = self.species_params[i]['prob_same_species']
            diff_species_birth[i] = self.species_params[i]['prob_diff_species']
        return same_species_birth, diff_species_birth

    def initialize_population(self, start_time, num_species):
        logger.info("Initializing population")
        initial_populations = np.random.uniform(low=1, high=100, size=num_species)
        return initial_populations
    start_time = datetime.datetime(year=4000, month=1, day=1)

    def solve(self, generations,num_time_steps, time_step):
        for _ in range(num_time_steps):
            delta_replication = self.replication_term()
            delta_decay = self.decay_term()
            delta_lifespan = self.lifespan_term()
            delta_antilife = self.anti_life_effects()
            delta_radiation = self.radiation_effect()
            same_species_birth = self.compute_same_species_birth()
            diff_species_birth = self.compute_diff_species_birth()
            probability_of_interaction = self.compute_interactions()

            # Handle None cases
            delta_replication = delta_replication if delta_replication is not None else np.zeros_like(self.states)
            delta_decay = delta_decay if delta_decay is not None else np.zeros_like(self.states)
            delta_lifespan = delta_lifespan if delta_lifespan is not None else np.zeros_like(self.states)
            delta_antilife = delta_antilife if delta_antilife is not None else np.zeros_like(self.states)
            delta_radiation = delta_radiation if delta_radiation is not None else np.zeros_like(self.states)
            same_species_birth = same_species_birth if same_species_birth is not None else np.zeros_like(self.states)
            diff_species_birth = diff_species_birth if diff_species_birth is not None else np.zeros_like(self.states)
            probability_of_interaction = probability_of_interaction if probability_of_interaction is not None else (np.zeros_like(self.states), np.zeros_like(self.states))

            wavefunction = self.states
            H = self.hamiltonian_operator()

            # Example: Adjust species parameters based on Hamiltonian
            for i in range(len(self.states)):
                #H = H.astype(np.float64)
                self.species_params[i]['competitive_strength'] += H[i, i] * 0.1  # Adjust based on diagonal elements of H
                self.species_params[i]['predation_rate'] += np.sum(H[i, :]) * 0.01  # Adjust based on row-wise sum of H
                self.species_params[i]['environmental_tolerance'] += np.mean(H[:, i]) * 0.005  # Adjust based on column-wise mean of H

            # Solve Schrödinger equation to determine species evolution
            wavefunction = self.states  # Example wavefunction (state populations)
            psi_t = self.schrodinger_equation(wavefunction, time_step)

            # Update species states based on computed psi_t
            self.states = psi_t

            # Ensure populations do not become negative
            self.states = np.maximum(0, self.states)

            # Replace extinct species with new ones
            self.replace_extinct_species()
            extinct_species = np.where(self.states == 0)[0]

            self.states += (
                delta_replication + delta_decay + delta_lifespan +
                delta_antilife + delta_radiation + same_species_birth +
                diff_species_birth + probability_of_interaction[0] +
                probability_of_interaction[1]
            )

            np.clip(self.states, 0, self.carrying_capacity, out=self.states)

            for extinct_species_index in extinct_species:
              # Create a new species based on the parameters of the extinct species
              new_species_params = self.species_params[extinct_species_index].copy()
              # Adjust parameters for the new species
              new_species_params['replication_rate'] *= np.random.uniform(0.5, 1.5)
              new_species_params['decay_rate'] *= np.random.uniform(0.5, 1.5)
              new_species_params['aging_rate'] *= np.random.uniform(0.5, 1.5)
              new_species_params['lifespan'] *= np.random.uniform(0.5, 1.5)
              new_species_params['anti_life_effect'] *= np.random.uniform(0.5, 1.5)
              new_species_params['radiation_effectiveness'] *= np.random.uniform(0.5, 1.5)
              new_species_params['prob_same_species'] *= np.random.uniform(0.5, 1.5)
              new_species_params['prob_diff_species'] *= np.random.uniform(0.5, 1.5)
              new_species_params['interaction_strength'] = np.random.uniform(-0.01, 0.01, len(self.states))
              new_species_params['competitive_strength'] *= np.random.uniform(0.5, 1.5)
              new_species_params['predation_rate'] *= np.random.uniform(0.5, 1.5)
              new_species_params['birth_rate'] *= np.random.uniform(0.5, 1.5)
              new_species_params['mutation_rate'] *= np.random.uniform(0.5, 1.5)
              new_species_params['dispersal_rate'] *= np.random.uniform(0.5, 1.5)
              new_species_params['environmental_tolerance'] *= np.random.uniform(0.5, 1.5)
              new_species_params['carrying_capacity'] *= np.random.uniform(0.5, 1.5)
              new_species_params['optimal_temperature'] *= np.random.uniform(0.5, 1.5)
              new_species_params['optimal_humidity'] *= np.random.uniform(0.5, 1.5)
              new_species_params['optimal_ph'] *= np.random.uniform(0.5, 1.5)
              new_species_params['Melanin'] = np.random.choice([True, False])  # Random boolean value for Melanin
              new_species_params['population'] = self.states[extinct_species_index]
              new_species_params['melanin_present'] = fake.boolean(),  # Using Faker for melanin presence
              new_species_params['dna_sequence'] = self.mutate_dna(self.generate_dna_sequence(1000))
              new_species_params['rna_sequence'] = self.generate_rna_sequence(new_species_params['dna_sequence'])


        return self.states

    def update_states(self):
        logger.info("Updating states")
        for _ in range(num_time_steps):
            delta_replication = self.replication_term()
            delta_decay = self.decay_term()
            delta_lifespan = self.lifespan_term()
            delta_antilife = self.anti_life_effects()
            delta_radiation = self.radiation_effect()
            same_species_birth = self.compute_same_species_birth()
            diff_species_birth = self.compute_diff_species_birth()
            probability_of_interaction = self.compute_interactions()

            # Handle None cases
            delta_replication = delta_replication if delta_replication is not None else np.zeros_like(self.states)
            delta_decay = delta_decay if delta_decay is not None else np.zeros_like(self.states)
            delta_lifespan = delta_lifespan if delta_lifespan is not None else np.zeros_like(self.states)
            delta_antilife = delta_antilife if delta_antilife is not None else np.zeros_like(self.states)
            delta_radiation = delta_radiation if delta_radiation is not None else np.zeros_like(self.states)
            same_species_birth = same_species_birth if same_species_birth is not None else np.zeros_like(self.states)
            diff_species_birth = diff_species_birth if diff_species_birth is not None else np.zeros_like(self.states)
            probability_of_interaction = probability_of_interaction if probability_of_interaction is not None else (np.zeros_like(self.states), np.zeros_like(self.states))

            wavefunction = self.states
            H = self.hamiltonian_operator()

            # Example: Adjust species parameters based on Hamiltonian
            for i in range(len(self.states)):
                #H = H.astype(np.float64)
                self.species_params[i]['competitive_strength'] += H[i, i] * 0.1  # Adjust based on diagonal elements of H
                self.species_params[i]['predation_rate'] += np.sum(H[i, :]) * 0.01  # Adjust based on row-wise sum of H
                self.species_params[i]['environmental_tolerance'] += np.mean(H[:, i]) * 0.005  # Adjust based on column-wise mean of H

            # Solve Schrödinger equation to determine species evolution
            wavefunction = self.states  # Example wavefunction (state populations)
            psi_t = self.schrodinger_equation(wavefunction, time_step)

            # Update species states based on computed psi_t
            self.states = psi_t

            # Replace extinct species with new ones
            self.replace_extinct_species()

            # Ensure populations do not become negative
            self.states = np.maximum(0, self.states)

              # Update species states
            self.states += (
            delta_replication + delta_decay + delta_lifespan +
            delta_antilife + delta_radiation + same_species_birth +
            diff_species_birth + probability_of_interaction[0] +
            probability_of_interaction[1])

            np.clip(self.states, 0, self.carrying_capacity, out=self.states)

        # Check for extinct species and replace them with new ones
        extinct_species = np.where(self.states == 0)[0]
        for extinct_species_index in extinct_species:
            # Create a new species based on the parameters of the extinct species
            new_species_params = self.species_params[extinct_species_index].copy()
            # Adjust parameters for the new species
            new_species_params['replication_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['decay_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['aging_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['lifespan'] *= np.random.uniform(0.5, 1.5)
            new_species_params['anti_life_effect'] *= np.random.uniform(0.5, 1.5)
            new_species_params['radiation_effectiveness'] *= np.random.uniform(0.5, 1.5)
            new_species_params['prob_same_species'] *= np.random.uniform(0.5, 1.5)
            new_species_params['prob_diff_species'] *= np.random.uniform(0.5, 1.5)
            new_species_params['interaction_strength'] = np.random.uniform(-0.01, 0.01, len(self.states))
            new_species_params['competitive_strength'] *= np.random.uniform(0.5, 1.5)
            new_species_params['predation_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['birth_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['mutation_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['dispersal_rate'] *= np.random.uniform(0.5, 1.5)
            new_species_params['environmental_tolerance'] *= np.random.uniform(0.5, 1.5)
            new_species_params['carrying_capacity'] *= np.random.uniform(0.5, 1.5)
            new_species_params['optimal_temperature'] *= np.random.uniform(0.5, 1.5)
            new_species_params['optimal_humidity'] *= np.random.uniform(0.5, 1.5)
            new_species_params['optimal_ph'] *= np.random.uniform(0.5, 1.5)
            new_species_params['Melanin'] = np.random.choice([True, False])  # Random boolean value for Melanin
            new_species_params['population'] = self.states[extinct_species_index]
            new_species_params['melanin_present'] = fake.boolean(),  # Using Faker for melanin presence
            new_species_params['dna_sequence'] = self.mutate_dna(self.generate_dna_sequence(1000))
            new_species_params['rna_sequence'] = self.generate_rna_sequence(new_species_params['dna_sequence'])

                        # Initialize the new species state
            self.states[extinct_species_index] = np.random.uniform(1, 100)
            # Update the species parameters with the new species
            self.species_params[extinct_species_index] = new_species_params
            logger.info(f"Species {extinct_species_index} went extinct and was replaced by a new species")

        return self.states

    def record_state(self):
        logger.info("Recording state")
        for i in range(len(self.states)):
            species_name = f"Species_{i}"
            combined_record = {
                "Species Name": species_name,
                "replication_rate": self.species_params[i]['replication_rate'],
                "decay_rate": self.species_params[i]['decay_rate'],
                "aging_rate": self.species_params[i]['aging_rate'],
                "lifespan": self.species_params[i]['lifespan'],
                "anti_life_effect": self.species_params[i]['anti_life_effect'],
                "radiation_effectiveness": self.species_params[i]['radiation_effectiveness'],
                "prob_same_species": self.species_params[i]['prob_same_species'],
                "prob_diff_species": self.species_params[i]['prob_diff_species'],
                "interaction_strength": self.species_params[i]['interaction_strength'],
                "competitive_strength": self.species_params[i]['competitive_strength'],
                "predation_rate": self.species_params[i]['predation_rate'],
                "birth_rate": self.species_params[i]['birth_rate'],
                "mutation_rate": self.species_params[i]['mutation_rate'],
                "dispersal_rate": self.species_params[i]['dispersal_rate'],
                "environmental_tolerance": self.species_params[i]['environmental_tolerance'],
                "carrying_capacity": self.species_params[i]['carrying_capacity'],
                "optimal_temperature": self.species_params[i]['optimal_temperature'],
                "optimal_humidity": self.species_params[i]['optimal_humidity'],
                "optimal_ph": self.species_params[i]['optimal_ph'],
                "Melanin": self.species_params[i]['Melanin'],
                "Population": self.states[i],
                "melanin_present": self.species_params[i]['melanin_present'],,
                "dna_sequence":self.species_params[i]['dna_sequence'],
                "rna_sequence":self.species_params[i]['rna_sequence']
            }
            combined_record.update(self.species_params[i])
            data.append(combined_record)
        return data


# Parameters
pop_size = 500
num_species = 90
generations = 4000  # Adjust as needed based on your time step and total simulation time
carrying_capacity = 100
ind_size = num_species * (2 + 19 + num_species)  # Individual size
initial_states = np.random.uniform(50, 150, num_species)
energies = np.random.uniform(0.5, 1.5, num_species)
num_time_steps = 4000
time_step = 1.0  # Time step in year
dna_size = 100  # Size of DNA sequence
rna_size = 100  # Size of RNA sequence
mutation_rate =  0.01  # Set your desired mutation rate here
wavefunction = np.array([0.5, 0.5j], dtype=np.complex128)
H = np.array([[1, 2j], [-2j, 3]], dtype=np.complex128)
# Create species parameters
species_params = [
    {
        'replication_rate': np.random.uniform(0.03, 0.08),
        'decay_rate': np.random.uniform(0.005, 0.02),
        'aging_rate': np.random.uniform(0.001, 0.005),
        'lifespan': np.random.randint(10, 50),
        'anti_life_effect': np.random.uniform(0.001, 0.005),
        'historical_radiation_level': np.random.uniform(0.001, 0.01),
        'prob_same_species': np.random.uniform(0.01, 0.05),
        'prob_diff_species': np.random.uniform(0.001, 0.01),
        'radiation_effectiveness': np.random.uniform(0.001, 0.01),
        'interaction_strength': np.random.uniform(-0.01, 0.01, 400),
        'competitive_strength': np.random.uniform(0.001, 0.005),
        'predation_rate': np.random.uniform(0.001, 0.005),
        'birth_rate': np.random.uniform(0.01, 0.1),
        'mutation_rate': np.random.uniform(0.001, 0.01),
        'dispersal_rate': np.random.uniform(0.001, 0.01),
        'environmental_tolerance': np.random.uniform(0.5, 1.0),
        'carrying_capacity': np.random.uniform(50, 200),
        'optimal_temperature': np.random.uniform(15, 35),
        'optimal_humidity': np.random.uniform(40, 80),
        'optimal_ph': np.random.uniform(5.5, 7.5),
        'contamination_zone': randint(0, 3),
        'Melanin': np.random.choice([True, False]),  # Random boolean value for Melanin
        'melanin_present': fake.boolean(),  # Using Faker for melanin presence
        'dna_sequence': ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=dna_size)),
        'rna_sequence': ''.join(np.random.choice(['A', 'U', 'C', 'G'], size=rna_size))
    }
    for _ in range(num_species)
]

data1 = []
quantum_system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params,mutation_rate)

quantum_system.solve(generations,num_time_steps,time_step)

for t in range(num_time_steps):
    quantum_system.update_states()
    quantum_system.record_state()
    # Record the current state
    for i in range(len(quantum_system.states)):
        species_name = f"Species_{i}"
        if quantum_system.states[i] > 0:
            combined_record = {
                "Species Name": species_name,
                "replication_rate": quantum_system.species_params[i]['replication_rate'],
                "decay_rate": quantum_system.species_params[i]['decay_rate'],
                "aging_rate": quantum_system.species_params[i]['aging_rate'],
                "lifespan": quantum_system.species_params[i]['lifespan'],
                "anti_life_effect": quantum_system.species_params[i]['anti_life_effect'],
                "radiation_effectiveness": quantum_system.species_params[i]['radiation_effectiveness'],
                "prob_same_species": quantum_system.species_params[i]['prob_same_species'],
                "prob_diff_species": quantum_system.species_params[i]['prob_diff_species'],
                "interaction_strength": quantum_system.species_params[i]['interaction_strength'],
                "competitive_strength": quantum_system.species_params[i]['competitive_strength'],
                "predation_rate": quantum_system.species_params[i]['predation_rate'],
                "birth_rate": quantum_system.species_params[i]['birth_rate'],
                "mutation_rate": quantum_system.species_params[i]['mutation_rate'],
                "dispersal_rate": quantum_system.species_params[i]['dispersal_rate'],
                "environmental_tolerance": quantum_system.species_params[i]['environmental_tolerance'],
                "carrying_capacity": quantum_system.species_params[i]['carrying_capacity'],
                "optimal_temperature": quantum_system.species_params[i]['optimal_temperature'],
                "optimal_humidity": quantum_system.species_params[i]['optimal_humidity'],
                "optimal_ph": quantum_system.species_params[i]['optimal_ph'],
                "Melanin": quantum_system.species_params[i]['Melanin'],
                "Population": quantum_system.states[i]['population'],
                "dna_sequence": quantum_system.species_params[i]['dna_sequence'],
                "rna_sequence": quantum_system.species_params[i]['rna_sequence']
            }
            combined_record.update(quantum_system.species_params[i])
            data1.append(combined_record)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data1)
# Save to CSV
df.to_csv('combineddata2.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
```

## Cell 15 (Code)

```python

# DEAP setup
def eval_species(species):
    logger.info("Evaluating species")
    initial_states = species[:num_species]
    energies = species[num_species:2*num_species]
    species_params = []

    param_start = 2*num_species
    for i in range(num_species):
        params = {
            'replication_rate': species[param_start],
            'decay_rate': species[param_start + 1],
            'aging_rate': species[param_start + 2],
            'lifespan': species[param_start + 3],
            'anti_life_effect': species[param_start + 4],
            'radiation_effectiveness': species[param_start + 5],
            'prob_same_species': species[param_start + 6],
            'prob_diff_species': species[param_start + 7],
            'interaction_strength': species[param_start + 8:param_start + 8 + num_species],
            'competitive_strength': species[param_start + 8 + num_species],
            'predation_rate': species[param_start + 9 + num_species],
            'birth_rate': species[param_start + 10 + num_species],
            'mutation_rate': species[param_start + 11 + num_species],
            'dispersal_rate': species[param_start + 12 + num_species],
            'environmental_tolerance': species[param_start + 13 + num_species],
            'carrying_capacity': species[param_start + 14 + num_species],
            'optimal_temperature': species[param_start + 15 + num_species],
            'optimal_humidity': species[param_start + 16 + num_species],
            'optimal_ph': species[param_start + 17 + num_species],
            'Melanin': species[param_start + 18 + num_species]
        }
        species_params.append(params)
        param_start += 19 + num_species

    system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params,mutation_rate)
    final_states = system.solve(generations)
    fitness = np.sum(final_states)  # Assuming fitness is the sum of the final populations

    logger.info(f"Fitness evaluated: {fitness}")
    return fitness,

# Adjust individual size to include DNA and RNA
ind_size += dna_size + rna_size
# DEAP genetic algorithm setup

# DEAP genetic algorithm setup
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("attr_float", np.random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ind_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", lambda ind: (sum(ind),))  # Dummy evaluation function

def main():
    logger.info("Starting DEAP Genetic Algorithm")
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, stats=stats, halloffame=hof, verbose=True)

    logger.info(f"Best individual: {hof[0]}")
    return population, stats, hof

if __name__ == "__main__":
    population, stats, hof = main()

    logger.info("Final population:")
    for ind in population:
        logger.info(ind)

    logger.info("Hall of Fame:")
    logger.info(hof)

    logger.info("Statistics:")
    logger.info(stats)

    pop = toolbox.population(n=pop_size)
    # Run the Genetic Algorithm
    for gen in range(generations):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the population with the offspring
        pop[:] = offspring

    # Select the best individual
    best_ind = tools.selBest(pop, 1)[0]
    best_params = best_ind

    best_params_dict = []
    for i in range(num_species):
        start = i * (2 + 19 + num_species)
        end = (i + 1) * (2 + 19 + num_species)
        dna_start = num_species * (2 + 19 + num_species)
        rna_start = dna_start + dna_size

        species_params = {
            'replication_rate': best_ind[start],
            'decay_rate': best_ind[start + 1],
            'aging_rate': best_ind[start + 2],
            'lifespan': best_ind[start + 3],
            'anti_life_effect': best_ind[start + 4],
            'radiation_effectiveness': best_ind[start + 5],
            'prob_same_species': best_ind[start + 6],
            'prob_diff_species': best_ind[start + 7],
            'interaction_strength': best_ind[start + 8:start + 8 + num_species],
            'competitive_strength': best_ind[start + 8 + num_species],
            'predation_rate': best_ind[start + 9 + num_species],
            'birth_rate': best_ind[start + 10 + num_species],
            'mutation_rate': best_ind[start + 11 + num_species],
            'dispersal_rate': best_ind[start + 12 + num_species],
            'environmental_tolerance': best_ind[start + 13 + num_species],
            'carrying_capacity': best_ind[start + 14 + num_species],
            'optimal_temperature': best_ind[start + 15 + num_species],
            'optimal_humidity': best_ind[start + 16 + num_species],
            'optimal_ph': best_ind[start + 17 + num_species],
            'Melanin': best_ind[start + 18 + num_species],
            'dna_sequence': best_ind[dna_start:dna_start + dna_size],
            'rna_sequence': best_ind[rna_start:rna_start + rna_size],
            'melanin_present': fake.boolean()  # Using Faker for melanin presence
        }
        best_params_dict.append(species_params)
  
    # Simulate with the best parameters
    quantum_system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, best_params_dict,mutation_rate)
    final_states = quantum_system.solve(generations)
    data = []
    for _ in range(num_time_steps):
        quantum_system.update_states()
        for i in range(len(quantum_system.states)):
            species_name = f"Species_{i}"
            if quantum_system.states[i] > 0:
                combined_record = {
                "Species Name": species_name,
                "replication_rate": quantum_system.species_params[i]['replication_rate'],
                "decay_rate": quantum_system.species_params[i]['decay_rate'],
                "aging_rate": quantum_system.species_params[i]['aging_rate'],
                "lifespan": quantum_system.species_params[i]['lifespan'],
                "anti_life_effect": quantum_system.species_params[i]['anti_life_effect'],
                "radiation_effectiveness": quantum_system.species_params[i]['radiation_effectiveness'],
                "prob_same_species": quantum_system.species_params[i]['prob_same_species'],
                "prob_diff_species": quantum_system.species_params[i]['prob_diff_species'],
                "interaction_strength": quantum_system.species_params[i]['interaction_strength'],
                "competitive_strength": quantum_system.species_params[i]['competitive_strength'],
                "predation_rate": quantum_system.species_params[i]['predation_rate'],
                "birth_rate": quantum_system.species_params[i]['birth_rate'],
                "mutation_rate": quantum_system.species_params[i]['mutation_rate'],
                "dispersal_rate": quantum_system.species_params[i]['dispersal_rate'],
                "environmental_tolerance": quantum_system.species_params[i]['environmental_tolerance'],
                "carrying_capacity": quantum_system.species_params[i]['carrying_capacity'],
                "optimal_temperature": quantum_system.species_params[i]['optimal_temperature'],
                "optimal_humidity": quantum_system.species_params[i]['optimal_humidity'],
                "optimal_ph": quantum_system.species_params[i]['optimal_ph'],
                "Melanin": quantum_system.species_params[i]['Melanin'],
                "Population": quantum_system.states[i],
                "dna_sequence":quantum_system.species_params[i]['dna_sequence'],
                "rna_sequence":quantum_system.species_params[i]['rna_sequence']
            }
            combined_record.update(quantum_system.species_params[i])
            data.append(combined_record)

    final_states = quantum_system.states
    print("Final species counts after optimization:", final_states)

    # Export data to CSV
    data = []
    output_file = 'species_population_dataoptimized.csv'
    df = pd.DataFrame(data)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"evolution_results_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

    print(f"Optimized data has been written to {output_file}")
```

