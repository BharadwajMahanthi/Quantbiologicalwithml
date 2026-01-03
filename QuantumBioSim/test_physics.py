
import numpy as np
from src.models import QuantumBiologicalSystem
import time

def test_physics():
    print("Testing Hadean Physics (Rad=50.0)...")
    
    # 1. Sensitive Species (No GATA)
    sensitive_params = [{
        'dna_sequence': "AAAAAAAAAA", # No GATA
        'replication_rate': 0.1, 'decay_rate': 0.1, 'aging_rate': 0.01,
        'lifespan': 20, 'anti_life_effect': 1.0, 
        'birth_rate': 0.05,
        'prob_same_species': 0.01, 'prob_diff_species': 0.0,
        'interaction_strength': 0.0
    }]
    
    # 2. Resistant Species (GATA)
    resistant_params = [{
        'dna_sequence': "TGATAATGATAA", # GATA
        'replication_rate': 0.1, 'decay_rate': 0.1, 'aging_rate': 0.01,
        'lifespan': 20, 'anti_life_effect': 1.0, 
        'birth_rate': 0.05,
        'prob_same_species': 0.01, 'prob_diff_species': 0.0,
        'interaction_strength': 0.0
    }]
    
    hadean_env = {"radiation": 50.0, "oxygen": 0.1, "water": 1.0, "food": 1.0, "co2": 20.0, "pollution": 5.0}
    
    # Run Sensitive
    print("\nRunning Sensitive Species (1 Step)...")
    sys_sens = QuantumBiologicalSystem([50.0], [0.0], 1000, sensitive_params, 0.0, hadean_env)
    t0 = time.time()
    final_sens = sys_sens.solve(generations=1)
    t1 = time.time()
    print(f"Sensitive Pop: {final_sens[0]:.2f} (Time: {(t1-t0)*1000:.2f}ms)")
    
    # Run Resistant
    print("\nRunning Resistant Species (1 Step)...")
    sys_res = QuantumBiologicalSystem([50.0], [0.0], 1000, resistant_params, 0.0, hadean_env)
    final_res = sys_res.solve(generations=1)
    print(f"Resistant Pop: {final_res[0]:.2f}")
    
    if final_sens[0] < final_res[0]:
        print("\nSUCCESS: Resistant species survived better.")
        if final_sens[0] < 50.0:
            print("Sensitive species declined (Correct).")
        if final_res[0] > 50.0:
            print("Resistant species grew (Correct).")
    else:
        print("\nFAILURE: Physics model not selecting correctly.")
        
if __name__ == "__main__":
    test_physics()
