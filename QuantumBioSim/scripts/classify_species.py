import os

def classify_taxonomy(file_path):
    print(f"Analyzing {file_path} for Human vs Non-Human content...")
    
    chunk_size = 100 * 1024 * 1024 # 100 MB
    
    human_count = 0
    microbial_count = 0
    unknown_count = 0
    total_reads = 0
    
    # Heuristics
    # Human GC Content: ~41% (Range 35-50% is safe window)
    # Microbial GC: Often High (>60%) or Low (<30%)
    
    # Alu Repeat (Most common human SINE, ~300bp, found in >10% of reads)
    # A short conserved snippet of Alu:
    ALU_SNIPPET = "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACC"
    ALU_SHORT = "GGCCGGGCGCGGTGG" # 15bp core
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            header = f.readline()
            if not header: break
            seq = f.readline().strip()
            plus = f.readline()
            qual = f.readline()
            
            # Stop if we processed 100MB worth of data (approx 500k reads)
            if total_reads > 500_000: 
                break
                
            total_reads += 1
            if not seq: continue
            
            # 1. Calculate GC Content
            gc_count = seq.count('G') + seq.count('C')
            gc_percent = (gc_count / len(seq)) * 100
            
            # 2. Classification Logic
            is_human_candidate = False
            is_microbial_candidate = False
            
            # Check Human Markers
            if (35 <= gc_percent <= 55):
                # GC matches Human Profile
                if 'TATAAA' in seq or ALU_SHORT in seq:
                    is_human_candidate = True
            
            # Check Microbial Markers
            if (gc_percent < 30) or (gc_percent > 65):
                is_microbial_candidate = True
            if 'TGATAA' in seq: # Fungal/Microbial GATA
                 is_microbial_candidate = True
                 is_human_candidate = False # Override if specifically Fungal
            
            # 3. Assign
            if is_human_candidate:
                human_count += 1
            elif is_microbial_candidate:
                microbial_count += 1
            else:
                # "Grey Zone" (Could be Human non-coding, or generic bacteria)
                unknown_count += 1

    print(f"--- SAMPLE RESULTS ({total_reads} reads) ---")
    print(f"Likely Human (Endogenous): {human_count} ({human_count/total_reads*100:.2f}%)")
    print(f"Likely Microbial (Exogenous): {microbial_count} ({microbial_count/total_reads*100:.2f}%)")
    print(f"Ambiguous/Unknown: {unknown_count} ({unknown_count/total_reads*100:.2f}%)")
    
    print("\n[INTERPRETATION]")
    print("Ancient Neanderthal genomes are typically 1-5% Endogenous Human DNA.")
    print("The rest is soil bacteria (Metagenome) from the cave floor.")
    
if __name__ == "__main__":
    path = r"C:\Users\mbpd1\Downloads\Quantbiologicalwithml\QuantumBioSim\data\ancestral_fastq\ERR229911.fastq\ERR229911.fastq"
    classify_taxonomy(path)
