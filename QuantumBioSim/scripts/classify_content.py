import os

def classify_file(file_path):
    file_size = os.path.getsize(file_path)
    print(f"File Size: {file_size / (1024**3):.2f} GB")
    
    chunk_size = 100 * 1024 * 1024 # 100 MB Scan
    dna_count = 0
    rna_count = 0
    total_records_in_chunk = 0
    bytes_read = 0
    
    print(f"Scanning {chunk_size/1024/1024} MB sample...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Read lines until we hit chunk limit
        while bytes_read < chunk_size:
            # FASTQ has 4 lines per record
            header = f.readline()
            seq = f.readline()
            plus = f.readline()
            qual = f.readline()
            
            if not qual: break # EOF
            
            bytes_read += len(header) + len(seq) + len(plus) + len(qual)
            total_records_in_chunk += 1
            
            # Classification Logic
            # DNA = A, T, C, G, N
            # RNA = A, U, C, G, N (if represented explicitly, though usually cDNA)
            
            if 'U' in seq:
                rna_count += 1
            else:
                dna_count += 1
                
    # Extrapolate
    avg_sl = bytes_read / total_records_in_chunk
    estimated_total = int(file_size / avg_sl)
    
    print(f"--- SAMPLE RESULTS ({total_records_in_chunk} sequences) ---")
    print(f"DNA Sequences (T-based): {dna_count}")
    print(f"RNA Sequences (U-based): {rna_count}")
    
    print(f"\n--- TOTAL ESTIMATION (Extrapolated to 45GB) ---")
    print(f"Estimated Total Sequences: {estimated_total:,}")
    print(f"Classification: {'100% DNA' if rna_count == 0 else 'Mixed DNA/RNA'}")

if __name__ == "__main__":
    path = r"C:\Users\mbpd1\Downloads\Quantbiologicalwithml\QuantumBioSim\data\ancestral_fastq\ERR229911.fastq\ERR229911.fastq"
    classify_file(path)
