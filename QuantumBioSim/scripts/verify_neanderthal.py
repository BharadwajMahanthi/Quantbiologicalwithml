import os

def analyze_neanderthal_dna(file_path, chunk_size_mb=100):
    print(f"Reading {chunk_size_mb} MB chunk from: {file_path}")
    
    tata_count = 0
    gata_count = 0
    total_reads = 0
    
    # Read only the first N bytes to avoid memory crash on 45GB file
    chunk_bytes = chunk_size_mb * 1024 * 1024
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Read a chunk
        block = f.read(chunk_bytes)
        lines = block.split('\n')
        
        # FASTQ format: 4 lines per read. Line 2 is sequence.
        # We start from index 1 and step by 4
        for i in range(1, len(lines), 4):
            if i < len(lines):
                seq = lines[i]
                total_reads += 1
                
                # Check for "Human" Metabolic Marker (TATA Box - TATAAA)
                if 'TATAAA' in seq:
                    tata_count += 1
                
                # Check for "Fungal" Radiotropism Marker (GATA - TGATAA)
                if 'TGATAA' in seq:
                    gata_count += 1

    print(f"Analysis Complete on {total_reads} reads.")
    print(f"Human Marker (TATAAA): {tata_count}")
    print(f"Fungal Marker (TGATAA): {gata_count}")
    
    if total_reads > 0:
        ratio = tata_count / total_reads
        print(f"TATA Frequency: {ratio:.5f}")
        return ratio
    return 0.0

if __name__ == "__main__":
    path = r"C:\Users\mbpd1\Downloads\Quantbiologicalwithml\QuantumBioSim\data\ancestral_fastq\ERR229911.fastq\ERR229911.fastq"
    analyze_neanderthal_dna(path)
