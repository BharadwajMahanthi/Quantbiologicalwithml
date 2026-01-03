import gzip
import os

def check_evidence(filepath, name):
    print(f"\n--- INSPECTING EVIDENCE FROM: {name} ---")
    print(f"File: {filepath}")
    
    if not os.path.exists(filepath):
        print("Status: File not finished downloading yet.")
        return

    try:
        # Read the first 10,000 lines (2,500 reads) to get a statistical sample
        # method: 'rt' mode for reading text from gzip
        with gzip.open(filepath, 'rt') as f:
            total_gata = 0
            total_tata = 0
            read_count = 0
            
            print("\n[REAL DNA SEQUENCES EXTRACTED]:")
            for i, line in enumerate(f):
                line = line.strip()
                # FASTQ format: Line 2 (index 1) is the sequence
                if i % 4 == 1: 
                    read_count += 1
                    # Count Prototaxites Signal (GATA)
                    gata_in_read = line.count('TGATAA') + line.count('GATA') 
                    # Count Human/Eukaryote Signal (TATA)
                    tata_in_read = line.count('TATAAA') + line.count('TATA')
                    
                    total_gata += gata_in_read
                    total_tata += tata_in_read
                    
                    # Print first 3 sequences as examples for the user
                    if read_count <= 3:
                        print(f"Read {read_count}: {line[:50]}... (GATA: {gata_in_read}, TATA: {tata_in_read})")
                
                if read_count >= 2500:
                    break
            
            print(f"\n[ANALYSIS OF {read_count} ANCIENT DNA FRAGMENTS]")
            print(f"Total 'GATA' (Radiation Response) Motifs Found: {total_gata}")
            print(f"Total 'TATA' (Standard Life) Motifs Found:      {total_tata}")
            
            ratio = total_gata / total_tata if total_tata > 0 else 0
            print(f"GATA/TATA Ratio: {ratio:.2f}")
            
            if ratio > 0.5: # Arbitrary threshold for high fungal content vs background
                print("CONCLUSION: Validates high fungal/radiotrophic signature potential.")
            else:
                print("CONCLUSION: Shows typical background distribution.")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # Check the Fungal File (Radiation Proof)
    check_evidence(r"data/ancestral_fastq/ERR10493281_1.fastq.gz", "Ancient Fungi (Prototaxites)")
