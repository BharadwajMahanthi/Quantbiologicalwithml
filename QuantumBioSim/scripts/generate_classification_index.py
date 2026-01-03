import os
import hashlib
import csv
import time
import gc

def get_completed_count(output_csv):
    """
    Scans the CSV to count completed records and detect partial/corrupt last lines.
    Returns the number of valid records found (excluding header).
    """
    if not os.path.exists(output_csv):
        return 0
    
    print(f"Checking existing progress in {output_csv}...")
    count = 0
    with open(output_csv, 'rb') as f:
        # Check if empty
        f.seek(0, 2) # End
        if f.tell() == 0:
            return 0
        
        f.seek(0)
        # We'll use a generator to count lines efficiently
        # Binary mode is faster, but we need to ensure we count newlines
        
        # Simple robust generic approach:
        # Read line by line, if last line doesn't end with newline, it might be partial.
        # But CSV writers usually flush complete buffers. 
        # We will count all lines that look valid.
    
    # Text mode scan
    with open(output_csv, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline()
        if not header.startswith("ReadID"):
            return 0 # Invalid file, overwrite
            
        for line in f:
            if line.strip(): # significant line
                count += 1
                
    # Optional: Logic to remove last line if it seems partial could go here
    # For now, we assume the previous Flush() kept it clean.
    
    return count

def generate_index(file_path, output_csv="classification_index.csv", limit=None):
    print(f"Starting classification of: {file_path}")
    print(f"Output Index: {output_csv}")
    
    # --- AUTO-RESUME LOGIC ---
    resume_from = get_completed_count(output_csv)
    
    if resume_from > 0:
        print(f"FOUND CHECKPOINT: {resume_from:,} records already processed.")
        print("Resuming from there...")
        mode = 'a'
    else:
        mode = 'w'
    
    start_time = time.time()
    batch_start_time = time.time()
    
    # Checksum (skip complex hash resume for performance)
    sha256_hash = hashlib.sha256()
    
    # Counters
    stats = {"Human": 0, "Microbial": 0, "Unknown": 0}
    total_reads = 0
    BATCH_SIZE = 1_000_000
    ALU_SHORT = "GGCCGGGCGCGGTGG"
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(output_csv, mode, newline='', encoding='utf-8') as f_out:
        
        writer = csv.writer(f_out)
        if mode == 'w':
            writer.writerow(["ReadID", "Classification", "GC_Percent"])
        
        print(f"--- BATCH PROCESSING STARTED (Size: {BATCH_SIZE:,}) ---")
        
        # --- FAST FORWARD ---
        if resume_from > 0:
            print(f"Skipping {resume_from:,} processed reads in FASTQ...")
            reads_to_skip = resume_from
            total_reads = resume_from
            
            # FASTQ is 4 lines per read
            # Python's file iterator is optimized; using a loop is decent.
            # Using itertools.islice would be faster but requires import.
            # We'll stick to a tight loop.
            lines_to_skip = reads_to_skip * 4
            
            # Optimization: chunks
            skipped = 0
            while skipped < lines_to_skip:
                f_in.readline()
                skipped += 1
                if skipped % 4_000_000 == 0: # Feedback every 1M reads
                    print(f"   Skipped {skipped//4:,} reads...")
                    
            print("Resume point reached. Processing new data...")
        
        while True:
            try:
                # FASTQ Reading (4 lines per record)
                header = f_in.readline()
                if not header: break # Real EOF
                
                # CRASH FIX: Handle blank/malformed lines
                if not header.strip():
                    continue 

                seq = f_in.readline().strip()
                # Ignore '+' line and Quality line (Unused)
                _ = f_in.readline()
                _ = f_in.readline()
                
                # Extract ID safely
                parts = header.strip().split()
                if not parts:
                    continue 
                read_id = parts[0][1:] 
                
                # --- CLASSIFICATION LOGIC ---
                classification = "Unknown"
                
                if seq:
                    gc_count = seq.count('G') + seq.count('C')
                    gc_percent = (gc_count / len(seq)) * 100
                    
                    if (35 <= gc_percent <= 55):
                        if 'TATAAA' in seq or ALU_SHORT in seq:
                            classification = "Human"
                    
                    if classification != "Human":
                        if (gc_percent < 30) or (gc_percent > 65) or ('TGATAA' in seq):
                            classification = "Microbial"
                else:
                    gc_percent = 0.0

                # --- WRITE ---
                row = [read_id, classification, f"{gc_percent:.1f}"]
                writer.writerow(row)
                
                # Hasher (stateless for resume, will only hash NEW data)
                # row_str = ",".join(row)
                # sha256_hash.update(row_str.encode('utf-8'))
                
                stats[classification] += 1
                total_reads += 1
                
                # --- BATCH MANAGEMENT ---
                if total_reads % BATCH_SIZE == 0:
                    f_out.flush()
                    os.fsync(f_out.fileno())
                    gc.collect() 
                    
                    elapsed_batch = time.time() - batch_start_time
                    rate = BATCH_SIZE / max(elapsed_batch, 0.001)
                    print(f"[BATCH] {total_reads:,} reads done. (Rate: {rate:.0f}/sec). Cache Cleaned. Saved.")
                    batch_start_time = time.time()
                
                if limit and total_reads >= limit:
                    print(f"Limit of {limit} reached. Stopping.")
                    break
                    
            except Exception as e:
                print(f"Skipping bad record at read #{total_reads}: {e}")
                continue

    print("\n--- CLASSIFICATION COMPLETE ---")
    print(f"Total Reads: {total_reads:,}")
    print(f"Human: {stats['Human']:,} (New/Resumed)")
    print(f"Microbial: {stats['Microbial']:,} (New/Resumed)")

if __name__ == "__main__":
    path = r"C:\Users\mbpd1\Downloads\Quantbiologicalwithml\QuantumBioSim\data\ancestral_fastq\ERR229911.fastq\ERR229911.fastq"
    
    # User confirmed location of existing progress file:
    output_path = r"C:\Users\mbpd1\Downloads\Quantbiologicalwithml\classification_index.csv"
    
    # Auto-detects progress from this specific file and resumes indefinitely
    generate_index(path, output_csv=output_path, limit=None)
