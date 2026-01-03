import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_population(csv_path):
    print(f"Analyzing Population from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print("Error: File not found.")
        return

    chunk_size = 5_000_000
    counts = {"Human": 0, "Microbial": 0, "Unknown": 0}
    total_processed = 0
    
    # Read in chunks to handle 3.6GB file
    print("Reading CSV in chunks...")
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=["Classification"]):
        # Count values in this chunk
        val_counts = chunk["Classification"].value_counts()
        
        for category, count in val_counts.items():
            if category in counts:
                counts[category] += count
            else:
                counts[category] = count # Should typically be Human/Microbial/Unknown
        
        total_processed += len(chunk)
        print(f"   Processed {total_processed:,} records...")

    print("\n--- FINAL POPULATION CENSUS ---")
    print(f"Total Sequences: {total_processed:,}")
    for cat, count in counts.items():
        percentage = (count / total_processed * 100) if total_processed > 0 else 0
        print(f"{cat}: {count:,} ({percentage:.2f}%)")

    # Visualization
    print("\nGenerating Population Plot...")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.keys(), counts.values(), color=['#2ca02c', '#d62728', '#7f7f7f'])
    
    plt.title("Ancestral DNA Composition (Altai Neanderthal Sample)", fontsize=14)
    plt.ylabel("Count (Reads)", fontsize=12)
    plt.yscale('log') # Log scale because Unknown >> Human
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')

    output_img = "population_census.png"
    plt.savefig(output_img)
    print(f"Plot saved to: {os.path.abspath(output_img)}")

if __name__ == "__main__":
    # Point to the Parent Directory file as confirmed
    input_csv = r"C:\Users\mbpd1\Downloads\Quantbiologicalwithml\classification_index.csv"
    analyze_population(input_csv)
