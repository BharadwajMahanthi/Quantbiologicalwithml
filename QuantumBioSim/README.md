# QuantumBioSim

A quantum-inspired biological simulation framework involving synthetic data generation, population dynamics, and genetic algorithms.

## Project Structure

- `src/models.py`: Contains the `QuantumBiologicalSystem` class which models species evolution using a Hamiltonian operator and Schrödinger equation alongside classical biological terms (replication, decay, radiation).
- `src/data_generation.py`: Generates synthetic biological datasets (e.g., Radiotrophic Fungi) using `Faker`.
- `src/simulation.py`: Implements Genetic Algorithms using `DEAP` to optimize species parameters.
- `src/analysis.py`: Tools for time-series analysis and plotting.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Data

Generate a synthetic dataset of mushrooms in radioactive environments:

```bash
python main.py --generate-data
```

### Run Simulation

Run the genetic algorithm optimization to find the fittest ecosystem parameters:

```bash
python main.py --run-sim
```

## Output

## Scientific Consensus & DNA Availability

**Disclaimer on "Big Bang" DNA**:
DNA and RNA are biological molecules that emerged only after the formation of Earth (approx. 4 billion years ago). They did NOT exist at the Big Bang. Furthermore, they degrade over time.

- **Oldest Recoverable DNA**: ~2 million years (Greenland eDNA, _Nature_ 2022).
- **Oldest Recoverable RNA**: ~39,000 years (Woolly Mammoth, _Cell_ 2025).

In this simulation, "Ancestral DNA" refers to **theoretically reconstructed sequences** (e.g., LUCA, Prototaxites) based on phylogenetics, not physical samples from the Hadean Eon.

### References

1.  **Kjær, K.H., et al.** "A 2-million-year-old ecosystem in Greenland uncovered by environmental DNA." _Nature_ 612, 283–291 (2022). [DOI: 10.1038/s41586-022-05453-y](https://www.nature.com/articles/s41586-022-05453-y)
2.  **Smith, O., et al.** "Ancient RNA expression profiles from the extinct woolly mammoth." _Cell_ (2025).
3.  **Dadachova, E., et al.** "Ionizing Radiation Changes the Electronic Properties of Melanin and Enhances the Growth of Melanized Fungi." _PLoS ONE_ 2(5): e457 (2007).
4.  **Prüfer, K., et al.** "The complete genome sequence of a Neanderthal from the Altai Mountains." _Nature_ 505, 43–49 (2014).
    - **Accession**: ERP002097 (Project), Run **ERR229911**.
    - **Significance**: Reference for complex eukaryotic biological systems (TATA-box regulation).
