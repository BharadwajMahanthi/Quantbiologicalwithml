# Quantbiologicalwithml
Here's the revised README.md with your specifications:

---

# Quantum Biological System Simulation

## Project Overview

This repository contains a simulation framework for studying quantum biological systems, specifically focusing on the effects of radiation on various fungal species. The project integrates data from different sources to analyze and simulate the growth, decay, and interactions of fungi under various environmental conditions, including radiation.

## Table of Contents

- [Project Overview](#project-overview)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [About](#about)
- [References](#references)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quantum-biological-system-simulation.git
   cd quantum-biological-system-simulation
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**:
   - Ensure you have the necessary data files for `df1` and `df2`.
   - Place the data files in the appropriate directory or update the script paths accordingly.

2. **Running the Simulation**:
   - Execute the .ipynb notebook files using jupyter or online google colab:

3. **Analyzing Results**:
   - Use the provided Jupyter Notebooks or scripts and modify the parapamters as you see to analyze the simulation results.

## Data

The datasets used in this project can be found on Kaggle:
- [Quantum Biological Datasets](https://www.kaggle.com/datasets/mbpd1999/quantbilogocial?select=combined_data2.csv)

### df1: Mushroom and Fungi Data
- Contains data on various mushroom species and their environmental factors.
- Columns include: `Strain ID`, `Radiation (µSv/hr)`, `Groups of Field Radiation Levels`, `No. of Diseased Plants`, `Total Number of Flowering Plants`, `Percentage of Diseased Plants`, `Edibility`, `Radiation at Origin`, `log (Radiation at Origin + 0.001)`, `Mean Log (Number of Viable Cells) Control ± SD`, `Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD`, `Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD`, `Location`, `GPS Coordinates`, `Date of Collection`, `Viable Cells After 30 min (%)`, `Viable Cells After 60 min (%)`, `Species Name`, `Common Name`, `Latitude`, `Longitude`, `Habitat`, `Soil Type`, `pH Level`, `Contamination Zone`, `Radioactive Contamination Level (kBq/m²)`, `Radioactive Isotope`, `Concentration in Mushroom (Bq/kg)`, `Committed Effective Dose (mSv/year)`, `Cap Diameter (cm)`, `Stem Length (cm)`, `Melanin`, `Fungi Growth Rate`, `Previous Decay Rate`, `Present Decay Rate`, `Years Since 2024`, `Estimated Birth Date`, `Estimated Death Date`, `Previous Decaying Rate`, `Present Decaying Rate`, `Life and Death Cycles`.

### df2: Species Growth and Radiation Data
- Contains data on how different species grow under radiation.
- Columns include: `Species Name`, `replication_rate`, `decay_rate`, `aging_rate`, `lifespan`, `anti_life_effect`, `radiation_effectiveness`, `prob_same_species`, `prob_diff_species`, `interaction_strength`, `competitive_strength`, `predation_rate`, `birth_rate`, `mutation_rate`, `dispersal_rate`, `environmental_tolerance`, `carrying_capacity`, `optimal_temperature`, `optimal_humidity`, `optimal_ph`, `Melanin`, `Population`, `melanin_present`, `dna_sequence`, `rna_sequence`.

## Features

- **Quantum Biological System Class**:
  - Initialization and configuration of the system.
  - Calculation of biological terms such as replication, decay, lifespan, anti-life effects, and radiation effects.
  - Solution of the Schrödinger equation using the Hamiltonian operator.
  - Integration with DEAP library for genetic algorithms.

- **Data Handling**:
  - Extensive logging for monitoring and error handling.
  - Data recording using Pandas DataFrames.
  - Handling missing values, data cleaning, and transformation.

- **Statistical Analysis and Visualization**:
  - Analysis of species growth under different radiation levels.
  - Visualization of data using Matplotlib.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements, bug fixes, or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


For any questions or further information, please contact the project maintainer at [mbpd.1999l@gmail.com].

## References

This project utilizes data and methodologies from the following research papers:

1. [Development of CAPS Markers for Evaluation of Genetic Diversity and Population Structure in the Germplasm of Button Mushroom (Agaricus bisporus)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8151297/) - Hyejin An, Hwa-Yong Lee, Donghwan Shim, Seong Ho Choi, Hyunwoo Cho, Tae Kyung Hyun, Ick-Hyun Jo, Jong-Wook Chung.
   
2. [Anther‐smut fungi from more contaminated sites in Chernobyl show lower infection ability and lower viability following experimental irradiation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7381591/) - Sylvie Arnaise, Jacqui A. Shykoff, Anders P. Møller, Timothy A. Mousseau, Tatiana Giraud.
