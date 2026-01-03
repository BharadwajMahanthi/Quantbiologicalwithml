# Extracted content from life_equ v.2.ipynb

## Cell 1 (Code)

```python
import numpy as np
import random
import datetime
import numpy as np
from faker import Faker
import csv
import pandas as pd

fake = Faker()

NUM_RECORDS = 1000000  # Adjust to create a dataset of around 1GB
DOSE_CONVERSION_COEFFICIENT = 1.3e-2  # µSv/Bq for 137Cs
ANNUAL_INTAKE = 130150  # kg/year

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
    "Cryptococcus neoformans": {"Common Name": "Cryptococcus", "Location": "Worldwide", "Min": 0.4, "Max": 24.0, "Edibility": "Unknown", "Melanin": True},
    "Psathyrella candolleana": {"Common Name": "Pale Brittlestem", "Median": 320, "Min": 200, "Max": 500, "Location": "Near Zalizia, Ukraine", "Edibility": "Edible", "Melanin": False},
    "Inocybe geophylla": {"Common Name": "Earthy Inocybe", "Median": 270, "Min": 180, "Max": 380, "Location": "EcoCenter, Ukraine", "Edibility": "Edible", "Melanin": False},
    "Coprinellus micaceus": {"Common Name": "Mica Cap", "Median": 350, "Min": 250, "Max": 450, "Location": "Vesniane, Ukraine", "Edibility": "Edible", "Melanin": False},
    "Lactarius deliciosus": {"Common Name": "Saffron Milk Cap", "Median": 260, "Min": 160, "Max": 400, "Location": "Red Forest, Ukraine", "Edibility": "Edible", "Melanin": False},
    "Hygrophorus russula": {"Common Name": "Russula Waxcap", "Median": 280, "Min": 170, "Max": 420, "Location": "Hilton, Ukraine", "Edibility": "Edible", "Melanin": False},
    "Cortinarius praestans": {"Common Name": "Radiation Webcap", "Median": 480, "Min": 300, "Max": 650, "Location": "Chernobyl Village", "Edibility": "Edible", "Melanin": False},
    "Mycena flavoalba": {"Common Name": "Yellowleg Bonnet", "Median": 330, "Min": 220, "Max": 500, "Location": "Vesniane, Ukraine", "Edibility": "Edible", "Melanin": False},
    "Hebeloma crustuliniforme": {"Common Name": "Poison Pie", "Median": 400, "Min": 250, "Max": 600, "Location": "Ivankov, Ukraine", "Edibility": "Toxic", "Melanin": False},
    "Marasmius oreades": {"Common Name": "Fairy Ring Champignon", "Median": 310, "Min": 200, "Max": 480, "Location": "Voronkov, Ukraine", "Edibility": "Edible", "Melanin": False},
    "Melanoleuca melaleuca": {"Common Name": "Common Cavalier", "Median": 290, "Min": 180, "Max": 420, "Location": "Near Chernobyl, Ukraine", "Edibility": "Edible", "Melanin": False},
    "Pholiota squarrosa": {"Common Name": "Shaggy Scalycap", "Median": 300, "Min": 190, "Max": 480, "Location": "Budaörs, Hungary", "Edibility": "Edible", "Melanin": False},
    "Coprinopsis atramentaria": {"Common Name": "Inky Cap", "Median": 330, "Min": 200, "Max": 500, "Location": "Bingen-am-Rhein, Germany", "Edibility": "Edible", "Melanin": False}
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
    {"Strain ID": "1161", "Radiation (µSv/hr)": 0.350, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 8, "Total Number of Flowering Plants": 25, "Percentage of Diseased Plants": 32.0, "Edibility": "Edible"},
    {"Strain ID": "1101-1", "Radiation (µSv/hr)": 0.370, "Groups of Field Radiation Levels": "Low", "No. of Diseased Plants": 10, "Total Number of Flowering Plants": 28, "Percentage of Diseased Plants": 35.7, "Edibility": "Not Edible"},
    {"Strain ID": "1116-1", "Radiation (µSv/hr)": 0.440, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 8, "Total Number of Flowering Plants": 29, "Percentage of Diseased Plants": 27.6, "Edibility": "Edible"},
    {"Strain ID": "1116-2", "Radiation (µSv/hr)": 0.580, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 5, "Total Number of Flowering Plants": 27, "Percentage of Diseased Plants": 18.5, "Edibility": "Not Edible"},
    {"Strain ID": "1116-3", "Radiation (µSv/hr)": 0.710, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 6, "Total Number of Flowering Plants": 25, "Percentage of Diseased Plants": 24.0, "Edibility": "Edible"},
    {"Strain ID": "1126-1", "Radiation (µSv/hr)": 0.980, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 9, "Total Number of Flowering Plants": 27, "Percentage of Diseased Plants": 33.3, "Edibility": "Not Edible"},
    {"Strain ID": "1126-2", "Radiation (µSv/hr)": 1.340, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 7, "Total Number of Flowering Plants": 28, "Percentage of Diseased Plants": 25.0, "Edibility": "Edible"},
    {"Strain ID": "1111-1", "Radiation (µSv/hr)": 1.420, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 4, "Total Number of Flowering Plants": 27, "Percentage of Diseased Plants": 14.8, "Edibility": "Not Edible"},
    {"Strain ID": "1111-2", "Radiation (µSv/hr)": 1.650, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 8, "Total Number of Flowering Plants": 25, "Percentage of Diseased Plants": 32.0, "Edibility": "Edible"},
    {"Strain ID": "1146-1", "Radiation (µSv/hr)": 1.700, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 8, "Total Number of Flowering Plants": 26, "Percentage of Diseased Plants": 30.8, "Edibility": "Not Edible"},
    {"Strain ID": "1146-2", "Radiation (µSv/hr)": 1.830, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 7, "Total Number of Flowering Plants": 28, "Percentage of Diseased Plants": 25.0, "Edibility": "Edible"},
    {"Strain ID": "1127-1", "Radiation (µSv/hr)": 2.140, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 5, "Total Number of Flowering Plants": 27, "Percentage of Diseased Plants": 18.5, "Edibility": "Not Edible"},
    {"Strain ID": "1127-2", "Radiation (µSv/hr)": 2.570, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 6, "Total Number of Flowering Plants": 25, "Percentage of Diseased Plants": 24.0, "Edibility": "Edible"},
    {"Strain ID": "1127-3", "Radiation (µSv/hr)": 3.120, "Groups of Field Radiation Levels": "Moderate", "No. of Diseased Plants": 9, "Total Number of Flowering Plants": 26, "Percentage of Diseased Plants": 34.6, "Edibility": "Not Edible"},
    {"Strain ID": "1130", "Radiation (µSv/hr)": 3.400, "Groups of Field Radiation Levels": "High", "No. of Diseased Plants": 7, "Total Number of Flowering Plants": 29, "Percentage of Diseased Plants": 24.1, "Edibility": "Edible"},
    {"Strain ID": "1131", "Radiation (µSv/hr)": 3.800, "Groups of Field Radiation Levels": "High", "No. of Diseased Plants": 8, "Total Number of Flowering Plants": 27, "Percentage of Diseased Plants": 29.6, "Edibility": "Not Edible"},
    {"Strain ID": "1132-1", "Radiation (µSv/hr)": 4.040, "Groups of Field Radiation Levels": "High", "No. of Diseased Plants": 7, "Total Number of Flowering Plants": 28, "Percentage of Diseased Plants": 25.0, "Edibility": "Edible"},
    {"Strain ID": "1132-2", "Radiation (µSv/hr)": 4.350, "Groups of Field Radiation Levels": "High", "No. of Diseased Plants": 6, "Total Number of Flowering Plants": 27, "Percentage of Diseased Plants": 22.2, "Edibility": "Not Edible"},
    {"Strain ID": "1132-3", "Radiation (µSv/hr)": 4.670, "Groups of Field Radiation Levels": "High", "No. of Diseased Plants": 8, "Total Number of Flowering Plants": 25, "Percentage of Diseased Plants": 32.0, "Edibility": "Edible"},
    {"Strain ID": "1134", "Radiation (µSv/hr)": 5.000, "Groups of Field Radiation Levels": "High", "No. of Diseased Plants": 9, "Total Number of Flowering Plants": 26, "Percentage of Diseased Plants": 34.6, "Edibility": "Not Edible"},
]



radiation_data = [
    {"Strain ID": "1142", "Radiation at Origin": 0, "log (Radiation at Origin + 0.001)": -3.00, "Mean Log (Number of Viable Cells) Control ± SD": "5.45 ± 0.08", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-0.93 ± 0.08", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.87 ± 0.06"},
    {"Strain ID": "1192", "Radiation at Origin": 0, "log (Radiation at Origin + 0.001)": -3.00, "Mean Log (Number of Viable Cells) Control ± SD": "5.48 ± 0.05", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-0.82 ± 0.08", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.03 ± 0.11"},
    {"Strain ID": "1164", "Radiation at Origin": 0.03, "log (Radiation at Origin + 0.001)": -1.51, "Mean Log (Number of Viable Cells) Control ± SD": "5.28 ± 0.13", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.07 ± 0.22", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.62 ± 0.12"},
    {"Strain ID": "1102-1", "Radiation at Origin": 0.20, "log (Radiation at Origin + 0.001)": -0.71, "Mean Log (Number of Viable Cells) Control ± SD": "5.50 ± 0.08", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.08 ± 0.08", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.89 ± 0.09"},
    {"Strain ID": "1101-3", "Radiation at Origin": 0.24, "log (Radiation at Origin + 0.001)": -0.63, "Mean Log (Number of Viable Cells) Control ± SD": "5.58 ± 0.12", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-0.92 ± 0.13", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.71 ± 0.14"},
    {"Strain ID": "1101-2", "Radiation at Origin": 0.24, "log (Radiation at Origin + 0.001)": -0.63, "Mean Log (Number of Viable Cells) Control ± SD": "5.55 ± 0.10", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-0.99 ± 0.18", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.89 ± 0.10"},
    {"Strain ID": "1161", "Radiation at Origin": 0.35, "log (Radiation at Origin + 0.001)": -0.46, "Mean Log (Number of Viable Cells) Control ± SD": "5.45 ± 0.09", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-0.95 ± 0.10", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.76 ± 0.11"},
    {"Strain ID": "1101-1", "Radiation at Origin": 1.23, "log (Radiation at Origin + 0.001)": 0.09, "Mean Log (Number of Viable Cells) Control ± SD": "5.38 ± 0.11", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.19 ± 0.15", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.71 ± 0.13"},
    {"Strain ID": "1103-3", "Radiation at Origin": 3.56, "log (Radiation at Origin + 0.001)": 0.55, "Mean Log (Number of Viable Cells) Control ± SD": "5.26 ± 0.14", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.02 ± 0.18", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.69 ± 0.16"},
    {"Strain ID": "1162", "Radiation at Origin": 8.35, "log (Radiation at Origin + 0.001)": 0.92, "Mean Log (Number of Viable Cells) Control ± SD": "5.15 ± 0.17", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.17 ± 0.21", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.62 ± 0.19"},
    {"Strain ID": "1105", "Radiation at Origin": 21.03, "log (Radiation at Origin + 0.001)": 1.32, "Mean Log (Number of Viable Cells) Control ± SD": "5.05 ± 0.19", "Mean (Log Number of Viable Cells in 30-min Exposure – Log Number of Cells in Control) ± SD": "-1.30 ± 0.23", "Mean (Log Number of Viable Cells in 60-min Exposure – Log Number of Cells in Control) ± SD": "-3.61 ± 0.20"}
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
```

## Cell 2 (Code)

```python
import pandas as pd

df1 = pd.read_csv('combined_dataset.csv')

df2 = pd.read_csv('combined_dataset2.csv')
```

## Cell 3 (Code)

```python
# Example correlation analysis
import numpy as np

# Extract radiation data from Dataset 1
radiation_data = df1["Radiation (µSv/hr)"].values

# Assume we are analyzing the first time point from Dataset 2
growth_data_species_1 = df2["Species"].values

# Step 1: Check the lengths of both arrays
len_radiation_data = len(radiation_data)
len_growth_data_species_1 = len(growth_data_species_1)

# Step 2: Trim or pad the arrays to make them equal in length
if len_radiation_data > len_growth_data_species_1:
    radiation_data = radiation_data[:len_growth_data_species_1]
elif len_growth_data_species_1 > len_radiation_data:
    growth_data_species_1 = growth_data_species_1[:len_radiation_data]
    
# Calculate correlation
correlation = np.corrcoef(radiation_data, growth_data_species_1[:len(radiation_data)])
print("Correlation between Radiation and Simulated Growth of Species 1:", correlation[0, 1])
```

## Cell 4 (Code)

```python
import matplotlib.pyplot as plt

# Scatter plot to visualize the relationship
plt.scatter(radiation_data, growth_data_species_1, alpha=0.5)
plt.title('Radiation vs. Simulated Growth of Species 1')
plt.xlabel('Radiation Data')
plt.ylabel('Simulated Growth Data (Species 1)')
plt.show()
```

## Cell 5 (Code)

```python
# Assuming df2 is the DataFrame containing the growth data for multiple species
species_columns = df2.columns[1:]  # Assuming the first column is an index or identifier

# Dictionary to store correlation values
correlation_results = {}

# Calculate and store correlation for each species
for species in species_columns:
    growth_data = df2[species].values[:len(radiation_data)]
    correlation = np.corrcoef(radiation_data, growth_data)[0, 1]
    correlation_results[species] = correlation

# Print the correlation results
for species, correlation in correlation_results.items():
    print(f"Correlation between Radiation and Simulated Growth of {species}: {correlation}")
```

## Cell 6 (Code)

```python
from scipy.stats import spearmanr

# Spearman rank correlation for Species 1
spearman_corr, _ = spearmanr(radiation_data, growth_data_species_1)
print(f"Spearman correlation between Radiation and Simulated Growth of Species 1: {spearman_corr}")
```

## Cell 7 (Code)

```python
from scipy.stats import ttest_ind

# Example: Compare the mean viable cell counts between Dataset 1 and simulated data
# Assuming we have viable cell count data for a similar time period in Dataset 1
viable_cells_real = df1["Mean Log (Number of Viable Cells) Control ± SD"].apply(lambda x: float(x.split("±")[0])).values
viable_cells_simulated = df2["Species_1"][:len(viable_cells_real)].values

# Perform t-test
t_stat, p_val = ttest_ind(viable_cells_real, viable_cells_simulated)
print("T-statistic:", t_stat)
print("P-value:", p_val)
```

## Cell 8 (Code)

```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Assuming df2 is loaded from a CSV file or another source
# Replace with your actual data loading mechanism
# df2 = pd.read_csv('your_data.csv')

# Function to check if an array contains constant values
def has_constant_values(arr):
    return np.all(arr == arr[0])

# Iterate over each column in df2
for col in df2.columns:
    if col == 'Extinct_Species_Index':
        continue  # Skip comparing with itself
    
    # Convert columns to numpy arrays for analysis
    data_index = df2['Extinct_Species_Index'].to_numpy()
    data_col = df2[col].to_numpy()

    # Check if data_col is constant
    if has_constant_values(data_col):
        print(f"Array {col} contains constant values. Cannot compute correlation.")
    else:
        # Calculate Pearson correlation coefficient
        corr_coeff, _ = pearsonr(data_index, data_col)
        print(f"Pearson correlation coefficient between Extinct_Species_Index and {col}: {corr_coeff}")
```

## Cell 9 (Code)

```python
import pandas as pd

# Assuming df1 and df2 are your datasets
# Merge df1 and df2 based on their index
merged_df = pd.merge(df1, df2, left_index=True, right_index=True)

# Now you can analyze merged_df to explore relationships between columns
# For example, calculate correlation
correlation_matrix = merged_df.corr()

# Visualize relationships or perform further analysis
```

## Cell 10 (Code)

```python
from deap import base, creator, tools, algorithms
import numpy as np
import csv
from random import uniform
import datetime
import pandas as pd
from random import randint
from scipy.linalg import expm
from faker import Faker
import logging
import os
import time
test_time_steps = 1000
start_time = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

# Create a Faker instance
fake = Faker()
data = []

class NoRepetitiveLogs(logging.Filter):
    def __init__(self):
        super().__init__()
        self.logged_messages = set()

    def filter(self, record):
        message = record.getMessage()
        if message in self.logged_messages:
            return False
        self.logged_messages.add(message)
        return True

logger = logging.getLogger('quantum_bio_system')
logger.setLevel(logging.INFO)

# Create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

# Add custom filter to logger
no_repetitive_logs_filter = NoRepetitiveLogs()
logger.addFilter(no_repetitive_logs_filter)

# Optionally, you can also redirect logs to a file
file_handler = logging.FileHandler('evolution.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# QuantumBiologicalSystem class definition
class QuantumBiologicalSystem:
    def __init__(self, initial_states, energies, carrying_capacity, species_params,mutation_rate):
        logger.info("Initializing QuantumBiologicalSystem")
        self.states = np.array(initial_states, dtype=np.float64)
        self.energies = np.array(energies, dtype=np.float64)
        self.carrying_capacity = carrying_capacity
        self.species_params = species_params
        self.mutation_rate = mutation_rate
        self.rna_sequences = [self.generate_rna_sequence(species['dna_sequence']) for species in self.species_params]
        self.dna_sequences = [species['dna_sequence'] for species in self.species_params]

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
            psi_t = psi_t.astype(np.float64)
            return psi_t
        except Exception as e:
            logger.error(f"Error solving Schrödinger equation: {e}")
            raise e
    def generate_dna_sequence(self,Length):
        # Example DNA initialization (random sequence of A, T, C, G)
        return ''.join(np.random.choice(['A', 'T', 'C', 'G'], Length))

    def generate_rna_sequence(self, dna_sequence):
      # Example RNA transcription (A->U, T->A, C->G, G->C)
        transcription_map = str.maketrans('ATCG', 'UAGC')  # Example translation map
        return dna_sequence.translate(transcription_map)

    def mutate_dna(self, dna, mutation_rate):
        # Example DNA mutation
        dna_list = list(dna)
        for i in range(len(dna_list)):
            if np.random.rand() < mutation_rate:
                dna_list[i] = np.random.choice(list('ATCG'))
        return ''.join(dna_list)
    def disperse(self):
        logger.info("Calculating dispersal")
        for i in range(len(self.states)):
            if np.random.rand() < self.species_params[i]['dispersal_rate']:
                destination = np.random.choice(len(self.states))
                self.states[i] -= 0.1 * self.states[i]  # Example reduction due to migration
                self.states[destination] += 0.1 * self.states[i]  # Example increase at destination

    def update_species_params(self, index):
        # Example of how DNA/RNA affects species parameters (simplified)
        self.species_params[index]['replication_rate'] *= 1 + (self.rna_sequences[index].count('A') - self.rna_sequences[index].count('G')) * 0.001
        self.species_params[index]['decay_rate'] *= 1 + (self.rna_sequences[index].count('C') - self.rna_sequences[index].count('T')) * 0.001

    def replication_term(self):
        logger.info("Calculating replication term")
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = self.states[i] * self.species_params[i]['replication_rate'] * (1 - self.states[i] / self.carrying_capacity)
        return deltas

    def decay_term(self):
        logger.info("Calculating decay term")
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['decay_rate']
        return deltas

    def lifespan_term(self):
        logger.info("Calculating lifespan term")
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['aging_rate'] * self.species_params[i]['lifespan']
        return deltas

    def anti_life_effects(self):
        logger.info("Calculating anti-life effects")
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['anti_life_effect']
        return deltas

    def radiation_effect(self):
        logger.info("Calculating radiation effect")
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['radiation_effectiveness']
        return deltas

    def compute_same_species_birth(self):
        logger.info("Calculating same species birth")
        deltas = np.zeros_like(self.states)
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                if i != j:
                    deltas[i] += (self.species_params[i]['prob_same_species'] *
                                  self.species_params[i]['birth_rate'] *
                                  self.states[j])
        return deltas

    def compute_diff_species_birth(self):
        logger.info("Calculating different species birth")
        deltas = np.zeros_like(self.states)
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
        deltas = np.zeros_like(self.states)
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
    
    start_time = datetime.datetime(year=4000, month=1, day=1)
    def initialize_population(self, start_time, num_species):
        logger.info("Initializing population")
        initial_populations = np.random.uniform(low=1, high=100, size=num_species)
        return initial_populations
    

    def solve(self, generations):
        for _ in range(generations):
            delta_replication = self.replication_term()
            if delta_replication is None:
                delta_replication = np.zeros_like(self.states)

            delta_decay = self.decay_term()
            if delta_decay is None:
                delta_decay = np.zeros_like(self.states)

            delta_lifespan = self.lifespan_term()
            if delta_lifespan is None:
                delta_lifespan = np.zeros_like(self.states)

            delta_antilife = self.anti_life_effects()
            if delta_antilife is None:
                delta_antilife = np.zeros_like(self.states)

            delta_radiation = self.radiation_effect()
            if delta_radiation is None:
                delta_radiation = np.zeros_like(self.states)

            same_species_birth = self.compute_same_species_birth()
            if same_species_birth is None:
                same_species_birth = np.zeros_like(self.states)

            diff_species_birth = self.compute_diff_species_birth()
            if diff_species_birth is None:
                diff_species_birth = np.zeros_like(self.states)

            probability_of_interaction = self.compute_interactions()
            if probability_of_interaction is None:
                probability_of_interaction = np.zeros_like(self.states)

            # Replace extinct species with new ones
            extinct_species = np.where(self.states == 0)[0]
            
            # Replace extinct species with new ones
            self.replace_extinct_species()
            
               # Ensure populations do not become negative
            self.states = np.maximum(0, self.states)
            wavefunction = self.states
            H = self.hamiltonian_operator()

            # Example: Adjust species parameters based on Hamiltonian
            for i in range(len(self.states)):
                self.species_params[i]['competitive_strength'] += H[i, i] * 0.1  # Adjust based on diagonal elements of H
                self.species_params[i]['predation_rate'] += np.sum(H[i, :]) * 0.01  # Adjust based on row-wise sum of H
                self.species_params[i]['environmental_tolerance'] += np.mean(H[:, i]) * 0.005  # Adjust based on column-wise mean of H
                # Additional environmental factors influencing species dynamics
                self.species_params[i]['migration_rate'] += np.max(H[i, :]) * 0.005  # Migration rate
                self.species_params[i]['mutation_rate'] += np.min(H[i, :]) * 0.001  # Mutation rate
                self.species_params[i]['resource_availability'] += np.sum(H[:, i]) * 0.02  # Resource availability
            # Normalize species parameters if necessary
            for params in self.species_params:
                params['competitive_strength'] = max(0, params['competitive_strength'])
                params['predation_rate'] = max(0, params['predation_rate'])
                params['environmental_tolerance'] = max(0, params['environmental_tolerance'])
                params['migration_rate'] = max(0, params['migration_rate'])
                params['mutation_rate'] = max(0, params['mutation_rate'])
                params['resource_availability'] = max(0, params['resource_availability'])

            # Solve Schrödinger equation to determine species evolution
            wavefunction = self.states  # Example wavefunction (state populations)
            psi_t = self.schrodinger_equation(wavefunction, time_step)

            replace_extinct_species = self.replace_extinct_species()

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
                new_species_params['replication_rate'] *= np.random.uniform(0.03, 0.08)
                new_species_params['decay_rate'] *= np.random.uniform(0.005, 0.02)
                new_species_params['aging_rate'] *= np.random.uniform(0.001, 0.005)
                new_species_params['lifespan'] *= np.random.uniform(10, 50)
                new_species_params['anti_life_effect'] *= np.random.uniform(0.001, 0.005)
                new_species_params['historical_radiation_level'] *= np.random.uniform(0.001, 0.01)
                new_species_params['prob_same_species'] *= np.random.uniform(0.01,0.05)
                new_species_params['prob_diff_species'] *= np.random.uniform(0.001, 0.01)
                new_species_params['radiation_effectiveness'] *= np.random.uniform(0.001, 0.01)
                new_species_params['interaction_strength'] = np.random.uniform(-0.01, 0.01, len(self.states))
                new_species_params['competitive_strength'] *= np.random.uniform(0.001, 0.005)
                new_species_params['predation_rate'] *= np.random.uniform(0.001, 0.005)
                new_species_params['birth_rate'] *= np.random.uniform(0.01, 0.1)
                new_species_params['mutation_rate'] *= np.random.uniform(0.001, 0.1)
                new_species_params['migration_rate'] *= np.random.uniform(0.01, 0.1)  # Include migration rate
                new_species_params['resource_availability'] *= np.random.uniform(0.1, 1.0)
                new_species_params['dispersal_rate'] *= np.random.uniform(0.001, 0.01)
                new_species_params['environmental_tolerance'] *= np.random.uniform(0.5, 1.0)
                new_species_params['carrying_capacity'] *= np.random.uniform(50, 200)
                new_species_params['optimal_temperature'] *= np.random.uniform(15, 35)
                new_species_params['optimal_humidity'] *= np.random.uniform(40, 80)
                new_species_params['optimal_ph'] *= np.random.uniform(5.5, 7.5)
                new_species_params['Melanin'] = np.random.choice([True, False])  # Random boolean value for Melanin
                new_species_params['melanin_present'] = fake.boolean()  # Using Faker for melanin presence
                new_species_params['dna_sequence'] = self.mutate_dna(self.generate_dna_sequence(1000),self.mutation_rate)
                new_species_params['rna_sequence'] = self.generate_rna_sequence(new_species_params['dna_sequence'])


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
                "historical_radiation_level": self.species_params[i]['historical_radiation_level'],
                "prob_same_species": self.species_params[i]['prob_same_species'],
                "prob_diff_species": self.species_params[i]['prob_diff_species'],
                "radiation_effectiveness": self.species_params[i]['radiation_effectiveness'],
                "interaction_strength": self.species_params[i]['interaction_strength'],
                "competitive_strength": self.species_params[i]['competitive_strength'],
                "predation_rate": self.species_params[i]['predation_rate'],
                "birth_rate": self.species_params[i]['birth_rate'],
                "mutation_rate": self.species_params[i]['mutation_rate'],
                "migration_rate": self.species_params[i]['migration_rate'],
                "resource_availability": self.species_params[i]['resource_availability'],
                "dispersal_rate": self.species_params[i]['dispersal_rate'],
                "environmental_tolerance": self.species_params[i]['environmental_tolerance'],
                "carrying_capacity": self.species_params[i]['carrying_capacity'],
                "optimal_temperature": self.species_params[i]['optimal_temperature'],
                "optimal_humidity": self.species_params[i]['optimal_humidity'],
                "optimal_ph": self.species_params[i]['optimal_ph'],
                "Melanin": self.species_params[i]['Melanin'],
                "Population": self.states[i],
                "melanin_present": self.species_params[i]['melanin_present'],
                "dna_sequence":self.species_params[i]['dna_sequence'],
                "rna_sequence":self.species_params[i]['rna_sequence']
            }
            combined_record.update(self.species_params[i])
            data.append(combined_record)
        return data

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
            new_species_params['replication_rate'] *= np.random.uniform(0.03, 0.08)
            new_species_params['decay_rate'] *= np.random.uniform(0.005, 0.02)
            new_species_params['aging_rate'] *= np.random.uniform(0.001, 0.005)
            new_species_params['lifespan'] *= np.random.uniform(10, 50)
            new_species_params['anti_life_effect'] *= np.random.uniform(0.001, 0.005)
            new_species_params['historical_radiation_level'] *= np.random.uniform(0.001, 0.01)
            new_species_params['prob_same_species'] *= np.random.uniform(0.01,0.05)
            new_species_params['prob_diff_species'] *= np.random.uniform(0.001, 0.01)
            new_species_params['radiation_effectiveness'] *= np.random.uniform(0.001, 0.01)
            new_species_params['interaction_strength'] = np.random.uniform(-0.01, 0.01, len(self.states))
            new_species_params['competitive_strength'] *= np.random.uniform(0.001, 0.005)
            new_species_params['predation_rate'] *= np.random.uniform(0.001, 0.005)
            new_species_params['birth_rate'] *= np.random.uniform(0.01, 0.1)
            new_species_params['mutation_rate'] *= np.random.uniform(0.001, 0.1)
            new_species_params['migration_rate'] *= np.random.uniform(0.01, 0.1)  # Include migration rate
            new_species_params['resource_availability'] *= np.random.uniform(0.1, 1.0)
            new_species_params['dispersal_rate'] *= np.random.uniform(0.001, 0.01)
            new_species_params['environmental_tolerance'] *= np.random.uniform(0.5, 1.0)
            new_species_params['carrying_capacity'] *= np.random.uniform(50, 200)
            new_species_params['optimal_temperature'] *= np.random.uniform(15, 35)
            new_species_params['optimal_humidity'] *= np.random.uniform(40, 80)
            new_species_params['optimal_ph'] *= np.random.uniform(5.5, 7.5)
            new_species_params['Melanin'] = np.random.choice([True, False])  # Random boolean value for Melanin
            new_species_params['melanin_present'] = fake.boolean()  # Using Faker for melanin presence
            new_species_params['dna_sequence'] = self.mutate_dna(self.generate_dna_sequence(1000),self.mutation_rate)
            new_species_params['rna_sequence'] = self.generate_rna_sequence(new_species_params['dna_sequence'])

                        # Initialize the new species state
            self.states[extinct_species_index] = np.random.uniform(1, 100)
            # Update the species parameters with the new species
            self.species_params[extinct_species_index] = new_species_params
            logger.info(f"Species {extinct_species_index} went extinct and was replaced by a new species")

        return self.states
    # Parameters
num_species = 50
carrying_capacity = 100
initial_states = np.random.uniform(50, 150, num_species)
energies = np.random.uniform(0.5, 1.5, num_species)
num_time_steps = 3000
time_step = 1.0  # Time step in year
dna_size = 100  # Size of DNA sequence
rna_size = 100  # Size of RNA sequence
mutation_rate = 0.01
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
        'migration_rate': np.random.uniform(0.01, 0.1),  # Include migration rate
        'resource_availability': np.random.uniform(0.1, 1.0),
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

```

## Cell 11 (Code)

```python


data1 = []
quantum_system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params, mutation_rate)


for _ in range(test_time_steps):
    quantum_system.update_states()
end_time = time.time()

elapsed_time = end_time - start_time
estimated_total_time = (elapsed_time / test_time_steps) * num_time_steps

print(f"Time for {test_time_steps} steps: {elapsed_time} seconds")
print(f"Estimated total time for {num_time_steps} steps: {estimated_total_time} seconds ({estimated_total_time / 3600} hours)")

```

## Cell 12 (Code)

```python

final_states = quantum_system.solve(num_time_steps)
for t in range(num_time_steps):
    quantum_system.update_states()
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
                "historical_radiation_level": quantum_system.species_params[i]['historical_radiation_level'],
                "prob_same_species": quantum_system.species_params[i]['prob_same_species'],
                "prob_diff_species": quantum_system.species_params[i]['prob_diff_species'],
                "radiation_effectiveness": quantum_system.species_params[i]['radiation_effectiveness'],
                "interaction_strength": quantum_system.species_params[i]['interaction_strength'],
                "competitive_strength": quantum_system.species_params[i]['competitive_strength'],
                "predation_rate": quantum_system.species_params[i]['predation_rate'],
                "birth_rate": quantum_system.species_params[i]['birth_rate'],
                "mutation_rate": quantum_system.species_params[i]['mutation_rate'],
                "migration_rate": quantum_system.species_params[i]['migration_rate'],
                "resource_availability": quantum_system.species_params[i]['resource_availability'],
                "dispersal_rate": quantum_system.species_params[i]['dispersal_rate'],
                "environmental_tolerance": quantum_system.species_params[i]['environmental_tolerance'],
                "carrying_capacity": quantum_system.species_params[i]['carrying_capacity'],
                "optimal_temperature": quantum_system.species_params[i]['optimal_temperature'],
                "optimal_humidity": quantum_system.species_params[i]['optimal_humidity'],
                "optimal_ph": quantum_system.species_params[i]['optimal_ph'],
                "Melanin": quantum_system.species_params[i]['Melanin'],
                "Population": quantum_system.states[i],
                "melanin_present": quantum_system.species_params[i]['melanin_present'],
                "dna_sequence":quantum_system.species_params[i]['dna_sequence'],
                "rna_sequence":quantum_system.species_params[i]['rna_sequence']
            }
            combined_record.update(quantum_system.species_params[i])
            data1.append(combined_record)
print("Final species counts after optimization:", final_states)
# Convert the data to a pandas DataFrame
df = pd.DataFrame(data1)
# Save to CSV
df.to_csv('combined_data2.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
```

## Cell 13 (Code)

```python
def eval_species(species):
    logger.info("Evaluating species")
    initial_states = species[:num_species]
    energies = species[num_species:2*num_species]
    species_params = []

    param_start = 2 * num_species
    for i in range(num_species):
        params = {
            'replication_rate': species[param_start] if param_start < len(species) else 1.0,  # Default to 1.0 if missing
            'decay_rate': species[param_start + 1] if param_start + 1 < len(species) else 0.1,  # Default to 0.1 if missing
            'aging_rate': species[param_start + 2] if param_start + 2 < len(species) else 0.01,  # Default to 0.01 if missing
            'lifespan': species[param_start + 3] if param_start + 3 < len(species) else 100,  # Default to 100 if missing
            'anti_life_effect': species[param_start + 4] if param_start + 4 < len(species) else 0.0,  # Default to 0.0 if missing
            'historical_radiation_level': species[param_start + 5] if param_start + 5 < len(species) else 0.0,  # Default to 0.0 if missing
            'prob_same_species': species[param_start + 6] if param_start + 6 < len(species) else 0.5,  # Default to 0.5 if missing
            'prob_diff_species': species[param_start + 7] if param_start + 7 < len(species) else 0.5,  # Default to 0.5 if missing
            'radiation_effectiveness': species[param_start + 8] if param_start + 8 < len(species) else 0.0,  # Default to 0.0 if missing
            'interaction_strength': species[param_start + 9:param_start + 9 + num_species] if param_start + 9 + num_species <= len(species) else np.ones(num_species) * 0.1,  # Default to array of 0.1 if missing
            'competitive_strength': species[param_start + 9 + num_species] if param_start + 9 + num_species < len(species) else 0.1,  # Default to 0.1 if missing
            'predation_rate': species[param_start + 10 + num_species] if param_start + 10 + num_species < len(species) else 0.01,  # Default to 0.01 if missing
            'birth_rate': species[param_start + 11 + num_species] if param_start + 11 + num_species < len(species) else 0.1,  # Default to 0.1 if missing
            'mutation_rate': species[param_start + 12 + num_species] if param_start + 12 + num_species < len(species) else 0.01,  # Default to 0.01 if missing
            'migration_rate': species[param_start + 13 + num_species] if param_start + 13 + num_species < len(species) else 0.1,  # Default to 0.1 if missing
            'resource_availability': species[param_start + 14 + num_species] if param_start + 14 + num_species < len(species) else 0.5,  # Default to 0.5 if missing
            'dispersal_rate': species[param_start + 15 + num_species] if param_start + 15 + num_species < len(species) else 0.01,  # Default to 0.01 if missing
            'environmental_tolerance': species[param_start + 16 + num_species] if param_start + 16 + num_species < len(species) else 0.5,  # Default to 0.5 if missing
            'carrying_capacity': species[param_start + 17 + num_species] if param_start + 17 + num_species < len(species) else 100,  # Default to 100 if missing
            'optimal_temperature': species[param_start + 18 + num_species] if param_start + 18 + num_species < len(species) else 25.0,  # Default to 25.0 if missing
            'optimal_humidity': species[param_start + 19 + num_species] if param_start + 19 + num_species < len(species) else 50.0,  # Default to 50.0 if missing
            'optimal_ph': species[param_start + 20 + num_species] if param_start + 20 + num_species < len(species) else 7.0,  # Default to 7.0 if missing
            'Melanin': species[param_start + 21 + num_species] if param_start + 21 + num_species < len(species) else 0.0,  # Default to 0.0 if missing
            'melanin_present': fake.boolean(),  # Using Faker for melanin presence
            'dna_sequence': str(species[param_start + 22 + num_species]) if param_start + 22 + num_species < len(species) else ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=dna_size)),
            'rna_sequence': str(species[param_start + 23 + num_species]) if param_start + 23 + num_species < len(species) else ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=dna_size))
        }
        species_params.append(params)
        param_start += 24 + num_species
    
    mutation_rate = species[param_start + 12 + num_species] if param_start + 12 + num_species < len(species) else 0.01
    system = QuantumBiologicalSystem(initial_states, energies, carrying_capacity, species_params,mutation_rate)
    final_states = system.solve(generations)
    fitness = np.sum(final_states)  # Assuming fitness is the sum of the final populations

    logger.info(f"Fitness evaluated: {fitness}")
    return fitness,
```

## Cell 14 (Code)

```python

# DEAP genetic algorithm setup
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
ind_size = num_species * (2 + 19 + num_species)  # Individual size
toolbox.register("attr_float", np.random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ind_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop_size = 5 * ind_size
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_species)
mutation_rate=0.001
generations = 300  # Adjust as needed based on your time step and total simulation time
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

```

## Cell 15 (Code)

```python

# Best individual parameters extraction
best_params_dict = []
for i in range(num_species):
    start = i * (2 + 19 + num_species)
    end = start + (2 + 19 + num_species)

    if end > len(best_ind):
        raise IndexError(f"Index out of range: Trying to access index {end} in best_ind of length {len(best_ind)}")

    species_params = {
            'replication_rate': best_ind[start] if start < len(best_ind) else 1.0,  # Default to 1.0 if missing
            'decay_rate': best_ind[start + 1] if start + 1 < len(best_ind) else 0.1,  # Default to 0.1 if missing
            'aging_rate': best_ind[start + 2] if start + 2 < len(best_ind) else 0.01,  # Default to 0.01 if missing
            'lifespan': best_ind[start + 3] if start + 3 < len(best_ind) else 100,  # Default to 100 if missing
            'anti_life_effect': best_ind[start + 4] if start + 4 < len(best_ind) else 0.0,  # Default to 0.0 if missing
            'historical_radiation_level': best_ind[start + 5] if start + 5 < len(best_ind) else 0.0,  # Default to 0.0 if missing
            'prob_same_species': best_ind[start + 6] if start + 6 < len(best_ind) else 0.5,  # Default to 0.5 if missing
            'prob_diff_species': best_ind[start + 7] if start + 7 < len(best_ind) else 0.5,  # Default to 0.5 if missing
            'radiation_effectiveness': best_ind[start + 8] if start + 8 < len(best_ind) else 0.0,  # Default to 0.0 if missing
            'interaction_strength': best_ind[start + 9:start + 9 + num_species] if start + 9 + num_species <= len(best_ind) else np.ones(num_species) * 0.1,  # Default to array of 0.1 if missing
            'competitive_strength': best_ind[start + 9 + num_species] if start + 9 + num_species < len(best_ind) else 0.1,  # Default to 0.1 if missing
            'predation_rate': best_ind[start + 10 + num_species] if start + 10 + num_species < len(best_ind) else 0.01,  # Default to 0.01 if missing
            'birth_rate': best_ind[start + 11 + num_species] if start + 11 + num_species < len(best_ind) else 0.1,  # Default to 0.1 if missing
            'mutation_rate': best_ind[start + 12 + num_species] if start + 12 + num_species < len(best_ind) else 0.01,  # Default to 0.01 if missing
            'migration_rate': best_ind[start + 13 + num_species] if start + 13 + num_species < len(best_ind) else 0.1,  # Default to 0.1 if missing
            'resource_availability': best_ind[start + 14 + num_species] if start + 14 + num_species < len(best_ind) else 0.5,  # Default to 0.5 if missing
            'dispersal_rate': best_ind[start + 15 + num_species] if start + 15 + num_species < len(best_ind) else 0.01,  # Default to 0.01 if missing
            'environmental_tolerance': best_ind[start + 16 + num_species] if start + 16 + num_species < len(best_ind) else 0.5,  # Default to 0.5 if missing
            'carrying_capacity': best_ind[start + 17 + num_species] if start + 17 + num_species < len(best_ind) else 100,  # Default to 100 if missing
            'optimal_temperature': best_ind[start + 18 + num_species] if start + 18 + num_species < len(best_ind) else 25.0,  # Default to 25.0 if missing
            'optimal_humidity': best_ind[start + 19 + num_species] if start + 19 + num_species < len(best_ind) else 50.0,  # Default to 50.0 if missing
            'optimal_ph': best_ind[start + 20 + num_species] if start + 20 + num_species < len(best_ind) else 7.0,  # Default to 7.0 if missing
            'Melanin': best_ind[start + 21 + num_species] if start + 21 + num_species < len(best_ind) else 0.0,  # Default to 0.0 if missing
            'melanin_present': fake.boolean(),  # Using Faker for melanin presence
            'dna_sequence': str(best_ind[start + 22 + num_species]) if start + 22 + num_species < len(best_ind) else ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=dna_size)),
            'rna_sequence': str(best_ind[start + 23 + num_species]) if start + 23 + num_species < len(best_ind) else ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=dna_size))
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
                    "historical_radiation_level": quantum_system.species_params[i]['historical_radiation_level'],
                    "prob_same_species": quantum_system.species_params[i]['prob_same_species'],
                    "prob_diff_species": quantum_system.species_params[i]['prob_diff_species'],
                    "radiation_effectiveness": quantum_system.species_params[i]['radiation_effectiveness'],
                    "interaction_strength": quantum_system.species_params[i]['interaction_strength'],
                    "competitive_strength": quantum_system.species_params[i]['competitive_strength'],
                    "predation_rate": quantum_system.species_params[i]['predation_rate'],
                    "birth_rate": quantum_system.species_params[i]['birth_rate'],
                    "mutation_rate": quantum_system.species_params[i]['mutation_rate'],
                    "migration_rate": quantum_system.species_params[i]['migration_rate'],
                    "resource_availability": quantum_system.species_params[i]['resource_availability'],
                    "dispersal_rate": quantum_system.species_params[i]['dispersal_rate'],
                    "environmental_tolerance": quantum_system.species_params[i]['environmental_tolerance'],
                    "carrying_capacity": quantum_system.species_params[i]['carrying_capacity'],
                    "optimal_temperature": quantum_system.species_params[i]['optimal_temperature'],
                    "optimal_humidity": quantum_system.species_params[i]['optimal_humidity'],
                    "optimal_ph": quantum_system.species_params[i]['optimal_ph'],
                    "Melanin": quantum_system.species_params[i]['Melanin'],
                    "Population": quantum_system.states[i],
                    "melanin_present": quantum_system.species_params[i]['melanin_present'],
                    "dna_sequence":quantum_system.species_params[i]['dna_sequence'],
                    "rna_sequence":quantum_system.species_params[i]['rna_sequence']
            }
            combined_record.update(quantum_system.species_params[i])
            data.append(combined_record)
            
final_states = quantum_system.states
print("Final species counts after optimization:", final_states)
```

## Cell 16 (Code)

```python

# Export data to CSV
output_file = 'species_population_data_optimized.csv'
df = pd.DataFrame(data)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = f"evolution_results_{timestamp}.csv"
df.to_csv(output_file, index=False)
logger.info(f"Results saved to {output_file}")

print(f"Optimized data has been written to {output_file}")
```

## Cell 17 (Code)

```python
import pandas as pd
import numpy as np
from faker import Faker

# Load your original dataset
original_df = pd.read_csv('combined_data2.csv')

# Number of times to extend the dataset
extension_factor = 100

# Initialize Faker
fake = Faker()

# Function to generate a new row
def generate_new_row(row):
    new_row = row.copy()
    new_row['Species Name'] = f"Species_{np.random.randint(0, len(original_df))}"
    new_row['Population'] = np.random.uniform(1, 100)
    new_row['replication_rate'] = np.random.uniform(0.03, 0.08)
    new_row['decay_rate'] = np.random.uniform(0.005, 0.02)
    new_row['aging_rate'] = np.random.uniform(0.001, 0.005)
    new_row['lifespan'] = np.random.randint(10, 50)
    new_row['anti_life_effect'] = np.random.uniform(0.001, 0.005)
    new_row['radiation_effectiveness'] = np.random.uniform(0.001, 0.01)
    new_row['prob_same_species'] = np.random.uniform(0.01, 0.05)
    new_row['prob_diff_species'] = np.random.uniform(0.001, 0.01)
    new_row['interaction_strength'] = np.random.uniform(-0.01, 0.01, len(original_df))
    new_row['competitive_strength'] = np.random.uniform(0.001, 0.005)
    new_row['predation_rate'] = np.random.uniform(0.001, 0.005)
    new_row['birth_rate'] = np.random.uniform(0.01, 0.1)
    new_row['mutation_rate'] = np.random.uniform(0.001, 0.01)
    new_row['dispersal_rate'] = np.random.uniform(0.001, 0.01)
    new_row['environmental_tolerance'] = np.random.uniform(0.5, 1.0)
    new_row['carrying_capacity'] = np.random.uniform(50, 200)
    new_row['optimal_temperature'] = np.random.uniform(15, 35)
    new_row['optimal_humidity'] = np.random.uniform(40, 80)
    new_row['optimal_ph'] = np.random.uniform(5.5, 7.5)
    new_row['Melanin'] = np.random.choice([True, False])
    new_row['melanin_present'] = fake.boolean()
    return new_row

# Generate the extended dataset
extended_data = []
for i in range(extension_factor):
    for _, row in original_df.iterrows():
        extended_data.append(generate_new_row(row))

# Convert to DataFrame
extended_df = pd.DataFrame(extended_data)

# Save the extended dataset
extended_df.to_csv('extended_dataset.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
```

## Cell 18 (Code)

```python
merged_df.head()
```

