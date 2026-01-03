import numpy as np
import pandas as pd
import random
import datetime
import logging
import csv
from faker import Faker
from scipy.sparse import csr_matrix
from scipy.linalg import expm
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr, ttest_ind
import os
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('quantum_bio_system')
file_handler = logging.FileHandler('evolution.log', mode='w')
file_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Initialize Faker
fake = Faker()

# Constants
DOSE_CONVERSION_COEFFICIENT = 1.3e-2  # µSv/Bq for 137Cs
ANNUAL_INTAKE = 130150  # kg/year
NUM_SPECIES = 24
GENERATIONS = 100
CARRYING_CAPACITY = 1000.0
DNA_SIZE = 100
RNA_SIZE = 100
MUTATION_RATE = 0.001
POP_SIZE = 500
NUM_TIME_STEPS = 100
TIME_STEP = 1.0  # Year
NUM_RECORDS = 10000
TEST_TIME_STEPS = 100

# Fungal species data
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

# Contamination zones
zones = {
    "Zone 1": {"Min Contamination": 1480, "Max Contamination": 5000},
    "Zone 2": {"Min Contamination": 555, "Max Contamination": 1480},
    "Zone 3": {"Min Contamination": 185, "Max Contamination": 555},
    "Zone 4": {"Min Contamination": 37, "Max Contamination": 185}
}

# Mushroom, radiation, and location data
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

# Define species_names
species_names = list(species_data.keys())

# Utility functions
def calculate_fungi_growth_rate(melanin_present, radiation_level):
    if melanin_present:
        return 1.5 * radiation_level
    return 1.0

def simulate_fungi_growth(location, radiation_level, fungi_species):
    if "reactor" in location.lower() or "soil" in location.lower():
        if radiation_level > 0:
            if fungi_species in ["Cladosporium sphaerospermum", "Wangiella dermatitidis", "Cryptococcus neoformans"]:
                return "High (Radiotropism)"
            return "High"
        return "Low"
    return "Normal"

def calculate_previous_decaying_rate(years_since_2024):
    half_life = 4468e6  # Uranium-238 half-life in years
    return 0.5 ** (years_since_2024 / half_life)

def calculate_present_decaying_rate(years_since_2024):
    half_life = 5730  # Carbon-14 half-life in years
    return 0.5 ** (years_since_2024 / half_life)

def estimate_life_and_death_cycles(previous_decaying_rate, present_decaying_rate):
    decay_rate_difference = previous_decaying_rate - present_decaying_rate
    return max(int(decay_rate_difference * 10), 1)

def calculate_committed_effective_dose(concentration):
    return concentration * DOSE_CONVERSION_COEFFICIENT * ANNUAL_INTAKE

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

def random_date(start_date, end_date):
    return start_date + datetime.timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

def estimate_dates(years_since_2024):
    birth_date = random_date(datetime.datetime(2024 - years_since_2024, 1, 1), datetime.datetime(2024, 12, 31))
    death_date = random_date(birth_date, datetime.datetime(2054, 12, 31))
    return birth_date, death_date

# Generate combined dataset
data = []
for _ in range(NUM_RECORDS):
    mushroom_record = random.choice(mushroom_data)
    radiation_record = random.choice(radiation_data)
    location_record = random.choice(location_data)
    combined_record = {**mushroom_record, **radiation_record, **location_record}
    combined_record = calculate_derived_fields(combined_record)
    species = random.choice(list(species_data.keys()))
    zone = random.choice(list(zones.keys()))
    concentration = random.uniform(species_data[species]["Min"], species_data[species]["Max"])
    committed_effective_dose = calculate_committed_effective_dose(concentration)
    melanin_present = species_data[species]["Melanin"]
    previous_decay_rate = np.random.uniform(0.5, 1.5)
    present_decay_rate = np.random.uniform(0.5, 1.5)
    years_since_2024 = np.random.randint(1, 31)
    birth_date, death_date = estimate_dates(years_since_2024)
    growth_rate = calculate_fungi_growth_rate(melanin_present, radiation_record["Radiation at Origin"])
    fungi_growth_rate = simulate_fungi_growth(location_record["Location"], combined_record["Radiation (µSv/hr)"], species)
    collection_date = fake.date_between(start_date='-36y', end_date='today')
    years_since_2024 = (datetime.date.today() - collection_date).days / 365.25
    combined_record.update({
        "Species Name": species,
        "Common Name": species_data[species]["Common Name"],
        "Location": species_data[species]["Location"],
        "Latitude": random.uniform(50.0, 52.0),
        "Longitude": random.uniform(27.0, 30.0),
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
        "Previous Decaying Rate": calculate_previous_decaying_rate(years_since_2024),
        "Present Decaying Rate": calculate_present_decaying_rate(years_since_2024),
        "Life and Death Cycles": estimate_life_and_death_cycles(calculate_previous_decaying_rate(years_since_2024), calculate_present_decaying_rate(years_since_2024))
    })
    data.append(combined_record)

df = pd.DataFrame(data)
df.to_csv('combined_dataset.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
logger.info("Combined dataset saved to combined_dataset.csv")

# Quantum Biological System Class
class QuantumBiologicalSystem:
    def __init__(self, initial_states, energies, carrying_capacity, species_params, mutation_rate):
        self.states = np.array(initial_states, dtype=np.float64)
        self.energies = np.array(energies, dtype=np.float64)
        self.carrying_capacity = carrying_capacity
        self.species_params = species_params
        self.mutation_rate = mutation_rate
        self.dna_sequences = [species['dna_sequence'] for species in species_params]
        self.rna_sequences = [self.generate_rna_sequence(dna) for dna in self.dna_sequences]
        self.extinct_indices = []

    def hamiltonian_operator(self):
        num_species = len(self.states)
        H = np.zeros((num_species, num_species), dtype=np.float64)
        for i in range(num_species):
            H[i, i] = self.energies[i]
            for j in range(i + 1, num_species):
                interaction_strength = self.species_params[i].get('interaction_strength', np.zeros(num_species))
                if isinstance(interaction_strength, (list, np.ndarray)) and j < len(interaction_strength):
                    H[i, j] = H[j, i] = interaction_strength[j]
        return csr_matrix(H)

    def schrodinger_equation(self, wavefunction, time_step):
        try:
            H = self.hamiltonian_operator()
            H = H.toarray().astype(np.complex128)
            exp_H = expm(-1j * H * time_step)
            psi_t = exp_H @ wavefunction
            return psi_t.real
        except Exception as e:
            logger.error(f"Error solving Schrödinger equation: {e}")
            return wavefunction

    def generate_dna_sequence(self, length):
        return ''.join(np.random.choice(['A', 'T', 'C', 'G'], length))

    def generate_rna_sequence(self, dna_sequence):
        transcription_map = str.maketrans('ATCG', 'UAGC')
        return dna_sequence.translate(transcription_map)

    def mutate_dna(self, dna, mutation_rate):
        dna_list = list(dna)
        for i in range(len(dna_list)):
            if np.random.rand() < mutation_rate:
                dna_list[i] = np.random.choice(['A', 'T', 'C', 'G'])
        return ''.join(dna_list)

    def disperse(self):
        for i in range(len(self.states)):
            if np.random.rand() < self.species_params[i]['dispersal_rate']:
                destination = np.random.choice(len(self.states))
                migration = 0.1 * self.states[i]
                self.states[i] -= migration
                self.states[destination] += migration

    def update_species_params(self, index):
        self.species_params[index]['replication_rate'] *= 1 + (self.rna_sequences[index].count('A') - self.rna_sequences[index].count('G')) * 0.001
        self.species_params[index]['decay_rate'] *= 1 + (self.rna_sequences[index].count('C') - self.rna_sequences[index].count('U')) * 0.001

    def replication_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = self.states[i] * self.species_params[i]['replication_rate'] * (1 - self.states[i] / self.carrying_capacity)
        return deltas

    def decay_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.states[i] * self.species_params[i]['decay_rate']
        return deltas

    def lifespan_term(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['aging_rate'] * self.species_params[i]['lifespan']
        return deltas

    def anti_life_effects(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            deltas[i] = -self.species_params[i]['anti_life_effect']
        return deltas

    def radiation_effect(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            try:
                dose = self.species_params[i]['radiation_effectiveness'] * DOSE_CONVERSION_COEFFICIENT
                damage = 0.01 * dose + 0.001 * dose**2
                deltas[i] = -self.states[i] * damage
            except OverflowError:
                deltas[i] = 0
        return deltas

    def compute_same_species_birth(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                if i != j:
                    deltas[i] += (self.species_params[i]['prob_same_species'] * self.species_params[i]['birth_rate'] * self.states[j])
        return deltas

    def compute_diff_species_birth(self):
        deltas = np.zeros(len(self.states))
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                if i != j:
                    deltas[i] += (self.species_params[i]['prob_diff_species'] * self.species_params[i]['birth_rate'] * self.states[j])
        return deltas

    def compute_interactions(self):
        same_species_birth = np.zeros(len(self.states))
        diff_species_birth = np.zeros(len(self.states))
        for i in range(len(self.states)):
            same_species_birth[i] = self.species_params[i]['prob_same_species']
            diff_species_birth[i] = self.species_params[i]['prob_diff_species']
        return same_species_birth, diff_species_birth

    def replace_extinct_species(self):
        extinct_species = np.where(self.states <= 0)[0]
        for i in extinct_species:
            self.extinct_indices.append(i)
            new_params = self.species_params[i].copy()
            new_params.update({
                'replication_rate': np.random.uniform(0.03, 0.08),
                'decay_rate': np.random.uniform(0.005, 0.02),
                'aging_rate': np.random.uniform(0.001, 0.005),
                'lifespan': np.random.uniform(10, 50),
                'anti_life_effect': np.random.uniform(0.001, 0.005),
                'historical_radiation_level': np.random.uniform(0.001, 0.01),
                'prob_same_species': np.random.uniform(0.01, 0.05),
                'prob_diff_species': np.random.uniform(0.001, 0.01),
                'radiation_effectiveness': np.random.uniform(0.001, 0.01),
                'interaction_strength': np.random.uniform(-0.01, 0.01, len(self.states)),
                'competitive_strength': np.random.uniform(0.001, 0.005),
                'predation_rate': np.random.uniform(0.001, 0.005),
                'birth_rate': np.random.uniform(0.01, 0.1),
                'mutation_rate': np.random.uniform(0.001, 0.01),
                'migration_rate': np.random.uniform(0.01, 0.1),
                'resource_availability': np.random.uniform(0.1, 1.0),
                'dispersal_rate': np.random.uniform(0.001, 0.01),
                'environmental_tolerance': np.random.uniform(0.5, 1.0),
                'carrying_capacity': np.random.uniform(50, 200),
                'optimal_temperature': np.random.uniform(15, 35),
                'optimal_humidity': np.random.uniform(40, 80),
                'optimal_ph': np.random.uniform(5.5, 7.5),
                'Melanin': np.random.choice([True, False]),
                'melanin_present': fake.boolean(),
                'dna_sequence': self.mutate_dna(self.generate_dna_sequence(DNA_SIZE), self.mutation_rate),
                'rna_sequence': self.generate_rna_sequence(self.mutate_dna(self.generate_dna_sequence(DNA_SIZE), self.mutation_rate))
            })
            self.species_params[i] = new_params
            self.states[i] = np.random.uniform(1, 100)
            self.dna_sequences[i] = new_params['dna_sequence']
            self.rna_sequences[i] = new_params['rna_sequence']
            logger.info(f"Species {i} extinct, replaced with new population {self.states[i]}")

    def initialize_population(self, start_time, num_species):
        return np.random.uniform(low=1, high=100, size=num_species)

    def solve(self, generations):
        for _ in range(generations):
            delta_replication = self.replication_term() or np.zeros_like(self.states)
            delta_decay = self.decay_term() or np.zeros_like(self.states)
            delta_lifespan = self.lifespan_term() or np.zeros_like(self.states)
            delta_antilife = self.anti_life_effects() or np.zeros_like(self.states)
            delta_radiation = self.radiation_effect() or np.zeros_like(self.states)
            same_species_birth = self.compute_same_species_birth() or np.zeros_like(self.states)
            diff_species_birth = self.compute_diff_species_birth() or np.zeros_like(self.states)
            probability_of_interaction = self.compute_interactions() or (np.zeros_like(self.states), np.zeros_like(self.states))
            self.disperse()
            self.states += (
                delta_replication + delta_decay + delta_lifespan +
                delta_antilife + delta_radiation + same_species_birth +
                diff_species_birth + probability_of_interaction[0] + probability_of_interaction[1]
            )
            np.clip(self.states, 0, self.carrying_capacity, out=self.states)
            self.replace_extinct_species()
            wavefunction = self.states
            H = self.hamiltonian_operator().toarray()
            for i in range(len(self.states)):
                self.species_params[i]['competitive_strength'] += H[i, i] * 0.1
                self.species_params[i]['predation_rate'] += np.sum(H[i, :]) * 0.01
                self.species_params[i]['environmental_tolerance'] += np.mean(H[:, i]) * 0.005
                self.species_params[i]['migration_rate'] += np.max(H[i, :]) * 0.005
                self.species_params[i]['mutation_rate'] += np.min(H[i, :]) * 0.001
                self.species_params[i]['resource_availability'] += np.sum(H[:, i]) * 0.02
                for key in ['competitive_strength', 'predation_rate', 'environmental_tolerance', 'migration_rate', 'mutation_rate', 'resource_availability']:
                    self.species_params[i][key] = max(0, self.species_params[i][key])
            psi_t = self.schrodinger_equation(wavefunction, TIME_STEP)
            self.states = np.clip(psi_t, 0, self.carrying_capacity)
        return self.states

    def update_states(self):
        for _ in range(NUM_TIME_STEPS):
            self.solve(1)
            for i in range(len(self.states)):
                self.update_species_params(i)

    def record_state(self):
        data = []
        for i in range(len(self.states)):
            if self.states[i] > 0:
                record = {
                    "Species Name": f"Species_{i}",
                    "Population": self.states[i],
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
                    "melanin_present": self.species_params[i]['melanin_present'],
                    "dna_sequence": self.dna_sequences[i],
                    "rna_sequence": self.rna_sequences[i],
                    "Extinct_Species_Index": i if i in self.extinct_indices else None
                }
                record.update(self.species_params[i])
                data.append(record)
        return data

# Initialize species parameters
species_params = []
for name in species_names:
    data = species_data[name]
    median = data.get('Median', (data['Min'] + data['Max']) / 2)
    params = {
        'replication_rate': np.random.uniform(0.03, 0.08),
        'decay_rate': np.random.uniform(0.005, 0.02),
        'aging_rate': np.random.uniform(0.001, 0.005),
        'lifespan': np.random.randint(10, 50),
        'anti_life_effect': np.random.uniform(0.001, 0.005),
        'historical_radiation_level': np.random.uniform(0.001, 0.01),
        'prob_same_species': np.random.uniform(0.01, 0.05),
        'prob_diff_species': np.random.uniform(0.001, 0.01),
        'radiation_effectiveness': median / 1000,
        'interaction_strength': np.random.uniform(-0.01, 0.01, NUM_SPECIES),
        'competitive_strength': np.random.uniform(0.001, 0.005),
        'predation_rate': np.random.uniform(0.001, 0.005),
        'birth_rate': np.random.uniform(0.01, 0.1),
        'mutation_rate': np.random.uniform(0.001, 0.01),
        'migration_rate': np.random.uniform(0.01, 0.1),
        'resource_availability': np.random.uniform(0.1, 1.0),
        'dispersal_rate': np.random.uniform(0.001, 0.01),
        'environmental_tolerance': np.random.uniform(0.5, 1.0),
        'carrying_capacity': np.random.uniform(50, 200),
        'optimal_temperature': np.random.uniform(15, 35),
        'optimal_humidity': np.random.uniform(40, 80),
        'optimal_ph': np.random.uniform(5.5, 7.5),
        'contamination_zone': random.randint(0, 3),
        'Melanin': data['Melanin'],
        'melanin_present': fake.boolean(),
        'dna_sequence': ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=DNA_SIZE)),
        'rna_sequence': ''.join(np.random.choice(['A', 'U', 'C', 'G'], size=RNA_SIZE))
    }
    species_params.append(params)

initial_states = np.random.uniform(50, 150, NUM_SPECIES)
energies = np.random.uniform(0.5, 1.5, NUM_SPECIES)
quantum_system = QuantumBiologicalSystem(initial_states, energies, CARRYING_CAPACITY, species_params, MUTATION_RATE)

# Time estimation
start_time = time.time()
for _ in range(TEST_TIME_STEPS):
    quantum_system.update_states()
end_time = time.time()
elapsed_time = end_time - start_time
estimated_total_time = (elapsed_time / TEST_TIME_STEPS) * NUM_TIME_STEPS
print(f"Time for {TEST_TIME_STEPS} steps: {elapsed_time} seconds")
print(f"Estimated total time for {NUM_TIME_STEPS} steps: {estimated_total_time} seconds ({estimated_total_time / 3600} hours)")

# Run simulation
final_states = quantum_system.solve(NUM_TIME_STEPS)
data1 = []
for t in range(NUM_TIME_STEPS):
    quantum_system.update_states()
    data1.extend(quantum_system.record_state())

print("Final species counts:", final_states)
df2 = pd.DataFrame(data1)
df2.to_csv('combined_data2.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

# DEAP Genetic Algorithm
def eval_species(species):
    try:
        initial_states = species[:NUM_SPECIES]
        energies = species[NUM_SPECIES:2*NUM_SPECIES]
        params = []
        param_start = 2 * NUM_SPECIES
        for i in range(NUM_SPECIES):
            param = {
                'replication_rate': species[param_start] if param_start < len(species) else 0.05,
                'decay_rate': species[param_start + 1] if param_start + 1 < len(species) else 0.01,
                'aging_rate': species[param_start + 2] if param_start + 2 < len(species) else 0.002,
                'lifespan': species[param_start + 3] if param_start + 3 < len(species) else 30,
                'anti_life_effect': species[param_start + 4] if param_start + 4 < len(species) else 0.002,
                'historical_radiation_level': species[param_start + 5] if param_start + 5 < len(species) else 0.005,
                'prob_same_species': species[param_start + 6] if param_start + 6 < len(species) else 0.03,
                'prob_diff_species': species[param_start + 7] if param_start + 7 < len(species) else 0.005,
                'radiation_effectiveness': species[param_start + 8] if param_start + 8 < len(species) else 0.005,
                'interaction_strength': np.array(species[param_start + 9:param_start + 9 + NUM_SPECIES]) if param_start + 9 + NUM_SPECIES <= len(species) else np.ones(NUM_SPECIES) * 0.005,
                'competitive_strength': species[param_start + 9 + NUM_SPECIES] if param_start + 9 + NUM_SPECIES < len(species) else 0.003,
                'predation_rate': species[param_start + 10 + NUM_SPECIES] if param_start + 10 + NUM_SPECIES < len(species) else 0.003,
                'birth_rate': species[param_start + 11 + NUM_SPECIES] if param_start + 11 + NUM_SPECIES < len(species) else 0.05,
                'mutation_rate': species[param_start + 12 + NUM_SPECIES] if param_start + 12 + NUM_SPECIES < len(species) else 0.005,
                'migration_rate': species[param_start + 13 + NUM_SPECIES] if param_start + 13 + NUM_SPECIES < len(species) else 0.05,
                'resource_availability': species[param_start + 14 + NUM_SPECIES] if param_start + 14 + NUM_SPECIES < len(species) else 0.5,
                'dispersal_rate': species[param_start + 15 + NUM_SPECIES] if param_start + 15 + NUM_SPECIES < len(species) else 0.005,
                'environmental_tolerance': species[param_start + 16 + NUM_SPECIES] if param_start + 16 + NUM_SPECIES < len(species) else 0.75,
                'carrying_capacity': species[param_start + 17 + NUM_SPECIES] if param_start + 17 + NUM_SPECIES < len(species) else 100,
                'optimal_temperature': species[param_start + 18 + NUM_SPECIES] if param_start + 18 + NUM_SPECIES < len(species) else 25.0,
                'optimal_humidity': species[param_start + 19 + NUM_SPECIES] if param_start + 19 + NUM_SPECIES < len(species) else 60.0,
                'optimal_ph': species[param_start + 20 + NUM_SPECIES] if param_start + 20 + NUM_SPECIES < len(species) else 6.5,
                'Melanin': species[param_start + 21 + NUM_SPECIES] if param_start + 21 + NUM_SPECIES < len(species) else random.choice([True, False]),
                'melanin_present': fake.boolean(),
                'dna_sequence': ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=DNA_SIZE)),
                'rna_sequence': ''.join(np.random.choice(['A', 'U', 'C', 'G'], size=RNA_SIZE))
            }
            params.append(param)
            param_start += 22 + NUM_SPECIES
        system = QuantumBiologicalSystem(initial_states, energies, CARRYING_CAPACITY, params, MUTATION_RATE)
        final_states = system.solve(GENERATIONS)
        fitness = np.sum(final_states) + 0.1 * len(np.where(final_states > 0)[0])
        logger.info(f"Fitness evaluated: {fitness}")
        return (fitness,)
    except Exception as e:
        logger.error(f"Error in eval_species: {e}")
        return (0,)

ind_size = NUM_SPECIES * (2 + 22 + NUM_SPECIES)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ind_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_species)

def main():
    population = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    logger.info(f"Best individual: {hof[0]}")
    best_params = []
    param_start = 2 * NUM_SPECIES
    for i in range(NUM_SPECIES):
        param = {
            'replication_rate': hof[0][param_start],
            'decay_rate': hof[0][param_start + 1],
            'aging_rate': hof[0][param_start + 2],
            'lifespan': hof[0][param_start + 3] * 50,
            'anti_life_effect': hof[0][param_start + 4],
            'historical_radiation_level': hof[0][param_start + 5],
            'prob_same_species': hof[0][param_start + 6],
            'prob_diff_species': hof[0][param_start + 7],
            'radiation_effectiveness': hof[0][param_start + 8],
            'interaction_strength': np.array(hof[0][param_start + 9:param_start + 9 + NUM_SPECIES]),
            'competitive_strength': hof[0][param_start + 9 + NUM_SPECIES],
            'predation_rate': hof[0][param_start + 10 + NUM_SPECIES],
            'birth_rate': hof[0][param_start + 11 + NUM_SPECIES],
            'mutation_rate': hof[0][param_start + 12 + NUM_SPECIES],
            'migration_rate': hof[0][param_start + 13 + NUM_SPECIES],
            'resource_availability': hof[0][param_start + 14 + NUM_SPECIES],
            'dispersal_rate': hof[0][param_start + 15 + NUM_SPECIES],
            'environmental_tolerance': hof[0][param_start + 16 + NUM_SPECIES],
            'carrying_capacity': hof[0][param_start + 17 + NUM_SPECIES],
            'optimal_temperature': hof[0][param_start + 18 + NUM_SPECIES],
            'optimal_humidity': hof[0][param_start + 19 + NUM_SPECIES],
            'optimal_ph': hof[0][param_start + 20 + NUM_SPECIES],
            'Melanin': bool(hof[0][param_start + 21 + NUM_SPECIES]),
            'melanin_present': fake.boolean(),
            'dna_sequence': ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=DNA_SIZE)),
            'rna_sequence': ''.join(np.random.choice(['A', 'U', 'C', 'G'], size=RNA_SIZE))
        }
        best_params.append(param)
        param_start += 22 + NUM_SPECIES
    return population, stats, hof, best_params

if __name__ == "__main__":
    population, stats, hof, best_params = main()
    logger.info("Final population:")
    for ind in population:
        logger.info(ind)
    logger.info("Hall of Fame:")
    logger.info(hof)
    logger.info("Statistics:")
    logger.info(stats)

    # Simulate with optimized parameters
    optimized_system = QuantumBiologicalSystem(initial_states, energies, CARRYING_CAPACITY, best_params, MUTATION_RATE)
    final_states = optimized_system.solve(GENERATIONS)
    optimized_data = optimized_system.record_state()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    optimized_df = pd.DataFrame(optimized_data)
    optimized_df.to_csv(f'species_population_data_optimized_{timestamp}.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
    print("Final species counts after optimization:", final_states)
    logger.info(f"Optimized results saved to species_population_data_optimized_{timestamp}.csv")

    # Statistical Analysis
    df1 = pd.read_csv('combined_dataset.csv')
    df2 = pd.read_csv('combined_data2.csv')
    def has_constant_values(arr):
        return np.all(arr == arr[0])

    radiation_data = df1["Radiation (µSv/hr)"].values
    for col in df2.columns:
        if col.startswith('Species_') and not has_constant_values(df2[col].values):
            growth_data = df2[col].values[:len(radiation_data)]
            corr, _ = pearsonr(radiation_data, growth_data)
            print(f"Pearson correlation between Radiation and {col}: {corr}")
            spearman_corr, _ = spearmanr(radiation_data, growth_data)
            print(f"Spearman correlation between Radiation and {col}: {spearman_corr}")
            plt.scatter(radiation_data, growth_data, alpha=0.5)
            plt.title(f'Radiation vs. {col}')
            plt.xlabel('Radiation (µSv/hr)')
            plt.ylabel(f'Population ({col})')
            plt.savefig(f'correlation_{col}_{timestamp}.png')
            plt.close()

    viable_cells_real = df1["Mean Log (Number of Viable Cells) Control ± SD"].apply(lambda x: float(x.split("±")[0])).values
    viable_cells_simulated = df2['Population'][:len(viable_cells_real)].values
    t_stat, p_val = ttest_ind(viable_cells_real, viable_cells_simulated)
    print(f"T-statistic: {t_stat}, P-value: {p_val}")

    if 'Extinct_Species_Index' in df2.columns:
        extinct_indices = df2['Extinct_Species_Index'].dropna().values
        for col in df2.columns:
            if col not in ['Extinct_Species_Index', 'Species Name', 'dna_sequence', 'rna_sequence'] and not has_constant_values(df2[col].values):
                data_col = df2[col].values[:len(extinct_indices)]
                corr, _ = pearsonr(extinct_indices, data_col)
                print(f"Pearson correlation between Extinct_Species_Index and {col}: {corr}")

    # Merge datasets
    merged_df = pd.merge(df1.reset_index(), df2.reset_index(), left_index=True, right_index=True, how='inner')
    correlation_matrix = merged_df.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm')
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.savefig(f'correlation_matrix_{timestamp}.png')
    plt.close()

    # Extend Dataset
    def generate_extended_row(row):
        new_row = row.copy()
        new_row['Species Name'] = random.choice(species_names)
        new_row['Population'] = np.random.uniform(1, 200)
        new_row['replication_rate'] = np.random.uniform(0.03, 0.08)
        new_row['decay_rate'] = np.random.uniform(0.005, 0.02)
        new_row['aging_rate'] = np.random.uniform(0.001, 0.005)
        new_row['lifespan'] = np.random.randint(10, 50)
        new_row['anti_life_effect'] = np.random.uniform(0.001, 0.005)
        new_row['radiation_effectiveness'] = np.random.uniform(0.001, 20)
        new_row['prob_same_species'] = np.random.uniform(0.01, 0.05)
        new_row['prob_diff_species'] = np.random.uniform(0.001, 0.01)
        new_row['interaction_strength'] = np.random.uniform(-0.01, 0.01, NUM_SPECIES)
        new_row['competitive_strength'] = np.random.uniform(0.001, 0.005)
        new_row['predation_rate'] = np.random.uniform(0.001, 0.005)
        new_row['birth_rate'] = np.random.uniform(0.01, 0.1)
        new_row['mutation_rate'] = np.random.uniform(0.001, 0.01)
        new_row['migration_rate'] = np.random.uniform(0.01, 0.1)
        new_row['resource_availability'] = np.random.uniform(0.1, 1.0)
        new_row['dispersal_rate'] = np.random.uniform(0.001, 0.01)
        new_row['environmental_tolerance'] = np.random.uniform(0.5, 1.0)
        new_row['carrying_capacity'] = np.random.uniform(50, 200)
        new_row['optimal_temperature'] = np.random.uniform(15, 35)
        new_row['optimal_humidity'] = np.random.uniform(40, 80)
        new_row['optimal_ph'] = np.random.uniform(5.5, 7.5)
        new_row['Melanin'] = species_data[new_row['Species Name']]['Melanin']
        new_row['melanin_present'] = fake.boolean()
        new_row['dna_sequence'] = quantum_system.generate_dna_sequence(DNA_SIZE)
        new_row['rna_sequence'] = quantum_system.generate_rna_sequence(new_row['dna_sequence'])
        new_row['Extinct_Species_Index'] = random.choice(quantum_system.extinct_indices) if quantum_system.extinct_indices else None
        return new_row

    extension_factor = 10
    extended_data = Parallel(n_jobs=-1)(delayed(generate_extended_row)(row) for row in data1 for _ in range(extension_factor))
    extended_df = pd.DataFrame(extended_data)
    extended_df.to_csv(f'extended_fungal_dataset_{timestamp}.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f'Extended dataset saved to extended_fungal_dataset_{timestamp}.csv')