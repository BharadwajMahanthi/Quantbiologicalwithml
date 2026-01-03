import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf # type: ignore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # type: ignore

def plot_population_history(history_df, save_path=None):
    plt.figure(figsize=(10, 6))
    for column in history_df.columns:
        if column.startswith('Species'):
             plt.plot(history_df.index, history_df[column], label=column)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Population')
    plt.title('Species Population Over Time')
    plt.legend(loc='upper right', fontsize='small') # type: ignore
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    return result[1] < 0.05 # True if stationary

def analyze_species(data, species_name):
    if species_name in data.columns:
        ts = data[species_name]
        stationarity = check_stationarity(ts)
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(ts, lags=20, ax=ax[0])
        plot_pacf(ts, lags=20, ax=ax[1])
        plt.tight_layout()
        plt.show()
    else:
        print(f"Species {species_name} not found in data.")
