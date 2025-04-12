import pandas as pd
import logging

def load_data():
    logging.info("Loading datasets...")
    D1 = 'https://raw.githubusercontent.com/anirudhajohare19/Energy_Consumption_Prediction_MLModel/refs/heads/main/train_energy_data.csv'
    D2 = 'https://raw.githubusercontent.com/anirudhajohare19/Energy_Consumption_Prediction_MLModel/refs/heads/main/test_energy_data.csv'
    
    df1 = pd.read_csv(D1)
    df2 = pd.read_csv(D2)

    df = pd.concat([df1, df2], ignore_index=True)
    logging.info("Datasets loaded and combined successfully.")
    return df
