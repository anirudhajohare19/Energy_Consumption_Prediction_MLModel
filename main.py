import sys
import os

# Add 'src' folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import logging
from src.Energy_Consumption_Prediction_MLModel.data_loader import load_data
from src.Energy_Consumption_Prediction_MLModel.preprocessing import preprocess_data
from src.Energy_Consumption_Prediction_MLModel.Model import build_model
from src.Energy_Consumption_Prediction_MLModel.train import train_model
from src.Energy_Consumption_Prediction_MLModel.evaluate import evaluate_model

# Setup logging
logging.basicConfig(filename='model.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')

def main():
    df = load_data()  # Load the data
    X_train, X_test, y_train, y_test = preprocess_data(df)  # Preprocess the data
    model = build_model()  # Build the model
    trained_model = train_model(model, X_train, y_train)  # Train the model
    evaluate_model(trained_model, X_test, y_test)  # Evaluate the model

if __name__ == "__main__":
    main()
