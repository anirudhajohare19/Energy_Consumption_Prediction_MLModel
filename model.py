# Importing Data Manipulation Libraries
import pandas as pd
import numpy as np

# importing Data Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the Filter Warning Library
import warnings
warnings.filterwarnings('ignore')

# importing logging library
import logging
logging.basicConfig(filename='model.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')

D1 = 'https://raw.githubusercontent.com/anirudhajohare19/Energy_Consumption_Prediction_MLModel/refs/heads/main/train_energy_data.csv'
D2 = 'https://raw.githubusercontent.com/anirudhajohare19/Energy_Consumption_Prediction_MLModel/refs/heads/main/test_energy_data.csv'

df1 = pd.read_csv(D1)
df2 = pd.read_csv(D2)


# Combining the datasets
df = pd.concat([df1,df2], ignore_index=True)

# Converting Categorical Variables to Numerical Variables ---> Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Building Type'] = le.fit_transform(df['Building Type'])
df['Day of Week'] = le.fit_transform(df['Day of Week'])

# spliting the dataset into X and y
X = df.drop('Energy Consumption',axis = 1)
y = df['Energy Consumption']

# Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaling the features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Random Forest Regressor Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RF = RandomForestRegressor()

# Training the model
RF.fit(X_train, y_train)

# Predicting on the test set
y_pred_RF = RF.predict(X_test)

r2_score(y_test, y_pred_RF)

print(f'R-squared Score: {r2_score(y_test, y_pred_RF)}')