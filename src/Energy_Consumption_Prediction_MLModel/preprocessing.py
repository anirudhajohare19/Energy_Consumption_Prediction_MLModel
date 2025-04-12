import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    logging.info("Starting preprocessing...")

    le = LabelEncoder()
    df['Building Type'] = le.fit_transform(df['Building Type'])
    df['Day of Week'] = le.fit_transform(df['Day of Week'])

    X = df.drop('Energy Consumption', axis=1)
    y = df['Energy Consumption']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logging.info("Preprocessing completed successfully.")
    return X_train_scaled, X_test_scaled, y_train, y_test
