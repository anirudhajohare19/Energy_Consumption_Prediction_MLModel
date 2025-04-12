import logging
from sklearn.ensemble import RandomForestRegressor

def build_model():
    logging.info("Building Random Forest Regressor model.")
    model = RandomForestRegressor()
    return model
