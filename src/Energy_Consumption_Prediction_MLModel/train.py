import logging

def train_model(model, X_train, y_train):
    logging.info("Training the model...")
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    return model
