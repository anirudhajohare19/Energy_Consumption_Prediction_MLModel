import logging
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f'R-squared Score: {r2}')
    logging.info(f'Evaluation complete. R2: {r2}, MSE: {mse}, MAE: {mae}')
    return r2, mse, mae
