# stock_prediction_app/models/model_training.py

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def hyperparameter_tuning(model, param_grid, X_train, y_train, search_type="random"):
    """
    Original hyperparam-tuning function from your script, supporting both random and grid search.
    """
    try:
        if search_type == "random":
            search = RandomizedSearchCV(
                model, 
                param_distributions=param_grid, 
                n_iter=10, 
                cv=3, 
                random_state=42, 
                scoring="neg_mean_squared_error"
            )
        elif search_type == "grid":
            search = GridSearchCV(
                model, 
                param_grid=param_grid, 
                cv=3, 
                scoring="neg_mean_squared_error"
            )
        else:
            raise ValueError("search_type must be 'random' or 'grid'")
        
        search.fit(X_train, y_train)
        return search.best_estimator_
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        return model


def train_and_predict(X_train, y_train, X_test, y_test):
    """
    The original function that trains multiple models with hyperparameter tuning,
    plus an LSTM, then returns predictions & metrics. 
    Not currently used in your final route code, but we keep it so you don't lose functionality!
    """
    models = {
        "RandomForest": (
            RandomForestRegressor(), 
            {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10]}
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(), 
            {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
        ),
        "XGBoost": (
            XGBRegressor(), 
            {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
        ),
        "CatBoost": (
            CatBoostRegressor(verbose=0), 
            {"iterations": [100, 200], "depth": [4, 6, 10]}
        ),
        "LightGBM": (
            LGBMRegressor(), 
            {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1]}
        )
    }
    predictions = {}

    # Train and predict for each model with hyperparam tuning
    for name, (model, param_grid) in models.items():
        print(f"Training {name}...")
        best_model = hyperparameter_tuning(model, param_grid, X_train, y_train)
        predictions[name] = best_model.predict(X_test)

    # LSTM model
    print("Training LSTM...")
    timesteps = 10
    # X_train, X_test might be DataFrames => convert to .values
    X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, timesteps)
    X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, timesteps)

    lstm_model = Sequential([
        LSTM(50, activation="relu", input_shape=(timesteps, X_train_seq.shape[2])),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=0)

    predictions["LSTM"] = lstm_model.predict(X_test_seq).flatten()

    # Ensemble: average all predictions
    ensemble_pred = np.mean(list(predictions.values()), axis=0)

    # Evaluate
    mse = mean_squared_error(y_test, ensemble_pred)
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)
    percent_error = np.mean((ensemble_pred - y_test) / y_test) * 100

    return predictions, {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "percent_error": percent_error
    }


def create_sequences(data, target, timesteps):
    """
    Helper function to create LSTM-friendly sequences from raw data.
    """
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        y.append(target[i + timesteps])
    return np.array(X), np.array(y)
