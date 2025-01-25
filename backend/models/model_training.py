# stock_prediction_app/models/model_training.py

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def hyperparameter_tuning(model, param_grid, X_train, y_train, search_type="random", n_iter=10, cv=3):
    """
    Supports random or grid search with expanded coverage.
    """
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    
    try:
        if search_type == "random":
            search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                random_state=42,
                scoring="neg_mean_squared_error",
                verbose=1
            )
        elif search_type == "grid":
            search = GridSearchCV(
                model,
                param_grid=param_grid,
                cv=cv,
                scoring="neg_mean_squared_error",
                verbose=1
            )
        else:
            raise ValueError("search_type must be 'random' or 'grid'")
        
        search.fit(X_train, y_train)
        return search.best_estimator_
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        return model

def create_sequences(data, target, timesteps):
    """
    Creates sliding-window sequences for an LSTM.
    data: shape (samples, features)
    target: shape (samples,)
    timesteps: window length
    returns: X_seq, y_seq
    """
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(target[i + timesteps])
    return np.array(X), np.array(y)


def train_and_predict(X_train, y_train, X_test, y_test):
    """
    Example pipeline:
      1) Hyperparam-tunes multiple tabular models (RF, GBM, XGB, CatBoost, LightGBM).
      2) Scales data + trains a deeper LSTM.
      3) Returns predictions & metrics.
    """

    # ---------------------------------------------------------
    # 1) Tabular Model Hyperparams
    # ---------------------------------------------------------
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    tabular_configs = {
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [50,100,200,300],
                "max_depth": [3,5,10,15],
                "min_samples_split": [2,5,10]
            }
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [50,100,200],
                "learning_rate": [0.01,0.05,0.1,0.2],
                "max_depth": [3,5,8]
            }
        ),
        "XGBoost": (
            XGBRegressor(random_state=42, use_label_encoder=False, eval_metric="rmse"),
            {
                "n_estimators": [50,100,200],
                "learning_rate": [0.01,0.05,0.1],
                "max_depth": [3,5,8]
            }
        ),
        "CatBoost": (
            CatBoostRegressor(verbose=0, random_state=42),
            {
                "iterations": [100,200,300],
                "depth": [4,6,10]
            }
        ),
        "LightGBM": (
            LGBMRegressor(random_state=42),
            {
                "n_estimators": [50,100,200],
                "learning_rate": [0.01,0.05,0.1,0.2],
                "max_depth": [-1,5,8]
            }
        )
    }

    predictions_dict = {}

    # ---------------------------------------------------------
    # 2) Train each tabular model
    # ---------------------------------------------------------
    for name, (model, param_grid) in tabular_configs.items():
        print(f"\n== Hyperparameter Tuning for {name} ==")
        best_model = hyperparameter_tuning(
            model,
            param_grid,
            X_train,
            y_train,
            search_type="random",
            n_iter=5,  # or 10 for more thorough search
            cv=3
        )
        predictions_dict[name] = best_model.predict(X_test)

    # ---------------------------------------------------------
    # 3) LSTM with scaling
    # ---------------------------------------------------------
    print("\n== Training LSTM with MinMax Scaling ==")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_arr = np.array(X_train)
    y_train_arr = np.array(y_train).reshape(-1,1)

    X_test_arr  = np.array(X_test)
    y_test_arr  = np.array(y_test).reshape(-1,1)

    X_train_scaled = scaler_X.fit_transform(X_train_arr)
    X_test_scaled  = scaler_X.transform(X_test_arr)
    y_train_scaled = scaler_y.fit_transform(y_train_arr)
    y_test_scaled  = scaler_y.transform(y_test_arr)

    timesteps = 10
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled.flatten(), timesteps)
    X_test_seq, y_test_seq   = create_sequences(X_test_scaled,  y_test_scaled.flatten(),  timesteps)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(timesteps, X_train_seq.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_train_seq, y_train_seq, 
                   validation_split=0.2,
                   epochs=20,
                   batch_size=32,
                   verbose=1)

    lstm_preds_scaled = lstm_model.predict(X_test_seq).flatten()
    lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled.reshape(-1,1)).flatten()

    # offset tabular predictions by timesteps
    for name in predictions_dict:
        predictions_dict[name] = predictions_dict[name][timesteps:]
    y_test_final = y_test_arr.flatten()[timesteps:]

    # LSTM final predictions
    predictions_dict["LSTM"] = lstm_preds

    # ---------------------------------------------------------
    # 4) Ensemble
    # ---------------------------------------------------------
    # We'll do a simple mean ensemble of all models that have the same length as LSTM preds
    final_preds_stack = np.column_stack([
        preds for preds in predictions_dict.values()
        if len(preds) == len(lstm_preds)
    ])
    ensemble_pred = final_preds_stack.mean(axis=1)

    # ---------------------------------------------------------
    # 5) Metrics
    # ---------------------------------------------------------
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse_val = mean_squared_error(y_test_final, ensemble_pred)
    mae_val = mean_absolute_error(y_test_final, ensemble_pred)
    r2_val  = r2_score(y_test_final, ensemble_pred)
    pct_err = np.mean((ensemble_pred - y_test_final)/y_test_final)*100

    metrics = {
        "mse": mse_val,
        "mae": mae_val,
        "r2": r2_val,
        "percent_error": pct_err
    }
    print("=== Ensemble Metrics ===")
    print(f"MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, R^2: {r2_val:.4f}, %Error: {pct_err:.2f}")

    return predictions_dict, metrics
