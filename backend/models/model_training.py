# stock_prediction_app/models/model_training.py

import sys
import warnings
import logging

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class SafeXGBRegressor(BaseEstimator, RegressorMixin):
    """
    A fully scikit-learn compliant XGBoost regressor that does *not* subclass
    xgboost.XGBRegressor. It uses xgboost.train() directly and enumerates each
    hyperparameter so that RandomizedSearchCV / GridSearchCV can set them safely.
    """

    _estimator_type = "regressor"  # Required by scikit-learn

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        eval_metric="rmse"
    ):
        """
        Store hyperparameters as normal Python attributes so that
        scikit-learn can inspect/update them with get_params/set_params.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.eval_metric = eval_metric

        self._booster = None  # Will hold the trained Booster

    def fit(self, X, y, sample_weight=None):
        """
        Train an XGBoost Booster using xgboost.train. 
        """
        dtrain = xgb.DMatrix(data=X, label=y, weight=sample_weight)
        params = {
            "objective": "reg:squarederror",
            "eval_metric": self.eval_metric,
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "seed": self.random_state,
        }
        self._booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators
        )
        return self

    def predict(self, X):
        """
        Predict using the trained Booster.
        """
        if self._booster is None:
            raise ValueError("SafeXGBRegressor not fitted yet.")
        dtest = xgb.DMatrix(data=X)
        return self._booster.predict(dtest)

    def __sklearn_tags__(self):
        """
        Minimal set of tags for scikit-learn >= 1.2
        so that cross-validation does not choke on them.
        """
        return {
            "allow_nan": True,
            "non_deterministic": True,
            "X_types": ["2darray"]
        }

    def get_params(self, deep=True):
        """
        Return model hyperparameters as a dict so scikit-learn can do hyperparameter search.
        """
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "eval_metric": self.eval_metric
        }

    def set_params(self, **params):
        """
        Set parameters. Ensures scikit-learn can pass hyperparameters to this class
        without turning it into a dict or messing with internal attributes.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


def hyperparameter_tuning(model, param_grid, X_train, y_train, search_type="random", n_iter=10, cv=3):
    """
    Performs hyperparameter tuning using RandomizedSearchCV or GridSearchCV.
    If tuning fails, falls back to training the model with default parameters.
    """
    try:
        if search_type == "random":
            search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                random_state=42,
                scoring="neg_mean_squared_error",
                verbose=1,
                error_score="raise"
            )
        elif search_type == "grid":
            search = GridSearchCV(
                model,
                param_grid=param_grid,
                cv=cv,
                scoring="neg_mean_squared_error",
                verbose=1,
                error_score="raise"
            )
        else:
            raise ValueError("search_type must be 'random' or 'grid'")

        logger.debug(f"Starting hyperparameter search for {model.__class__.__name__} with search type '{search_type}'.")
        search.fit(X_train, y_train)
        logger.info(f"Best parameters for {model.__class__.__name__}: {search.best_params_}")
        return search.best_estimator_

    except Exception as e:
        logger.error(f"Hyperparameter tuning failed for {model.__class__.__name__}: {e}")
        logger.info("Falling back to the default model without hyperparameter tuning.")
        try:
            model.fit(X_train, y_train)
            logger.info(f"{model.__class__.__name__} fitted with default parameters.")
            return model
        except Exception as fit_e:
            logger.error(f"Failed to fit the default model: {fit_e}")
            raise fit_e


def train_and_predict(X_train, y_train, X_test, y_test):
    """
    Pipeline to train multiple tabular models and a Temporal Fusion Transformer (TFT),
    make predictions, create an ensemble, and calculate metrics.
    """
    predictions_dict = {}

    # ---------------------------------------------------------
    # 1) Train Tabular Models
    # ---------------------------------------------------------
    logger.info("== Training Tabular Models ==")
    tabular_configs = {
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 10, 15],
                "min_samples_split": [2, 5, 10]
            }
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 8]
            }
        ),
        # Now use SafeXGBRegressor in place of xgboost.XGBRegressor:
        "XGBoost": (
            SafeXGBRegressor(random_state=42, eval_metric="rmse"),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 8],
                # You can tune other XGBoost params like subsample, colsample_bytree, etc.:
                # "subsample": [0.8, 1.0],
                # "colsample_bytree": [0.8, 1.0],
            }
        ),
        "CatBoost": (
            CatBoostRegressor(verbose=0, random_state=42),
            {
                "iterations": [100, 200, 300],
                "depth": [4, 6, 10]
            }
        ),
        "LightGBM": (
            LGBMRegressor(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [-1, 5, 8]
            }
        )
    }

    for name, (model, param_grid) in tabular_configs.items():
        logger.info(f"\n== Hyperparameter Tuning for {name} ==")
        best_model = hyperparameter_tuning(
            model,
            param_grid,
            X_train,
            y_train,
            search_type="random",
            n_iter=5,
            cv=3
        )
        logger.debug(f"Generating predictions for {name}.")
        try:
            predictions = best_model.predict(X_test)
            predictions_dict[name] = predictions
            logger.info(f"{name} training and prediction completed.")
        except Exception as pred_e:
            logger.error(f"Prediction failed for {name}: {pred_e}")

    # ---------------------------------------------------------
    # 2) Train Temporal Fusion Transformer (TFT)
    # ---------------------------------------------------------
    logger.info("\n== Training Temporal Fusion Transformer (TFT) ==")

    time_idx_train = np.arange(len(X_train))
    time_idx_test = np.arange(len(X_train), len(X_train) + len(X_test))

    train_df = pd.DataFrame({
        "time_idx": time_idx_train,
        "ticker": "AAPL.US",
        "target": y_train
    })

    test_df = pd.DataFrame({
        "time_idx": time_idx_test,
        "ticker": "AAPL.US",
        "target": y_test
    })

    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    for idx, feature in enumerate(feature_names):
        train_df[feature] = X_train[:, idx]
        test_df[feature] = X_test[:, idx]

    max_encoder_length = 30
    max_prediction_length = 1

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["ticker"],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=feature_names,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)

    batch_size = 32
    num_workers = 4
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    try:
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=1,
            loss=RMSE(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
    except Exception as e:
        logger.error(f"Failed to initialize TFT: {e}")
        raise e

    if not isinstance(tft, torch.nn.Module):
        raise TypeError("TemporalFusionTransformer is not a torch.nn.Module")

    trainer = Trainer(
        max_epochs=30,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            TQDMProgressBar(refresh_rate=10)
        ],
        logger=True,
        enable_progress_bar=True,
    )

    logger.info("Starting TFT training...")
    try:
        trainer.fit(
            model=tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        logger.info("TFT training completed.")
    except Exception as train_e:
        logger.error(f"Error during TFT training: {train_e}")
        raise train_e

    # Predict on validation
    logger.info("Generating TFT predictions...")
    predictions = tft.predict(val_dataloader, mode="prediction")
    tft_preds = predictions.numpy().flatten()
    predictions_dict["TFT"] = tft_preds

    # ---------------------------------------------------------
    # 3) Ensemble
    # ---------------------------------------------------------
    logger.info("\n== Creating Ensemble Prediction ==")
    aligned_predictions = [preds for preds in predictions_dict.values() if len(preds) == len(tft_preds)]
    if not aligned_predictions:
        raise ValueError("No aligned predictions found for ensemble.")

    final_preds_stack = np.column_stack(aligned_predictions)
    ensemble_pred = final_preds_stack.mean(axis=1)

    # ---------------------------------------------------------
    # 4) Metrics
    # ---------------------------------------------------------
    logger.info("\n== Calculating Metrics ==")
    if len(y_test) != len(ensemble_pred):
        raise ValueError("Ensemble predictions and y_test have mismatched lengths!")

    mse_val = mean_squared_error(y_test, ensemble_pred)
    mae_val = mean_absolute_error(y_test, ensemble_pred)
    r2_val = r2_score(y_test, ensemble_pred)
    pct_err = np.mean((ensemble_pred - y_test) / y_test) * 100

    metrics = {
        "mse": mse_val,
        "mae": mae_val,
        "r2": r2_val,
        "percent_error": pct_err
    }
    logger.info("=== Ensemble Metrics ===")
    logger.info(f"MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, R^2: {r2_val:.4f}, %Error: {pct_err:.2f}")

    return predictions_dict, metrics


# ------------------------------------------------------------------
# EXAMPLE OF SAFELY PROCESSING FUNDAMENTAL DATA
# ------------------------------------------------------------------
def process_fundamental_data(fund_data_json):
    try:
        if not fund_data_json or fund_data_json == 0:
            logger.warning("Fundamental data is empty or invalid.")
            return {}
        return fund_data_json
    except KeyError as e:
        logger.error(f"Error processing fundamental data: {e}")
        return {}
