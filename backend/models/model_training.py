# stock_prediction_app/models/model_training.py

import sys
import warnings
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def xgboost_dmatrix(X, y=None, sample_weight=None):
    """
    Helper to build an xgb.DMatrix with optional labels/sample_weight.
    """
    if y is None:
        return xgb.DMatrix(data=X)
    else:
        return xgb.DMatrix(data=X, label=y, weight=sample_weight)


class SafeXGBRegressor(BaseEstimator, RegressorMixin):
    """
    A fully scikit-learn compliant XGBoost regressor that does NOT subclass
    xgboost.XGBRegressor. It uses xgboost.train() under the hood so we can
    integrate with RandomizedSearchCV / GridSearchCV without issues.

    The critical fix is `_more_tags_()`, which ensures scikit-learn 1.2+ 
    sees this as a regressor and doesn't throw `'dict' object has no attribute 'estimator_type'`.
    """

    _estimator_type = "regressor"

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

        self._booster = None

    def _more_tags_(self):
        """
        For scikit-learn 1.2+, `_more_tags_` is how we specify
        'estimator_type' so that `check_cv(..., is_classifier(estimator))`
        won't break.
        """
        return {
            "allow_nan": True,
            "non_deterministic": True,
            "X_types": ["2darray"],
            "estimator_type": "regressor"
        }

    def fit(self, X, y, sample_weight=None):
        dtrain = xgboost_dmatrix(X, y, sample_weight)
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
        if self._booster is None:
            raise ValueError("SafeXGBRegressor is not fitted yet.")
        dtest = xgboost_dmatrix(X)
        return self._booster.predict(dtest)

    def get_params(self, deep=True):
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
        for k, v in params.items():
            setattr(self, k, v)
        return self


def train_tabular_ensemble(
    X_train, y_train, X_test, y_test,
    random_seeds=[42],
    cv_folds=3,
    n_iter=5
):
    """
    Train an ensemble of tabular models (RF, GradientBoost, XGBoost, CatBoost, LightGBM)
    using RandomizedSearchCV with 'cv_folds' folds and 'n_iter' search combos.

    For each model, we average predictions over all seeds.
    Then we ensemble across models by averaging predictions.

    Returns:
      final_preds_dict -> { model_name -> final (averaged) predictions on X_test }
      ensemble_pred -> np.array, the average of all model predictions
      metrics -> {'mse':..., 'mae':..., 'r2':...}
    """

    param_grids = {
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 8, 12],
            "min_samples_split": [2, 5, 10]
        },
        "GradientBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.02, 0.05, 0.1],
            "max_depth": [3, 5, 8]
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.02, 0.05],
            "max_depth": [3, 5, 8],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [0, 0.1, 1]
        },
        "CatBoost": {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8]
        },
        "LightGBM": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.02, 0.05],
            "max_depth": [-1, 3, 5, 8],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        }
    }

    models = {
        "RandomForest": RandomForestRegressor(),
        "GradientBoost": GradientBoostingRegressor(),
        "XGBoost": SafeXGBRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0),
        "LightGBM": LGBMRegressor()
    }

    from sklearn.model_selection import RandomizedSearchCV

    # predictions for each model across seeds
    seed_predictions = {m_name: [] for m_name in models}

    for seed in random_seeds:
        for m_name, base_model in models.items():
            if hasattr(base_model, "random_state"):
                base_model.set_params(random_state=seed)
            if isinstance(base_model, SafeXGBRegressor):
                base_model.set_params(random_state=seed)

            param_dist = param_grids[m_name]
            logger.info(f"[{m_name}] Searching {n_iter} param combos with seed={seed}")
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                verbose=1,
                random_state=seed
            )
            search.fit(X_train, y_train)
            best_est = search.best_estimator_
            logger.info(f"[{m_name}] Best params (seed={seed}): {search.best_params_}")

            # Predict
            test_preds = best_est.predict(X_test)
            seed_predictions[m_name].append(test_preds)

    # Average across seeds
    final_preds_dict = {}
    for m_name, pred_lists in seed_predictions.items():
        if pred_lists:
            arr = np.column_stack(pred_lists)
            avg_preds = arr.mean(axis=1)
            final_preds_dict[m_name] = avg_preds
        else:
            final_preds_dict[m_name] = np.zeros(len(X_test))

    # Ensemble across models
    if final_preds_dict:
        model_arrays = list(final_preds_dict.values())
        ensemble_pred = np.mean(np.column_stack(model_arrays), axis=1)
    else:
        ensemble_pred = np.zeros(len(X_test))

    # Evaluate
    mse_val = mean_squared_error(y_test, ensemble_pred)
    mae_val = mean_absolute_error(y_test, ensemble_pred)
    r2_val  = r2_score(y_test, ensemble_pred)

    metrics = {
        "mse": mse_val,
        "mae": mae_val,
        "r2": r2_val
    }

    return final_preds_dict, ensemble_pred, metrics
