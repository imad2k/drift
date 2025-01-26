import os
import multiprocessing as mp
os.environ["PYTORCH_FORECASTING_DISABLE_PTOPT_DETECTION"] = "1"
mp.set_start_method("spawn", force=True)

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import warnings
warnings.filterwarnings("ignore")

def train_minimal_tft(df: pd.DataFrame, ticker="AAPL.US"):
    """
    Minimal script to test if TFT can train without segfault.
    Requirements:
      - df has columns: date (optional), target, plus numeric features
    """

    # 1) Basic checks
    if "target" not in df.columns:
        raise ValueError("DataFrame must have a 'target' column.")

    # 2) Sort + add time_idx + ticker
    if "date" in df.columns:
        df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["time_idx"] = np.arange(len(df))
    df["ticker"]   = ticker

    # 3) Identify known reals
    ignore = ["date", "target", "time_idx", "ticker"]
    known_reals = [c for c in df.columns if c not in ignore]

    # 4) Create TimeSeriesDataSet
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["ticker"],
        min_encoder_length=15,
        max_encoder_length=30,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["ticker"],
        time_varying_known_reals=known_reals,
        target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # 5) Dataloader
    train_loader = training.to_dataloader(
        train=True,
        batch_size=16,
        num_workers=0,   # single-threaded to avoid concurrency issues
    )

    # 6) Build TFT from dataset
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=8,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=1,
        loss=QuantileLoss(),
    )

    # 7) Trainer
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=True,
        logger=False,
        num_sanity_val_steps=0,  # skip val sanity check
        fast_dev_run=False,      # can set True if you want just 1 batch test
    )

    # 8) Fit
    trainer.fit(tft, train_loader)

    # 9) Return final train_loss (if any)
    train_loss = trainer.callback_metrics.get("train_loss")
    loss_val = float(train_loss) if train_loss is not None else None
    print(f"Finished minimal TFT training. Train loss = {loss_val}")

    return tft, loss_val

if __name__ == "__main__":
    # Example usage:
    # Create a dummy DataFrame with target
    data_len = 100
    df_example = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=data_len, freq="D"),
        "intraday_close_mean": np.random.rand(data_len) * 100,
        "macro_open": np.random.rand(data_len),
        "target": np.random.rand(data_len) * 50,
    })

    # Attempt training
    train_minimal_tft(df_example, ticker="AAPL.US")
