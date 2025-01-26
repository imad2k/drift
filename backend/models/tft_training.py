import os
import multiprocessing as mp
os.environ["PYTORCH_FORECASTING_DISABLE_PTOPT_DETECTION"] = "1"
mp.set_start_method("spawn", force=True)

import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data._utils.collate import default_collate
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import warnings
warnings.filterwarnings("ignore")

import signal
import sys
from contextlib import contextmanager
from torch.utils.data import DataLoader
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# If you want to store results in the same DB table:
from db.database import save_to_rds
from datetime import datetime

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Training timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def cleanup_workers(dataloader):
    if isinstance(dataloader, DataLoader) and hasattr(dataloader, '_iterator'):
        try:
            dataloader._iterator._shutdown_workers()
        except:
            pass

def prepare_data(df, ticker):
    """Prepare data for TFT model"""
    df = df.copy()
    df["time_idx"] = np.arange(len(df))
    df["ticker"] = ticker

    # Define features
    known_reals = [c for c in df.columns 
                   if c not in ["date", "target", "time_idx", "ticker"]]
    return df, known_reals

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, df, ticker, batch_size=32):
        super().__init__()
        self.df = df
        self.ticker = ticker
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        df = self.df.copy()
        df["time_idx"] = np.arange(len(df))
        df["ticker"] = self.ticker
        
        self.known_reals = [c for c in df.columns 
                           if c not in ["date", "target", "time_idx", "ticker"]]
        
        self.training = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="target",
            group_ids=["ticker"],
            min_encoder_length=15,
            max_encoder_length=30,
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=["ticker"],
            time_varying_known_reals=self.known_reals,
            target_normalizer=GroupNormalizer(
                groups=["ticker"],
                transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        
    def train_dataloader(self):
        return self.training.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=0
        )

class TFTDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=self.collate_fn, **kwargs)
        
    def collate_fn(self, batch):
        # Convert batch to dictionary format
        if len(batch) == 0:
            return {}
            
        batch_converted = {
            'encoder_cat': torch.stack([b[0] for b in batch]),
            'encoder_cont': torch.stack([b[1] for b in batch]),
            'encoder_lengths': torch.tensor([b[2] for b in batch]),
            'decoder_cat': torch.stack([b[3] for b in batch]),
            'decoder_cont': torch.stack([b[4] for b in batch]),
            'decoder_lengths': torch.tensor([b[5] for b in batch])
        }
        
        if len(batch[0]) > 6:
            batch_converted['decoder_target'] = torch.stack([b[6] for b in batch])
        if len(batch[0]) > 7:
            batch_converted['target_scale'] = torch.stack([b[7] for b in batch])
            
        return batch_converted

@dataclass
class BatchData:
    encoder_cat: torch.Tensor
    encoder_cont: torch.Tensor
    encoder_lengths: torch.Tensor
    decoder_cat: torch.Tensor
    decoder_cont: torch.Tensor
    decoder_lengths: torch.Tensor
    decoder_target: Optional[torch.Tensor] = None
    target_scale: Optional[torch.Tensor] = None

class StockDataset(TimeSeriesDataSet):
    def __init__(self, data, **kwargs):
        self.data = data
        super().__init__(data, **kwargs)
    
    def __getitem__(self, idx):
        # Get base item and convert to BatchData
        item = super().__getitem__(idx)
        return BatchData(
            encoder_cat=item[0],
            encoder_cont=item[1],
            encoder_lengths=item[2],
            decoder_cat=item[3],
            decoder_cont=item[4],
            decoder_lengths=item[5],
            decoder_target=item[6] if len(item) > 6 else None,
            target_scale=item[7] if len(item) > 7 else None
        ).__dict__

class StockTFT(pl.LightningModule):
    def __init__(self, dataset):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset'])
        self.learning_rate = 0.001
        # NOTE: Use single quantile [0.5] if output_size=1
        self.model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=self.learning_rate,
            hidden_size=8,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=1,
            loss=QuantileLoss(quantiles=[0.5])  # single dimension => fix IndexError
        )

    def _validate_batch(self, x):
        required_keys = [
            'encoder_cat', 'encoder_cont', 'encoder_lengths',
            'decoder_cat', 'decoder_cont', 'decoder_lengths'
        ]
        if not all(k in x for k in required_keys):
            raise ValueError(f"Missing required keys in batch. Found: {list(x.keys())}")
        return x

    def forward(self, x):
        # Handle different input types
        if isinstance(x, (list, tuple)):
            # Convert single item
            x = x[0] if len(x) > 0 else {}
        
        if not isinstance(x, dict):
            raise ValueError(f"Expected dict input, got {type(x)}")
            
        x = self._validate_batch(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def custom_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Convert batch to TFT format"""
    elem = batch[0]
    if isinstance(elem, dict):
        return {
            key: torch.stack([d[key] for d in batch]) 
            if torch.is_tensor(elem[key]) 
            else [d[key] for d in batch]
            for key in elem.keys()
        }
    return batch

def validate_data(df):
    """Validate and clean input data"""
    # Check for NaN values
    if df.isna().any().any():
        df = df.fillna(method='ffill')
        
    # Ensure proper types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    
    return df

def train_tft_on_aggdf(df_merged_fund, ticker="AAPL.US"):
    df = df_merged_fund.copy()
    df["time_idx"] = np.arange(len(df))
    df["ticker"] = ticker
    
    known_reals = [c for c in df.columns 
                   if c not in ["date", "target", "time_idx", "ticker"]]
    
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
        target_normalizer=GroupNormalizer(
            groups=["ticker"],
            transformation="softplus"
        )
    )
    
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    try:
        model = StockTFT(training)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=True,
            logger=False
        )
        
        trainer.fit(model, train_dataloader)
        return model, {
            "train_loss": float(trainer.callback_metrics.get("train_loss", 0.0))
        }
    except Exception as e:
        print(f"[train_tft] Training error: {str(e)}")
        raise

def process_batch(batch):
    """Convert batch to TFT format"""
    if isinstance(batch[0], (tuple, list)):
        keys = ['encoder_cat', 'encoder_cont', 'encoder_lengths', 
                'decoder_cat', 'decoder_cont', 'decoder_lengths']
        processed = {k: torch.stack([b[i] for b in batch]) 
                    if i < 2 or i > 2 
                    else torch.tensor([b[i] for b in batch])
                    for i, k in enumerate(keys)}
        
        if len(batch[0]) > 6:
            processed['decoder_target'] = torch.stack([b[6] for b in batch])
        if len(batch[0]) > 7:
            processed['target_scale'] = torch.stack([b[7] for b in batch])
        return processed
    return batch
