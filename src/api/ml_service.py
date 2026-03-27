import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Re-create the exact LSTM architecture used during training
class ProductivityLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

# Global variables to cache the model so it doesn't slow down every API request
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_model = None
_feature_scaler = None
_target_scaler = None
_config = None

def load_ml_assets():
    """ Loads the trained PyTorch model and scikit-learn scalers into memory """
    global _model, _feature_scaler, _target_scaler, _config
    if _model is not None:
        return
        
    # 1. Load the scalers
    with open(os.path.join(MODELS_DIR, "lstm_scaler.pkl"), "rb") as f:
        scalers = pickle.load(f)
        _feature_scaler = scalers["feature_scaler"]
        _target_scaler = scalers["target_scaler"]
        
    # 2. Load the secure PyTorch checkpoint (forcing CPU for web server safety)
    checkpoint = torch.load(os.path.join(MODELS_DIR, "lstm_productivity.pth"), map_location=torch.device('cpu'), weights_only=False)
    _config = checkpoint["config"]
    
    # 3. Instantiate and inject the trained weights
    _model = ProductivityLSTM(
        input_dim=_config["input_dim"],
        hidden_dim=_config["hidden_dim"],
        num_layers=_config["num_layers"],
        dropout=0.1
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval() # Set to evaluation mode!

def generate_forecast(df: pd.DataFrame) -> int:
    """ 
    Takes the 5 days of BigQuery data, runs the exact transformations used 
    during training, and calculates tomorrow's predicted commits.
    """
    load_ml_assets()
    
    seq_len = _config["sequence_length"]
    if len(df) < seq_len:
        # Pad with zero rows at the beginning for developers with sparse data
        pad_rows = seq_len - len(df)
        pad_df = pd.DataFrame(0, index=range(pad_rows), columns=df.columns)
        pad_df["event_date"] = pd.NaT
        df = pd.concat([pad_df, df], ignore_index=True)
        
    # Derive the temporal features that the model expects
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["day_of_week"] = df["event_date"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_number"] = df["event_date"].dt.isocalendar().week.fillna(1).astype(int)
    
    # Extract just the 9 features in the exact order the LSTM expects
    features = df[_config["features"]].values
    
    # Scale down the inputs
    feat_scaled = _feature_scaler.transform(features)
    
    # Convert to a PyTorch tensor (Batch=1, Seq=5, Features=9)
    tensor_input = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(0)
    
    # Run the live prediction!
    with torch.no_grad():
        pred_scaled = _model(tensor_input).numpy()
        
    # Inverse the mathematical transformations (the model was trained on log1p)
    log_pred = _target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    raw_pred = np.expm1(log_pred)[0]
    
    # We can't commit 0.3 times, so round to the nearest whole number
    return int(round(max(0, raw_pred)))
