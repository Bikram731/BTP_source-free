import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
import random
import re

# --- 1. CONFIGURATION ---
class CFG:
    # --- Data Paths ---
    DATA_DIR = "./data/seu_gear_data/" 
    OUTPUT_MODEL_NAME = "source_model_seu.pt"
    
    # --- Data Processing ---
    WINDOW_SIZE = 2000
    STEP = 500
    COLUMN_TO_USE = 0 

    # --- Training ---
    NUM_CLASSES = 4 # Healthy, Chipped, Crack, Missing
    TRAIN_RATIO = 0.8
    RANDOM_STATE = 42
    LR = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 50 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore")
def set_seed(seed=CFG.RANDOM_STATE):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed()

# --- 2. MODEL DEFINITION ---
class TargetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Flatten(),
        )
        self.label_predictor = nn.Sequential(
            nn.Linear(in_features=8000, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=num_classes)
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.label_predictor(features)
        return logits

# --- 3. DATA LOADING & PREPARATION ---
class SourceData(Dataset):
    def __init__(self, X, y):
        self.X = X.astype("float32")
        self.y = y.astype(int)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx]; y = self.y[idx]
        return torch.from_numpy(x[None,:]), torch.tensor(y)

def _windows_from_signal(sig, win=CFG.WINDOW_SIZE, step=CFG.STEP):
    if sig.size < win: sig = np.pad(sig, (0, win - sig.size), mode="edge")
    out = []
    for st in range(0, len(sig) - win + 1, step):
        w = sig[st:st + win].astype("float32")
        m, s = float(w.mean()), float(w.std())
        w = (w - m) / (s + 1e-6)
        out.append(w)
    return out

def get_label_from_filename(filename):
    """
    Gets the label (0-3) from the SEU filename.
    """
    filename = os.path.basename(filename).lower()
    
    if filename.startswith("health"):
        return 0  # Maps to target "healthy"
    if filename.startswith("chipped"):
        return 1  # Maps to target "chipped"
    if filename.startswith("root"):
        return 2  # Maps to target "crack"
    if filename.startswith("miss"):
        return 3  # Maps to target "missing"
        
    return None # Skip 'Surface' files or any others

def load_seu_data():
    all_windows = []
    all_labels = []
    
    search_path = os.path.join(CFG.DATA_DIR, "*.csv")
    file_paths = glob.glob(search_path)
    
    if not file_paths:
        raise FileNotFoundError(f"No .csv files found in {CFG.DATA_DIR}.")
        
    print(f"Found {len(file_paths)} .csv files. Loading...")

    for file_path in file_paths:
        label = get_label_from_filename(file_path)
        
        if label is None:
            continue
            
        try:
            # --- THIS IS THE FIXED LINE ---
            # Use delim_whitespace=True for space-separated files
            df = pd.read_csv(file_path, header=None, delim_whitespace=True, skiprows=16)
            
        except Exception as e:
            print(f"  - ERROR loading {os.path.basename(file_path)}: {e}")
            continue
            
        if CFG.COLUMN_TO_USE >= len(df.columns):
            print(f"  - ERROR: File {os.path.basename(file_path)} does not have a column {CFG.COLUMN_TO_USE}")
            continue
            
        signal = pd.to_numeric(df[CFG.COLUMN_TO_USE], errors='coerce').dropna().values
        
        if signal.size == 0:
            print(f"  - WARNING: No numeric data found in {os.path.basename(file_path)}. Skipping.")
            continue
            
        windows = _windows_from_signal(signal)
        
        if windows:
            all_windows.extend(windows)
            all_labels.extend([label] * len(windows))
            
    if not all_windows:
        raise ValueError("No data was loaded! Check your 'get_label_from_filename' function and `skiprows` value.")
            
    return np.stack(all_windows), np.array(all_labels)

# --- 4. TRAINING & EVALUATION ---
def main():
    print(f"Using device: {CFG.DEVICE}")
    print("Loading 4-CLASS source (SEU Gear) data...")
    X, y = load_seu_data()
    print(f"Data loaded. Shape: {X.shape}, Labels: {y.shape}")
    
    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution:")
    for label, count in zip(unique, counts):
        print(f"  - Class {label}: {count} samples")
    
    idx_tr, idx_va = train_test_split(
        np.arange(len(X)), test_size=(1 - CFG.TRAIN_RATIO),
        random_state=CFG.RANDOM_STATE, stratify=y
    )
    
    ds_tr = SourceData(X[idx_tr], y[idx_tr])
    ds_va = SourceData(X[idx_va], y[idx_va])
    
    dl_tr = DataLoader(ds_tr, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(ds_tr)}, Validation samples: {len(ds_va)}")

    model = TargetModel(num_classes=CFG.NUM_CLASSES).to(CFG.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.LR)
    loss_fn = nn.CrossEntropyLoss()
    
    best_val_acc = -1.0
    best_model_state = None
    
    print("🚀 Starting training on 4-CLASS SEU source data...")
    for epoch in range(1, CFG.EPOCHS + 1):
        model.train(); total_loss = 0
        for x_b, y_b in dl_tr:
            x_b, y_b = x_b.to(CFG.DEVICE), y_b.to(CFG.DEVICE)
            logits = model(x_b)
            loss = loss_fn(logits, y_b)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            
        model.eval(); correct, total = 0, 0
        with torch.no_grad():
            for x_b, y_b in dl_va:
                x_b, y_b = x_b.to(CFG.DEVICE), y_b.to(CFG.DEVICE)
                logits = model(x_b)
                preds = logits.argmax(1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)
                
        val_acc = correct / total if total > 0 else 0
        avg_loss = total_loss / len(dl_tr) if len(dl_tr) > 0 else 0
        
        print(f"Epoch {epoch:03d}/{CFG.EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print(f"  -> New best model saved with Val Acc: {best_val_acc:.4f}")

    if best_model_state:
        torch.save(best_model_state, CFG.OUTPUT_MODEL_NAME)
        print(f"\n✅ Training complete. Best 4-CLASS SEU model saved to '{CFG.OUTPUT_MODEL_NAME}'")
    else:
        print("\n❌ Training failed to improve. No model saved.")

if __name__ == "__main__":
    main()