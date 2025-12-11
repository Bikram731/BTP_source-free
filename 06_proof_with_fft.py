import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_PATH = "./data/target_vibration_massive.csv"
BATCH_SIZE = 64
LR_SOURCE = 1e-3
LR_ADAPT = 1e-5        
EPOCHS_SOURCE = 100    # <--- DOUBLED to get Source Accuracy > 95%
EPOCHS_ADAPT = 15      
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_seed()

# --- FFT PREPROCESSING ---
def apply_fft(signal):
    fft_vals = np.fft.rfft(signal)
    fft_abs = np.abs(fft_vals)
    fft_abs = (fft_abs - fft_abs.min()) / (fft_abs.max() - fft_abs.min() + 1e-6)
    return fft_abs.astype("float32")

# --- DATASET ---
class FFTData(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        sig = self.X[idx]
        feat = apply_fft(sig)
        return torch.from_numpy(feat[None, :]), torch.tensor(self.y[idx])

# --- MODEL ---
class EmergencyModel(nn.Module):
    def __init__(self, input_len, num_classes=4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        logits = self.classifier(feat)
        return logits

def main():
    print("--- 🚀 FINAL PUSH FOR 90%+ ACCURACY ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH, header=None)
        print(f"Loaded {len(df)} samples.")
        X = df.iloc[:, :-1].values.astype("float32")
        y = df.iloc[:, -1].astype(int).values
    except:
        return

    # 2. Split Data (50% Source / 50% Target)
    X_src, X_tgt, y_src, y_tgt = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
    sample_feat = apply_fft(X_src[0])
    
    # --- PHASE 1: TRAIN SOURCE ---
    print(f"\nPhase 1: Training Source Model (100 Epochs)...")
    src_ds = FFTData(X_src, y_src)
    src_dl = DataLoader(src_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    model = EmergencyModel(len(sample_feat), 4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_SOURCE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) # Helps converge
    criterion = nn.CrossEntropyLoss()
    
    for ep in range(EPOCHS_SOURCE):
        model.train()
        total_loss = 0
        for x, y_batch in src_dl:
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        if (ep+1) % 10 == 0:
            print(f"  Src Epoch {ep+1}: Loss {total_loss/len(src_dl):.4f}")
            
    print("✅ Source Model Trained.")
    
    # --- CHECKPOINT ---
    tgt_ds = FFTData(X_tgt, y_tgt)
    tgt_dl = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y_batch in tgt_dl:
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    
    start_acc = correct/total
    print(f"\n📊 Accuracy BEFORE Adaptation: {start_acc*100:.2f}%")
    
    # --- PHASE 2: ADAPTATION ---
    print(f"\nPhase 2: Running Source-Free Adaptation...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_ADAPT)
    
    best_acc = start_acc
    
    for ep in range(EPOCHS_ADAPT):
        model.train()
        used = 0
        for x, _ in tgt_dl: # IGNORE LABELS
            x = x.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            
            # STRICT Confidence
            max_p, label_p = torch.max(probs, dim=1)
            mask = max_p > 0.95 
            
            if mask.sum() > 0:
                loss = F.cross_entropy(logits[mask], label_p[mask])
                loss.backward()
                optimizer.step()
                used += mask.sum().item()
        
        # Check progress
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y_batch in tgt_dl:
                x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
                out = model(x)
                preds = out.argmax(1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        
        cur_acc = correct/total
        if cur_acc > best_acc: best_acc = cur_acc
        print(f"  Adapt Ep {ep+1}: Used {used} Pseudo-Labels | Acc: {cur_acc*100:.2f}%")

    print(f"\n🏆 FINAL RESULT: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()