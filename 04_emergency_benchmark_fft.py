import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings

# --- CONFIG ---
TARGET_CSV = "./data/target_vibration_massive.csv"  # Pointing to your NEW massive file
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore")

# --- 1. FFT PREPROCESSING ---
def apply_fft(signal):
    # Apply Fast Fourier Transform
    fft_vals = np.fft.rfft(signal)
    fft_abs = np.abs(fft_vals)
    # Normalize
    fft_abs = (fft_abs - fft_abs.min()) / (fft_abs.max() - fft_abs.min() + 1e-6)
    return fft_abs.astype("float32")

class FFTData(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        # Convert Time-Series to Frequency Domain on the fly
        signal = self.X[idx]
        fft_features = apply_fft(signal)
        return torch.from_numpy(fft_features[None, :]), torch.tensor(self.y[idx])

# --- 2. ROBUST MODEL ---
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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

def main():
    print(f"--- 🚑 EMERGENCY BENCHMARK WITH FFT ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(TARGET_CSV, header=None)
        print(f"Loaded {len(df)} samples.")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Check classes
    y_check = df.iloc[:, -1].values
    unique_classes = np.unique(y_check)
    print(f"Classes found: {unique_classes}")
    if len(unique_classes) < 2:
        print("❌ CRITICAL ERROR: Only 1 class found. Model cannot learn.")
        return

    X = df.iloc[:, :-1].values.astype("float32")
    y = df.iloc[:, -1].astype(int).values

    # 2. Split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Setup with FFT Dataset
    train_ds = FFTData(X_tr, y_tr)
    test_ds = FFTData(X_te, y_te)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Determine input size
    sample_signal = X_tr[0]
    fft_len = len(apply_fft(sample_signal))
    print(f"Input Feature Size (Frequency Bins): {fft_len}")

    model = EmergencyModel(input_len=fft_len, num_classes=len(unique_classes)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # --- FIXED LINE BELOW (Removed verbose=True) ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print("🚀 Training Started...")
    for ep in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y_batch in train_dl:
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y_batch in test_dl:
                x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
                out = model(x)
                preds = out.argmax(1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        
        acc = correct / total
        
        # Step the scheduler
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_emergency_model.pt")
            print(f"⭐ Epoch {ep+1}: Acc {acc*100:.2f}% (Saved)")
        elif (ep+1) % 10 == 0:
            print(f"   Epoch {ep+1}: Acc {acc*100:.2f}% | Loss: {train_loss/len(train_dl):.4f}")

    print(f"\n🏆 FINAL FFT BENCHMARK ACCURACY: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()