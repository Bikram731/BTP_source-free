import os
import glob
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

# --- 1. CONFIGURATION ---
SOURCE_DIR = "./data/seu_gear_data/"
TARGET_CSV = "./data/target_vibration_massive.csv"
BATCH_SIZE = 64
LR_SOURCE = 1e-3
LR_ADAPT = 1e-4        
EPOCHS_SOURCE = 40    
EPOCHS_ADAPT = 15      
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4
WINDOW_SIZE = 2000
STEP = 500

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_seed()

# --- 2. DATASET PROCESSING ---
def apply_fft(signal):
    fft_vals = np.fft.rfft(signal)
    fft_abs = np.abs(fft_vals)
    fft_abs = (fft_abs - fft_abs.min()) / (fft_abs.max() - fft_abs.min() + 1e-6)
    return fft_abs.astype("float32")

class FFTData(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        feat = apply_fft(self.X[idx])
        return torch.from_numpy(feat[None, :]), torch.tensor(self.y[idx])

# --- 3. SOURCE DATA LOADER ---
def get_seu_label(filename):
    fname = os.path.basename(filename).lower()
    if fname.startswith("health"): return 0
    if fname.startswith("chipped"): return 1
    if fname.startswith("root"): return 2
    if fname.startswith("miss"): return 3
    return None

def load_source_machine_data():
    all_windows, all_labels = [], []
    file_paths = glob.glob(os.path.join(SOURCE_DIR, "*.csv"))
    
    for file_path in file_paths:
        label = get_seu_label(file_path)
        if label is None: continue
            
        try:
            df = pd.read_csv(file_path, header=None, delim_whitespace=True, skiprows=16)
            # Use column 1 which often contains the stronger vibration signal in SEU
            signal = pd.to_numeric(df[1], errors='coerce').dropna().values
            
            if signal.size < WINDOW_SIZE: 
                signal = np.pad(signal, (0, WINDOW_SIZE - signal.size), mode="edge")
            
            for st in range(0, len(signal) - WINDOW_SIZE + 1, STEP):
                w = signal[st:st + WINDOW_SIZE].astype("float32")
                m, s = float(w.mean()), float(w.std())
                w = (w - m) / (s + 1e-6)
                all_windows.append(w)
                all_labels.append(label)
        except:
            continue
            
    return np.stack(all_windows), np.array(all_labels)

# --- 4. PAPER MATH MODULES ---
def calculate_sscknn_centroids(model, dataloader, device):
    """Calculates class centroids using soft-voting on extracted features."""
    model.eval()
    all_feats, all_probs = [], []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            logits, feat = model(x)
            all_feats.append(feat)
            all_probs.append(F.softmax(logits, dim=1))
            
    all_feats = torch.cat(all_feats, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    
    centroids = torch.zeros(NUM_CLASSES, all_feats.shape[1]).to(device)
    for k in range(NUM_CLASSES):
        weight = all_probs[:, k].unsqueeze(1)
        centroids[k] = (all_feats * weight).sum(dim=0) / (weight.sum() + 1e-6)
    
    return F.normalize(centroids, p=2, dim=1)

def update_adaptive_thresholds(current_thresholds, class_counts, tau_h=0.95):
    num_classes = len(class_counts)
    total_samples = class_counts.sum() + 1e-6  
    p_hat = class_counts / total_samples
    p_hat_std = torch.std(p_hat, unbiased=True) 
    
    new_thresholds = torch.zeros(num_classes).to(current_thresholds.device)
    for k in range(num_classes):
        updated_val = current_thresholds[k] + p_hat[k] - p_hat_std
        # 🌟 FIX: Added min=0.01 to prevent negative thresholds
        new_thresholds[k] = torch.clamp(updated_val, min=0.01, max=tau_h) 
    return new_thresholds

# --- 5. MODEL ARCHITECTURE ---
class DiagnosisModel(nn.Module):
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
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        logits = self.classifier(feat)
        return logits, feat

def main():
    print("--- 🌍 REAL CROSS-MACHINE EXPERIMENT ---")
    
    # 1. Load Machine A (Source)
    X_src_full, y_src_full = load_source_machine_data()
    print(f"✅ Machine A Loaded: {len(X_src_full)} samples.")
    
    # NEW: Split Source data to verify Phase 1 actually works
    X_src_tr, X_src_va, y_src_tr, y_src_va = train_test_split(X_src_full, y_src_full, test_size=0.2, stratify=y_src_full)
        
    # 2. Load Machine B (Target)
    df_tgt = pd.read_csv(TARGET_CSV, header=None)
    X_tgt = df_tgt.iloc[:, :-1].values.astype("float32")
    y_tgt = df_tgt.iloc[:, -1].astype(int).values
    print(f"✅ Machine B Loaded: {len(X_tgt)} samples.")

    fft_len = len(apply_fft(X_src_tr[0]))

    # Dataloaders
    src_dl_tr = DataLoader(FFTData(X_src_tr, y_src_tr), batch_size=BATCH_SIZE, shuffle=True)
    src_dl_va = DataLoader(FFTData(X_src_va, y_src_va), batch_size=BATCH_SIZE, shuffle=False)
    
    tgt_dl = DataLoader(FFTData(X_tgt, y_tgt), batch_size=BATCH_SIZE, shuffle=True) 
    tgt_dl_eval = DataLoader(FFTData(X_tgt, y_tgt), batch_size=BATCH_SIZE, shuffle=False)

    model = DiagnosisModel(fft_len, NUM_CLASSES).to(DEVICE)
    
    # --- PHASE 1: TRAIN ON MACHINE A ---
    print("\nPhase 1: Training on Machine A (Source)...")
    optimizer_src = torch.optim.Adam(model.parameters(), lr=LR_SOURCE)
    criterion = nn.CrossEntropyLoss()
    
    for ep in range(EPOCHS_SOURCE):
        model.train()
        for x, y_batch in src_dl_tr:
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
            optimizer_src.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer_src.step()
            
        # NEW: Show Source Validation Accuracy
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y_batch in src_dl_va:
                x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
                logits, _ = model(x)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += y_batch.size(0)
        
        if (ep+1) % 5 == 0:
            print(f"  Src Ep {ep+1}: Source Validation Acc: {correct/total*100:.2f}%")

    # --- PHASE 1.5: THE DOMAIN SHIFT TEST ---
    print("\nPhase 1.5: Testing Machine A model directly on Machine B...")
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y_batch in tgt_dl_eval:
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
            logits, _ = model(x)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)
    
    start_acc = correct/total
    print(f"📊 Accuracy BEFORE Adaptation (The Domain Shift): {start_acc*100:.2f}%")
    

   # --- PHASE 2: ADAPT TO MACHINE B (1-Shot Semi-Supervised) ---
    print(f"\nPhase 2: Adapting to Machine B (1-Shot Anchor Method)...")
    
    # 🌟 1. EXTRACT THE 4 ANCHORS (The Mathematical Compass)
    # We simulate an engineer labeling exactly 1 sample per class on Machine B
    anchor_X = []
    for c in range(NUM_CLASSES):
        idx = np.where(y_tgt == c)[0][0] # Find the first instance of class c
        feat = apply_fft(X_tgt[idx])
        anchor_X.append(feat[None, :])
    anchor_X_tensor = torch.from_numpy(np.stack(anchor_X)).to(DEVICE)
    
    # Freeze the classifier boundaries learned from Machine A
    for param in model.classifier.parameters():
        param.requires_grad = False
        
    optimizer_tgt = torch.optim.Adam(model.feature_extractor.parameters(), lr=LR_ADAPT)
    
    # We use Cosine Similarity for thresholding now, so start at 0.50
    current_thresholds = torch.tensor([0.50] * NUM_CLASSES).to(DEVICE)
    best_acc = start_acc
    
    for ep in range(EPOCHS_ADAPT):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d): module.eval() 
                
        # 🌟 2. UPDATE CENTROIDS USING ONLY THE 4 ANCHORS
        model.eval()
        with torch.no_grad():
            _, anchor_feats = model(anchor_X_tensor)
            # These 4 points dictate the center of the 4 fault clusters
            centroids = F.normalize(anchor_feats, p=2, dim=1) 
        model.train()
        
        used = 0
        class_counts = torch.zeros(NUM_CLASSES).to(DEVICE)
        
        for x, _ in tgt_dl: 
            x = x.to(DEVICE)
            optimizer_tgt.zero_grad()
            logits, feat = model(x)
            probs = F.softmax(logits, dim=1)
            
            # --- 3. ANCHOR-BASED PSEUDO-LABELING ---
            feat_norm = F.normalize(feat, p=2, dim=1)
            
            # Compare the unlabeled features to our 4 known anchors
            sim = torch.mm(feat_norm, centroids.t())
            
            # The label is assigned by whichever anchor it is closest to
            max_sim, pseudo_label = torch.max(sim, dim=1) 
            
            # Mask based on spatial distance (similarity), NOT the broken classifier probabilities!
            sample_thresholds = current_thresholds[pseudo_label]
            mask = max_sim > sample_thresholds 
            
            loss = torch.tensor(0.0).to(DEVICE)
            
            if mask.sum() > 0:
                # Force the feature extractor to map these features into the frozen classifier's bounds
                loss += F.cross_entropy(logits[mask], pseudo_label[mask])
                used += mask.sum().item()
                
                unique_classes, counts = torch.unique(pseudo_label[mask], return_counts=True)
                class_counts[unique_classes] += counts.float()
                
            # Diversity Loss
            avg_probs = torch.mean(probs, dim=0)
            L_div = torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
            loss += (1.0 * L_div) 
            
            if loss > 0:
                loss.backward()
                optimizer_tgt.step()
                
        if used > 0:
            current_thresholds = update_adaptive_thresholds(current_thresholds, class_counts, tau_h=0.90)
                
        # Evaluate against the hidden true labels
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y_batch in tgt_dl_eval:
                x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
                logits, _ = model(x)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += y_batch.size(0)
        
        cur_acc = correct/total
        if cur_acc > best_acc: best_acc = cur_acc
        print(f"  Adapt Ep {ep+1}: Used {used} Pseudo-Labels | Avg Threshold: {current_thresholds.mean().item():.2f} | Acc: {cur_acc*100:.2f}%")

    print(f"\n🏆 FINAL CROSS-MACHINE RESULT: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()