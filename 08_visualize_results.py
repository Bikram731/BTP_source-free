import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# --- CONFIG ---
DATA_PATH = "./data/target_vibration_massive.csv"
MODEL_PATH = "best_emergency_model.pt"  # Your 91.98% model
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Healthy", "Chipped", "Crack", "Missing"] # The 4 classes

# --- 1. MODEL & PREPROCESSING (Must match Training Script) ---
def apply_fft(signal):
    fft_vals = np.fft.rfft(signal)
    fft_abs = np.abs(fft_vals)
    fft_abs = (fft_abs - fft_abs.min()) / (fft_abs.max() - fft_abs.min() + 1e-6)
    return fft_abs.astype("float32")

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
        return logits, feat # Return features for t-SNE

class FFTData(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(apply_fft(self.X[idx])[None, :]), torch.tensor(self.y[idx])

def main():
    print("--- 📊 GENERATING VISUALS (WITHOUT RETRAINING) ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH, header=None)
        X = df.iloc[:, :-1].values.astype("float32")
        y = df.iloc[:, -1].astype(int).values
    except:
        print("❌ Data not found."); return

    # Split exactly as we did in training to evaluate the Test Set
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Determine input size from first sample
    sample_feat = apply_fft(X_te[0])
    input_len = len(sample_feat)
    
    test_dl = DataLoader(FFTData(X_te, y_te), batch_size=BATCH_SIZE)
    
    # 2. Load Model
    model = EmergencyModel(input_len, 4).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model loaded successfully.")
    except:
        print("❌ Model file not found. Did you rename it?"); return

    model.eval()
    all_preds = []
    all_trues = []
    all_feats = []

    print("running Inference...")
    with torch.no_grad():
        for x, y_batch in test_dl:
            x = x.to(DEVICE)
            logits, feats = model(x)
            preds = logits.argmax(1).cpu().numpy()
            
            all_preds.extend(preds)
            all_trues.extend(y_batch.numpy())
            all_feats.extend(feats.cpu().numpy())

    # --- 3. GENERATE REPORT (Like Rival's Screenshot) ---
    print("\n" + "="*40)
    print("🏆 YOUR CLASSIFICATION REPORT")
    print("="*40)
    print(classification_report(all_trues, all_preds, target_names=CLASSES, digits=4))
    
    # --- 4. GENERATE CONFUSION MATRIX ---
    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Acc: {np.mean(np.array(all_preds)==np.array(all_trues))*100:.2f}%)')
    plt.savefig("01_confusion_matrix.png")
    print("✅ Saved '01_confusion_matrix.png'")

    # --- 5. GENERATE t-SNE PLOT (The "Pro" Graph) ---
    # This visualizes how well the model separates the classes
    print("Generating t-SNE plot (this might take a moment)...")
    # Take a subset if data is too huge (e.g., 2000 points) to speed up
    subset_idx = np.random.choice(len(all_feats), min(len(all_feats), 2000), replace=False)
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(np.array(all_feats)[subset_idx])
    y_subset = np.array(all_trues)[subset_idx]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_subset, cmap='jet', alpha=0.6)
    plt.legend(handles=scatter.legend_elements()[0], labels=CLASSES)
    plt.title("t-SNE Visualization of Feature Space")
    plt.savefig("02_tsne_plot.png")
    print("✅ Saved '02_tsne_plot.png'")

if __name__ == "__main__":
    main()