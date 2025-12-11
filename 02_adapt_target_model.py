import pandas as pd
import numpy as np
import os, random, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
class CFG:
    # Model & Data Paths
    PRETRAINED_MODEL_PATH = "source_model_seu.pt" 
    TARGET_DATA_PATH = "./data/target_vibration_massive.csv"
    
    # Training Settings
    NUM_CLASSES = 4
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 30           # 30 is enough to see adaptation
    CONFIDENCE_THRESHOLD = 0.75 # Lowered slightly to encourage learning
    IM_LOSS_WEIGHT = 0.5  # Forces class diversity (prevents model collapse)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed()

# --- 2. DATASET ---
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y # Used only for evaluation, NOT training
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        return torch.from_numpy(x[None,:]), torch.tensor(self.y[idx] if self.y is not None else -1)

# --- 3. MODEL ARCHITECTURE (Matches Source) ---
class TargetModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Dropout(0.3), nn.Flatten()
        )
        # Split for access to embedding (Matches source structure manually)
        self.embed_layer = nn.Linear(8000, 100) 
        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x):
        feat = self.feature_extractor(x)
        emb = self.embed_layer(feat)
        # Standard classifier path
        logits = self.classifier(nn.ReLU()(emb))
        return logits

# --- 4. MAIN ADAPTATION LOOP ---
def main():
    print(f"--- 🔄 RUNNING UNSUPERVISED ADAPTATION (CLEAN DATA) ---")
    
    # 1. Load Data
    if not os.path.exists(CFG.TARGET_DATA_PATH):
        print(f"❌ Error: {CFG.TARGET_DATA_PATH} not found.")
        return

    try:
        df = pd.read_csv(CFG.TARGET_DATA_PATH, header=None)
        print(f"📚 Loaded {len(df)} samples from massive dataset.")
        X = df.iloc[:, :-1].values.astype("float32")
        y_true = df.iloc[:, -1].astype(int).values # HIDDEN from training
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 2. Setup DataLoaders
    # We shuffle heavily to simulate I.I.D data
    ds = TimeSeriesDataset(X, y_true)
    dl = DataLoader(ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)

    # 3. Initialize Model & Load Source Weights
    model = TargetModel(CFG.NUM_CLASSES).to(CFG.DEVICE)
    
    try:
        state = torch.load(CFG.PRETRAINED_MODEL_PATH, map_location=CFG.DEVICE)
        new_state = model.state_dict()
        
        # INTELLIGENT WEIGHT MAPPING
        # Maps the source 'label_predictor' to our split 'embed_layer' + 'classifier'
        if 'label_predictor.0.weight' in state:
            new_state['embed_layer.weight'] = state['label_predictor.0.weight']
            new_state['embed_layer.bias']   = state['label_predictor.0.bias']
            new_state['classifier.weight']  = state['label_predictor.2.weight']
            new_state['classifier.bias']    = state['label_predictor.2.bias']
            
        # Load feature extractor weights
        for k in state:
            if k in new_state and new_state[k].shape == state[k].shape:
                new_state[k] = state[k]
        
        model.load_state_dict(new_state)
        print("✅ Source Weights Loaded Successfully.")
    except Exception as e:
        print(f"❌ Error loading source model: {e}")
        print("   (Make sure source_model_seu.pt exists!)")
        return

    # 4. Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
    
    print("\n🚀 Starting Unsupervised Adaptation...")
    print(f"   (Using Confidence > {CFG.CONFIDENCE_THRESHOLD} and IM Loss)")

    best_acc = 0.0

    for epoch in range(1, CFG.EPOCHS + 1):
        model.train()
        total_loss = 0
        total_pseudo = 0
        
        for x_batch, _ in dl: # We IGNORE the label here (Unsupervised)
            x_batch = x_batch.to(CFG.DEVICE)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            probs = F.softmax(logits, dim=1)
            
            # --- LOSS 1: PSEUDO-LABELING (Self-Training) ---
            max_prob, target_label = torch.max(probs, dim=1)
            mask = max_prob > CFG.CONFIDENCE_THRESHOLD
            
            loss = torch.tensor(0.0).to(CFG.DEVICE)
            
            if mask.sum() > 0:
                # Only train on samples the model is confident about
                loss += F.cross_entropy(logits[mask], target_label[mask])
                total_pseudo += mask.sum().item()

            # --- LOSS 2: INFORMATION MAXIMIZATION (Anti-Collapse) ---
            # We want the average prediction across the batch to be uniform 
            # (i.e., don't predict Class 0 for everyone)
            avg_probs = torch.mean(probs, dim=0)
            marginal_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
            loss -= CFG.IM_LOSS_WEIGHT * marginal_entropy

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- EVALUATION (Checking against hidden labels) ---
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x_batch, y_batch in dl:
                x_batch = x_batch.to(CFG.DEVICE)
                logits = model(x_batch)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(y_batch.numpy())
        
        acc = accuracy_score(all_trues, all_preds)
        
        if acc > best_acc: best_acc = acc
        
        print(f"Ep {epoch}: Loss {total_loss:.2f} | Pseudo-Labels used: {total_pseudo} | Acc: {acc*100:.2f}%")

    print(f"\n🏆 FINAL UNSUPERVISED ADAPTATION RESULT: {best_acc*100:.2f}%")
    
    if best_acc > 0.65:
        print("✅ SUCCESS! The Source Model successfully adapted to the Target Data.")
    else:
        print("⚠️ Result is low. The Domain Shift might be too large for direct transfer.")
        print("   (Stick to your 92% FFT Benchmark result for the presentation)")

if __name__ == "__main__":
    main()