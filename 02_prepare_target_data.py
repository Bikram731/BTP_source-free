import os
import glob
import pandas as pd
import numpy as np
import scipy.io
import warnings

# --- CONFIGURATION ---
CFG = {
    # ✅ CORRECT PATH based on your screenshot
    "MAT_FILE_DIR": "./data/",   
    "OUT_DIR": "./data/",
    "WINDOW_SIZE": 2000, 
    "STEP": 200,  # Overlap to get more data
}

warnings.filterwarnings("ignore")

def get_true_label_from_filename(filename):
    fname = os.path.basename(filename).lower()
    
    # --- 1. STRICT FILTER: MUST BE A VIBRATION FILE ---
    if "vibration" not in fname:
        print(f"  ⛔ Skipping Non-Vibration File: {fname}")
        return -1
        
    # --- 2. ASSIGN LABELS ---
    if "healthy" in fname: return 0
    elif "chipped" in fname: return 1
    elif "crack" in fname: return 2
    elif "miss" in fname: return 3
    
    return -1

def main():
    print(f"--- 🛠️ PREPARING TARGET DATA (VIBRATION ONLY) ---")
    
    all_data = []
    all_labels = []
    
    # Search for .mat files in the specific folder
    search_path = os.path.join(CFG["MAT_FILE_DIR"], "*.mat")
    mat_files = glob.glob(search_path)
    
    if not mat_files:
        print(f"❌ CRITICAL ERROR: No .mat files found in {CFG['MAT_FILE_DIR']}")
        return

    print(f"Found {len(mat_files)} files. Filtering...")

    for file_path in mat_files:
        true_label = get_true_label_from_filename(file_path)
        
        # If it's not a vibration file or label is unknown, skip
        if true_label == -1: 
            continue 
            
        try:
            mat = scipy.io.loadmat(file_path)
            found_data = False
            
            # Find the data matrix inside the .mat file
            for key in mat.keys():
                if key.startswith('__'): continue
                data = mat[key]
                
                # We expect 4 columns of vibration data
                if data.ndim == 2 and data.shape[1] >= 4:
                    print(f"  ✅ Processing: {os.path.basename(file_path)} (Label: {true_label})")
                    
                    # EXTRACT ALL 4 COLUMNS
                    for col_idx in range(4):
                        signal = data[:, col_idx].flatten()
                        
                        # Create Sliding Windows
                        num_windows = (len(signal) - CFG["WINDOW_SIZE"]) // CFG["STEP"]
                        if num_windows <= 0: continue
                        
                        # Vectorized slicing
                        idx = np.arange(CFG["WINDOW_SIZE"])[None, :] + np.arange(num_windows)[:, None] * CFG["STEP"]
                        windows = signal[idx]
                        
                        # Normalize (Standardization)
                        means = windows.mean(axis=1, keepdims=True)
                        stds = windows.std(axis=1, keepdims=True)
                        windows = (windows - means) / (stds + 1e-6)
                        
                        all_data.append(windows)
                        all_labels.extend([true_label] * len(windows))
                    found_data = True
                    break 
            
            if not found_data:
                print(f"  ⚠️ Warning: {os.path.basename(file_path)} has no valid data matrix.")
                
        except Exception as e:
            print(f"  Error reading {os.path.basename(file_path)}: {e}")

    # --- SAVE ---
    if len(all_data) > 0:
        final_X = np.vstack(all_data).astype("float32")
        final_y = np.array(all_labels).astype(int)
        
        out_filename = "target_vibration_massive.csv"
        out_path = os.path.join(CFG["OUT_DIR"], out_filename)
        
        df = pd.DataFrame(final_X)
        df['label'] = final_y
        df.to_csv(out_path, index=False, header=False)
        
        print(f"\n✅ SUCCESS!")
        print(f"   Saved cleaned dataset to: {out_filename}")
        print(f"   Total Samples: {len(df)}")
    else:
        print("\n❌ ERROR: No data processed. Check the folder path again.")

if __name__ == "__main__":
    main()