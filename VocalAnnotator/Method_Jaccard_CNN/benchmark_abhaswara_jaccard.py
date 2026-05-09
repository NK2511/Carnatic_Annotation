import pandas as pd
import numpy as np
import ast
import sys
import os
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from tqdm import tqdm
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carnatic_functions import CARNATIC_RATIOS
from melakarta_signatures import MELAKARTA_RAGAS

# ==========================
# CONFIGURATION
# ==========================
MODEL_FILE = r"C:\Desktop\Python\CarnaticAnnotater\VocalAnnotator\Method_Jaccard_CNN\abhaswara_model.pth"
SWARAS = ['Sa', 'Ri1', 'Ri2', 'Ga2', 'Ga3', 'Ma1', 'Ma2', 'Pa', 'Da1', 'Da2', 'Ni2', 'Ni3']
TOLERANCE_CENTS = 50
TOLERANCE_RATIO = 2 ** (TOLERANCE_CENTS / 1200.0)
FIXED_LENGTH = 60
BASE_DIR = r"C:\Desktop\Python\CarnaticAnnotater"

# Mapping Ground Truth (Folder Names) to Melakarta Dictionary Keys
RAGA_NAME_MAPPING = {
    "Kalyani": "Kalyani",
    "Keeravani": "Keeravani",
    "Kharaharapriya": "Kharaharapriya",
    "Mayamalavagowlai": "Mayamalavagowlai",
    "Panthuvarali": "Pantuvarali",  # Fixed the spelling mismatch causing your KeyError
    "Shankarabharanam": "Shankarabharanam",
    "Shanmukhapriya": "Shanmukhapriya",
    "Thodi": "Thodi"
}

BASE_8_RAGAS = list(RAGA_NAME_MAPPING.keys())
SONGS_PER_RAGA = 15  # Total n=120 as per paper

class StabilityCNN(nn.Module):
    def __init__(self):
        super(StabilityCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 15, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_purvanga(signature):
    """Extracts Ri, Ga, Ma combination to identify the Raga's Chakra/Family."""
    return set(n for n in signature if n.startswith('Ri') or n.startswith('Ga') or n.startswith('Ma'))

def predict_song_cnn(df, model, device, restrict_to_8=False):
    detected_scale = set(['Sa', 'Pa'])
    
    # Extract Stable Notes via CNN
    for note in SWARAS:
        base_ratio = CARNATIC_RATIOS[note]
        while base_ratio >= 2.0: base_ratio /= 2.0
        while base_ratio < 1.0: base_ratio *= 2.0
        
        extracted = []
        target_ratios = [base_ratio * 0.5, base_ratio, base_ratio * 2.0]
        
        for _, row in df.iterrows():
            try:
                if isinstance(row['SegmentList'], str):
                    full_seg = np.array(ast.literal_eval(row['SegmentList']), dtype=float)
                else:
                    full_seg = np.array(row['SegmentList'], dtype=float)
                
                if len(full_seg) < 5: continue
                mask = np.zeros(len(full_seg), dtype=bool)
                for tr in target_ratios:
                    lower = tr / TOLERANCE_RATIO
                    upper = tr * TOLERANCE_RATIO
                    mask |= (full_seg >= lower) & (full_seg <= upper)
                
                if not np.any(mask): continue
                indices = np.where(mask)[0]
                if len(indices) < 5: continue
                
                breaks = np.where(np.diff(indices) > 1)[0]
                splits = np.split(indices, breaks + 1)
                for split in splits:
                    if len(split) > 5:
                        sub_seg = full_seg[split]
                        avg_r = np.mean(sub_seg)
                        if avg_r < 0.8 * base_ratio: target = base_ratio * 0.5
                        elif avg_r > 1.8 * base_ratio: target = base_ratio * 2.0
                        else: target = base_ratio
                        cents = 1200 * np.log2(sub_seg / target)
                        x_old = np.linspace(0, 1, len(cents))
                        x_new = np.linspace(0, 1, FIXED_LENGTH)
                        f = interp1d(x_old, cents, kind='linear')
                        extracted.append(f(x_new))
            except: continue
            
        if not extracted: continue
        
        # PRE-CONVERT TO NUMPY FOR SPEED
        X_np = np.stack(extracted)
        X = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1).to(device)
        X = X / 50.0 
        
        with torch.no_grad():
            outputs = model(X)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
        if probs.ndim == 0: probs = np.array([probs])
        quality_score = np.sum(probs > 0.5) / len(extracted)
        if quality_score > 0.25:
             detected_scale.add(note)
             
    best_match = None
    best_overlap = 0
    if restrict_to_8:
        search_space = {RAGA_NAME_MAPPING[k]: MELAKARTA_RAGAS[RAGA_NAME_MAPPING[k]] for k in BASE_8_RAGAS}
    else:
        search_space = MELAKARTA_RAGAS
    
    for raga_name, signature in search_space.items():
        intersection = len(detected_scale.intersection(signature))
        union = len(detected_scale.union(signature))
        if union == 0: continue
        score = intersection / union
        if score > best_overlap:
            best_overlap = score
            best_match = raga_name
    return best_match

def main():
    print("🚀 Running Formal 120-Song Research Benchmark for Abhaswara CNN...")
    
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: Model missing at {MODEL_FILE}")
        return
        
    device = torch.device("cpu")
    model = StabilityCNN().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    
    total_songs = 0
    correct_base_8 = 0
    correct_full_72 = 0
    correct_chakra = 0
    
    random.seed(42) 
    for folder_name in BASE_8_RAGAS:
        csv_path = os.path.join(BASE_DIR, "Raagas", folder_name, f"{folder_name}_Data", f"carva_{folder_name}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(BASE_DIR, "Raagas", folder_name, f"{folder_name}_Data", "carva.csv")
            if not os.path.exists(csv_path): continue
        
        dict_raga_name = RAGA_NAME_MAPPING[folder_name]
        df = pd.read_csv(csv_path)
        grouped = list(df.groupby('AudioPath'))
        sample_size = min(len(grouped), SONGS_PER_RAGA)
        eval_batch = random.sample(grouped, sample_size)
        
        for _, group_df in tqdm(eval_batch, desc=f"Evaluating {folder_name} "):
            total_songs += 1
            pred_8 = predict_song_cnn(group_df, model, device, restrict_to_8=True)
            if pred_8 == dict_raga_name: correct_base_8 += 1
            pred_72 = predict_song_cnn(group_df, model, device, restrict_to_8=False)
            if pred_72 == dict_raga_name: correct_full_72 += 1
            if pred_72 and dict_raga_name in MELAKARTA_RAGAS:
                if get_purvanga(MELAKARTA_RAGAS[pred_72]) == get_purvanga(MELAKARTA_RAGAS[dict_raga_name]):
                    correct_chakra += 1

    print("\n" + "="*55)
    print(" 🏆 FORMAL RESEARCH METRICS SUMMARY ")
    print("="*55)
    print(f"Total Evaluated Dataset: n={total_songs} songs")
    if total_songs > 0:
        print(f"\nExperiment 1: Limited 8-Raga Search Space")
        print(f"• Overall Accuracy: {(correct_base_8/total_songs)*100:.1f}% ({correct_base_8}/{total_songs})")
        print(f"\nExperiment 2: Full 72-Melakarta Search space")
        print(f"• Top-1 Accuracy: {(correct_full_72/total_songs)*100:.1f}% ({correct_full_72}/{total_songs})")
        print(f"• Chakra/Family Recognition: {(correct_chakra/total_songs)*100:.1f}% ({correct_chakra}/{total_songs})")
    print("="*55)

if __name__ == "__main__":
    main()
