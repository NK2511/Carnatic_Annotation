"""
Abhaswara vs Swara Dataset Generator.
Extracts segments from all Ragas and labels them as 1 (Valid Swara) or 0 (Abhaswara).
Input: carva_*.csv (Normalized segments).
"""

import pandas as pd
import numpy as np
import ast
import sys
import os
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carnatic_functions import CARNATIC_RATIOS

# ==========================
# CONFIG
# ==========================
BASE_DIR = r"C:\Desktop\Python\CarnaticAnnotater"
OUTPUT_FILE = r"C:\Desktop\Python\CarnaticAnnotater\VocalAnnotator\Method_Jaccard_CNN\abhaswara_dataset.pth"

# Ragas & Their Valid Swaras
RAGA_SIGNATURES = {
    "Keeravani":        {'Sa', 'Ri2', 'Ga2', 'Ma1', 'Pa', 'Da1', 'Ni3'},
    "Shankarabharanam": {'Sa', 'Ri2', 'Ga3', 'Ma1', 'Pa', 'Da2', 'Ni3'},
    "Kalyani":          {'Sa', 'Ri2', 'Ga3', 'Ma2', 'Pa', 'Da2', 'Ni3'},
    "Kharaharapriya":   {'Sa', 'Ri2', 'Ga2', 'Ma1', 'Pa', 'Da2', 'Ni2'},
    "Shanmukhapriya":   {'Sa', 'Ri2', 'Ga2', 'Ma2', 'Pa', 'Da1', 'Ni2'},
    "Mayamalavagowlai": {'Sa', 'Ri1', 'Ga3', 'Ma1', 'Pa', 'Da1', 'Ni3'},
}

SWARAS = ['Sa', 'Ri1', 'Ri2', 'Ga2', 'Ga3', 'Ma1', 'Ma2', 'Pa', 'Da1', 'Da2', 'Ni2', 'Ni3']
TOLERANCE_CENTS = 50 # Expanded from 30 to capture Gamakas
TOLERANCE_RATIO = 2 ** (TOLERANCE_CENTS / 1200.0)
FIXED_LENGTH = 60 # Interpolate to 60 points

def main():
    print("🚀 Generating Abhaswara Dataset...")
    
    all_segments = []
    all_labels = [] # 1=Swara, 0=Abhaswara
    all_metadata = [] # (Raga, Note)
    
    for raga, valid_notes in RAGA_SIGNATURES.items():
        if "Thodi" in raga: continue
        
        print(f"\n📂 Processing {raga}...")
        csv_path = os.path.join(BASE_DIR, "Raagas", raga, f"{raga}_Data", f"carva_{raga}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ CSV not found: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        
        # Determine Raga's Valid/Invalid Map
        is_valid = {note: (note in valid_notes) for note in SWARAS}
        
        # Process each note type
        for note in SWARAS:
            segments = extract_segments(df, note)
            label = 1 if is_valid[note] else 0
            
            # Subsample if too many valid notes?
            # Or keep all to show dominance.
            # But class imbalance will be huge (Valid >> Invalid).
            # We might need to balance later or use weighted loss.
            # For now, collect all.
            
            for seg in segments:
                all_segments.append(seg)
                all_labels.append(label)
                all_metadata.append((raga, note))
                
            print(f"   {note}: {len(segments)} segments ({'✅ Swara' if label else '❌ Abhaswara'})")

    # Convert to Tensor
    print("\n📦 Packaging Dataset...")
    X = np.array(all_segments, dtype=np.float32) # Shape (N, 60)
    y = np.array(all_labels, dtype=np.int64)     # Shape (N,)
    
    # Metadata as pandas
    meta_df = pd.DataFrame(all_metadata, columns=['Raga', 'Note'])
    
    # Check Balance
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    print(f"   Total Samples: {len(X)}")
    print(f"   Valid Swaras: {n_pos} ({n_pos/len(X)*100:.1f}%)")
    print(f"   Abhaswaras: {n_neg} ({n_neg/len(X)*100:.1f}%)")
    
    # Save
    torch.save({
        'segments': torch.from_numpy(X),
        'labels': torch.from_numpy(y),
        'metadata': meta_df
    }, OUTPUT_FILE)
    print(f"✅ Saved to {OUTPUT_FILE}")

def extract_segments(df, note_name):
    if note_name not in CARNATIC_RATIOS: return []
    base_ratio = CARNATIC_RATIOS[note_name]
    while base_ratio >= 2.0: base_ratio /= 2.0
    while base_ratio < 1.0: base_ratio *= 2.0
    
    extracted = []
    target_ratios = [base_ratio * 0.5, base_ratio, base_ratio * 2.0]
    
    for _, row in df.iterrows():
        try:
            full_seg = np.array(ast.literal_eval(row['SegmentList']), dtype=float)
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
                    
                    # Normalize & Interpolate
                    avg_r = np.mean(sub_seg)
                    # Find octave target
                    if avg_r < 0.8 * base_ratio: target = base_ratio * 0.5
                    elif avg_r > 1.8 * base_ratio: target = base_ratio * 2.0
                    else: target = base_ratio
                    
                    # Log Cents (Relative to Target Note)
                    cents = 1200 * np.log2(sub_seg / target)
                    
                    # Interpolate to 60
                    x_old = np.linspace(0, 1, len(cents))
                    x_new = np.linspace(0, 1, FIXED_LENGTH)
                    f = interp1d(x_old, cents, kind='linear')
                    y_interp = f(x_new)
                    
                    extracted.append(y_interp)
        except: continue
        
    return extracted

if __name__ == "__main__":
    main()
