import os
import sys
import pandas as pd
import numpy as np
import re
import glob
from pathlib import Path
from tqdm import tqdm

# Add current dir to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from carnatic_functions import get_raaga_context, clean_np_float_list

# Configuration
BASE_DIR = r"C:\Desktop\Python\CarnaticAnnotater\Kharaharapriya"
AUDIO_DIR = os.path.join(BASE_DIR, "Kharaharapriya_CSVs")
DATA_DIR = os.path.join(BASE_DIR, "Kharaharapriya_Data")
OUTPUT_CSV = os.path.join(DATA_DIR, "crepe_Kharaharapriya.csv")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def consolidate_crepe_files(audio_dir):
    # Ignoring context auto-guess for specific path control
    raaga_name = "Mayamalavagowlai"
    output_csv = OUTPUT_CSV
    
    print(f"📂 Audio Dir: {audio_dir}")
    print(f"🎯 Output Master CSV: {output_csv}")
    
    # 1. Find all pitch CSVs in the Audio Directory
    # Pattern: *_pitch.csv
    search_pattern = os.path.join(audio_dir, "*_pitch.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print("❌ No *_pitch.csv files found in the audio directory.")
        return

    # Sort naturally
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
    
    csv_files.sort(key=natural_keys)
    print(f"found {len(csv_files)} pitch files.")

    all_rows = []
    
    # Header for Master CSV
    # Index,AudioPath,Raaga,SongName,Tonic,Time,Frequency,Confidence,Tonic_Normalized_Frequency
    
    for idx, csv_path in enumerate(tqdm(csv_files, desc="Consolidating")):
        file_path = Path(csv_path)
        filename = file_path.name
        
        # Parse Filename for Metadata
        # Expected: "SongName_TonicInt_TonicDec_Note_pitch.csv"
        # Example: "Song_01_200_00_Sa_pitch.csv" -> pure guesswork usually, let's try the user's specific format
        # User format seen previously: "Song_438_44_A4_pitch.csv" -> 438.44 Hz
        
        name_parts = filename.replace("_pitch.csv", "").split("_")
        
        tonic = 200.0
        song_name = "Unknown"
        
        try:
            # Try parsing Start-End-Note format (e.g. 138_50_C#)
            # Strategy: Look for last 3 parts
            if len(name_parts) >= 4:
                # Check for "438", "44", "A4" pattern
                p_int = name_parts[-3]
                p_dec = name_parts[-2]
                
                if p_int.isdigit() and p_dec.isdigit():
                    tonic = float(f"{p_int}.{p_dec}")
                    song_name = "_".join(name_parts[:-3])
                else:
                    song_name = "_".join(name_parts)
            else:
                song_name = "_".join(name_parts)
                
        except Exception:
            song_name = "_".join(name_parts)

        # Find corresponding Audio file (WAV)
        # Assuming wav has same prefix as csv or is just the csv name without _pitch
        # Simple heuristic: look for .wav with largest overlap in name?
        # Standard convention: "SongName.wav" -> "SongName_pitch.csv" is unlikely.
        # usually "SongName.wav" -> produce "SongName_pitch.csv" ? 
        # Actually, let's assume the WAV file is "Song_Name_Parts... .wav"
        # If we can't find it, we just point to the CSV path or leave it empty? Logic needs AudioPath.
        
        # Try to find wav that matches the stem of the CSV before "_pitch"
        wav_candidate = os.path.join(audio_dir, filename.replace("_pitch.csv", ".wav"))
        if not os.path.exists(wav_candidate):
            # Try removing the tonic parts? 
            # If filename is "Song_438_44_A4_pitch.csv", maybe wav is "Song.wav"? 
            # Or "Song_438_44_A4.wav"?
            wav_candidate_2 = os.path.join(audio_dir, filename.replace("_pitch.csv", "") + ".wav")
            if os.path.exists(wav_candidate_2):
                wav_candidate = wav_candidate_2
            else:
                # Check if there is a file that matches the song name part
                pass

        # Load CSV
        try:
            df = pd.read_csv(csv_path)
            
            # Normalize columns
            df.columns = df.columns.str.strip().str.lower()
            
            if 'frequency' not in df.columns:
                print(f"Skipping {filename}: No 'frequency' column.")
                continue
                
            freqs = df['frequency'].values
            
            # Time: synthesize if missing (assuming 10ms or 20ms?)
            # CREPE standard is 10ms usually. 
            if 'time' in df.columns:
                times = df['time'].values
            else:
                times = np.arange(len(freqs)) * 0.01
                
            conf = df['confidence'].values if 'confidence' in df.columns else np.ones_like(freqs)
            
            # Normalize
            norm_freqs = freqs / tonic
            
            # Build Rows
            # We want to append to a list, then create DataFrame once for speed
            # But the structure is flat.
            
            # Efficient method: Create DF and append to output file incrementally
            song_df = pd.DataFrame({
                "Index": idx,
                "AudioPath": wav_candidate,
                "Raaga": raaga_name,
                "SongName": song_name,
                "Tonic": tonic,
                "Time": times,
                "Frequency": freqs,
                "Confidence": conf,
                "Tonic_Normalized_Frequency": norm_freqs
            })
            
            # Append mode with header only first time
            header = (idx == 0)
            mode = 'w' if idx == 0 else 'a'
            song_df.to_csv(output_csv, mode=mode, index=False, header=header)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("✅ Consolidation Complete.")

if __name__ == "__main__":
    consolidate_crepe_files(AUDIO_DIR)
