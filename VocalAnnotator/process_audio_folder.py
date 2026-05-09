import os
import sys
import shutil
import subprocess
import argparse
import pandas as pd
import numpy as np
import librosa
import crepe
from tqdm import tqdm
import torch

# ==========================
# CONFIGURATION
# ==========================

try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except Exception:
    DEVICE = "cpu"

print(f"🖥️  Audio Processing Device: {DEVICE.upper()}")

# Path to the Python environment that has Demucs installed                   
DEMUCS_PYTHON = r"C:\Desktop\Python\CarnaticAnnotater\demucs_env\Scripts\python.exe"
# Ensure the Scripts folder (where ffmpeg might be) is in PATH for child process
os.environ["PATH"] = os.path.dirname(DEMUCS_PYTHON) + os.pathsep + os.environ["PATH"]

# CREPE Configuration
CREPE_MODEL = 'tiny' # 'tiny', 'small', 'medium', 'large', 'full'
CREPE_STEP_SIZE = 20 # milliseconds
CREPE_VITERBI = True

# ==========================
# UTILS
# ==========================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 Created directory: {path}")

def get_demucs_command():
    return [DEMUCS_PYTHON, "-m", "demucs.separate"]

# ==========================
# PHASE 1: VOCAL EXTRACTION
# ==========================

def run_demucs_extraction(songs_dir, vocals_dir):
    print(f"\n🎵 PHASE 1: Extracting Vocals from {songs_dir}...")
    ensure_dir(vocals_dir)

    wav_files = [f for f in os.listdir(songs_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not wav_files:
        print("⚠️ No .wav files found in Songs directory.")
        return

    demucs_cmd = get_demucs_command()
    
    for wav_file in tqdm(wav_files, desc="Separating Vocals"):
        base_name = os.path.splitext(wav_file)[0]
        input_path = os.path.join(songs_dir, wav_file)
        
        # Expected output filename matches user convention: Name_vocals.wav
        # Check if ANY file in vocals directory starts with "{base_name}_vocals"
        # This handles "Song_vocals.wav" AND "Song_vocals_123_45_C#3.wav"
        existing_candidates = [
            f for f in os.listdir(vocals_dir) 
            if f.startswith(f"{base_name}_vocals") and f.lower().endswith('.wav')
        ]

        if existing_candidates:
            # print(f"Skipping {base_name}: Found {existing_candidates[0]}")
            continue # Skip existing
            
        final_vocal_path = os.path.join(vocals_dir, f"{base_name}_vocals.wav")

        # Temp output for Demucs structure
        temp_out = os.path.join(vocals_dir, "__demucs_tmp__")
        
        cmd = demucs_cmd + [
            "-n", "htdemucs_ft",
            "--two-stems=vocals",
            "-d", DEVICE,
            "-o", temp_out,
            input_path,
        ]

        try:
            # Run Demucs with real-time output monitoring
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True, 
                bufsize=1, 
                universal_newlines=True
            )
            
            # Print output in real-time
            for line in process.stdout:
                print(line, end='')
                
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            # Locate output: temp_out / htdemucs / <song_name> / vocals.wav
            demucs_output_path = os.path.join(temp_out, "htdemucs", base_name, "vocals.wav")
            
            if os.path.exists(demucs_output_path):
                shutil.move(demucs_output_path, final_vocal_path)
            else:
                print(f"❌ Error: Demucs did not produce output for {wav_file}")
        except Exception as e:
            print(f"❌ Failed to process {wav_file}: {e}")
        finally:
            # Cleanup temp folder for this song
            shutil.rmtree(temp_out, ignore_errors=True)

    print("✅ Vocal extraction complete.")

# ==========================
# PHASE 2: CREPE PITCH EXTRACTION
# ==========================

def run_crepe_analysis(vocals_dir, csv_dir):
    print(f"\n📈 PHASE 2: Generating Pitch CSVs in {csv_dir}...")
    ensure_dir(csv_dir)

    vocal_files = [f for f in os.listdir(vocals_dir) if f.lower().endswith('.wav')]
    if not vocal_files:
        print("⚠️ No vocal files found to process.")
        return

    for vocal_file in tqdm(vocal_files, desc="Running CREPE"):
        base_name = os.path.splitext(vocal_file)[0]

        
        csv_name = base_name + ".csv"
        csv_path = os.path.join(csv_dir, csv_name)
        
        # Similar check for CSVs
        # If we have "Song_vocals.csv" OR "Song_vocals_123_45_C#3.csv"
        existing_csvs = [
             f for f in os.listdir(csv_dir)
             if f.startswith(base_name) and f.lower().endswith('.csv')
        ]
        
        if existing_csvs:
            continue

        audio_path = os.path.join(vocals_dir, vocal_file)
        
        try:
            # Load Audio
            y, sr = librosa.load(audio_path, sr=16000) # CREPE likes 16k
            
            # Predict
            time, frequency, confidence, _ = crepe.predict(
                y, sr, 
                viterbi=CREPE_VITERBI, 
                step_size=CREPE_STEP_SIZE, 
                model_capacity=CREPE_MODEL,
                verbose=0
            )
            
            # Save CSV
            df = pd.DataFrame({
                "time": time,
                "frequency": frequency,
                "confidence": confidence
            })
            
            df.to_csv(csv_path, index=False)
            
        except Exception as e:
            print(f"❌ Error processing {vocal_file}: {e}")

    print("✅ CREPE analysis complete.")


# ==========================
# SINGLE FILE PROCESSING
# ==========================

def process_single_file(file_path):
    print(f"🎵 Processing Single File: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # Determine directories
    # Place 'Vocals' and 'CSVs' in the same parent dir as the file
    parent_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create output folders 
    vocals_dir = os.path.join(parent_dir, "Processed_Vocals")
    csv_dir = os.path.join(parent_dir, "Processed_CSVs")
    
    ensure_dir(vocals_dir)
    ensure_dir(csv_dir)

    # --- Phase 1: Demucs ---
    print(f"\n--- Phase 1: Demucs Extraction ---")
    demucs_cmd = get_demucs_command()
    final_vocal_path = os.path.join(vocals_dir, f"{base_name}_vocals.wav")
    
    if not os.path.exists(final_vocal_path):
        temp_out = os.path.join(vocals_dir, f"temp_{base_name}")
        
        cmd = demucs_cmd + [
            "-n", "htdemucs_ft",
            "--two-stems=vocals",
            "-d", DEVICE,
            "-o", temp_out,
            file_path,
        ]
        
        try:
            print(f"Executing Demucs...")
            # Capture output to debug
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            found_vocal = None
            for root, dirs, files in os.walk(temp_out):
                if "vocals.wav" in files:
                    found_vocal = os.path.join(root, "vocals.wav")
                    break
            
            if found_vocal:
                shutil.move(found_vocal, final_vocal_path)
                print(f"✅ Vocals extracted: {final_vocal_path}")
            else:
                print(f"❌ Demucs finished but 'vocals.wav' not found in temp output.")
                print(f"Stdout: {result.stdout}")
                print(f"Stderr: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Demucs Failed: {e}")
            print(f"Error Output:\n{e.stderr}")
        finally:
            shutil.rmtree(temp_out, ignore_errors=True)
    else:
        print(f"✅ Vocals already exist: {final_vocal_path}")

    # --- Phase 2: CREPE ---
    print(f"\n--- Phase 2: CREPE Pitch Extraction ---")
    csv_name = f"{base_name}.csv"
    csv_path = os.path.join(csv_dir, csv_name)
    
    if os.path.exists(csv_path):
        print(f"✅ CSV already exists: {csv_path}")
    elif os.path.exists(final_vocal_path):
        try:
            y, sr = librosa.load(final_vocal_path, sr=16000)
            print("Audio loaded. Running CREPE...")
            time, frequency, confidence, _ = crepe.predict(
                y, sr, 
                viterbi=CREPE_VITERBI, 
                step_size=CREPE_STEP_SIZE, 
                model_capacity=CREPE_MODEL,
                verbose=1
            )
            df = pd.DataFrame({"time": time, "frequency": frequency, "confidence": confidence})
            df.to_csv(csv_path, index=False)
            print(f"✅ CSV saved: {csv_path}")
        except Exception as e:
            print(f"❌ CREPE Failed: {e}")
    else:
        print("❌ Cannot run CREPE: Vocal file missing.")

# ==========================
# MAIN
# ==========================

def main():
    print("=== Vocal Annotator Pipeline ===")
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = input("Enter Path (Folder Name or File Path): ").strip()
        
    if not input_path:
        print("❌ Invalid Input.")
        return

    # Handle Input
    if os.path.isfile(input_path):
        # Single File Mode
        process_single_file(input_path)
    else:
        # Folder Mode (Original Logic)
        if os.path.isabs(input_path):
            base_path = input_path
            raga_name_only = os.path.basename(os.path.normpath(input_path))
        else:
            base_path = os.path.abspath(input_path)
            raga_name_only = input_path # Rough assumption

        print(f"📂 Processing Root: {base_path}")

        # Define Subfolders
        songs_dir = os.path.join(base_path, f"{raga_name_only}_Songs")
        vocals_dir = os.path.join(base_path, f"{raga_name_only}_Vocals")
        csv_dir = os.path.join(base_path, f"{raga_name_only}_CSVs")

        if not os.path.exists(songs_dir):
            # Fallback: maybe the input IS the songs directory?
            if os.path.isdir(base_path) and "Songs" not in base_path:
                 if base_path.endswith("_Songs"):
                     songs_dir = base_path
                     parent = os.path.dirname(base_path)
                     pass 
            
            if not os.path.exists(songs_dir):
                print(f"❌ Songs directory not found: {songs_dir}")
                print(f"   Expected structure: {raga_name_only}/{raga_name_only}_Songs")
                return

        # Execute
        run_demucs_extraction(songs_dir, vocals_dir)
        run_crepe_analysis(vocals_dir, csv_dir)
    
    print("\n🎉 Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()
