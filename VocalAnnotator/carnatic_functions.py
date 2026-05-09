# ============================================================================

# IMPORTS

# ============================================================================

import os
import sys
import random
import numpy as np
import pandas as pd
import json
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed

# Scientific & Audio Processing Libraries
import librosa
import crepe
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import find_peaks
try:
    from melakarta_signatures import MELAKARTA_RAGAS, get_allowed_swaras
    COMMON_RAGAS = MELAKARTA_RAGAS
except ImportError:
    COMMON_RAGAS = {}
    MELAKARTA_RAGAS = {}
    def get_allowed_swaras(r): return None
from scipy.spatial.distance import pdist, squareform
from fastdtw import fastdtw

# Machine Learning Libraries
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.metrics import silhouette_score

# Plotting & Display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import Audio, display, HTML
import IPython.display as ipd
import ipywidgets as widgets
from IPython.display import display, clear_output

# ============================================================================

# CONFIGURATION

# ============================================================================

AUDIO_CONFIG = {
    "sample_rate": 44100,
    "confidence_threshold": 0.7
}

CREPE_CONFIG = {
    "viterbi": True,
    "step_size": 20,
    "model_capacity": "tiny",
    "verbose": False
}

CARNATIC_RATIOS = {
    # Lower Octave (Mandra Sthayi) - underscore suffix
    'Sa_': 0.5, 'Ri1_': 0.5 * 16/15, 'Ri2_': 0.5 * 9/8, 'Ga2_': 0.5 * 6/5,
    'Ga3_': 0.5 * 5/4, 'Ma1_': 0.5 * 4/3, 'Ma2_': 0.5 * 45/32, 'Pa_': 0.5 * 3/2,
    'Da1_': 0.5 * 8/5, 'Da2_': 0.5 * 5/3, 'Ni2_': 0.5 * 16/9, 'Ni3_': 0.5 * 15/8,
    
    # Middle Octave (Madhya Sthayi) - no suffix
    'Sa': 1.0, 'Ri1': 1.0 * 16/15, 'Ri2': 1.0 * 9/8, 'Ga2': 1.0 * 6/5,
    'Ga3': 1.0 * 5/4, 'Ma1': 1.0 * 4/3, 'Ma2': 1.0 * 45/32, 'Pa': 1.0 * 3/2,
    'Da1': 1.0 * 8/5, 'Da2': 1.0 * 5/3, 'Ni2': 1.0 * 16/9, 'Ni3': 1.0 * 15/8,
    
    # Upper Octave (Tara Sthayi) - apostrophe suffix
    "Sa'": 2.0, "Ri1'": 2.0 * 16/15, "Ri2'": 2.0 * 9/8, "Ga2'": 2.0 * 6/5,
    "Ga3'": 2.0 * 5/4, "Ma1'": 2.0 * 4/3, "Ma2'": 2.0 * 45/32, "Pa'": 2.0 * 3/2,
    "Da1'": 2.0 * 8/5, "Da2'": 2.0 * 5/3, "Ni2'": 2.0 * 16/9, "Ni3'": 2.0 * 15/8,
}



# ============================================================================

# UTILITY FUNCTIONS

# ============================================================================

def get_raaga_context(audio_dir: str) -> Dict[str, str]:
    """Derives raaga name and sets up all data file paths."""
    # Resolve absolute path to handle relative inputs correctly
    path_obj = Path(audio_dir).resolve()
    raaga_name = path_obj.name.replace('_Vocals', '')
    
    # User Explicit Request: "data/RaagaName" structure ONLY. No "_Data" suffix.
    # We check:
    # 1. Local 'data' (e.g. VocalAnnotator/data/Mayamalavagowlai)
    # 2. Root 'data' (e.g. CarnaticAnnotater/data/Mayamalavagowlai) - via Audio Parent
    
    candidates = [
        # 1. Standard: Raaga Root / Raaga_Data (e.g. Mayamalavagowlai/Mayamalavagowlai_Data)
        path_obj.parent / f"{raaga_name}_Data",

        # 2. Local 'data' relative to CWD
        Path.cwd() / "data" / raaga_name,
        
        # 3. 'data' at the same level as the Raga Root folder (if audio is in Raga/Vocals)
        path_obj.parent.parent / "data" / raaga_name,
        
        # 4. Simple relative fallback
        Path("data") / raaga_name
    ]
    
    data_dir = None
    for cand in candidates:
        if (cand / f"crepe_{raaga_name}.csv").exists():
            data_dir = cand
            break
            
    # Fallback: Default to standard "_Data" structure
    if not data_dir:
        data_dir = path_obj.parent / f"{raaga_name}_Data"
        # If even that doesn't exist, try just the passed dir
        if not data_dir.exists():
             data_dir = path_obj
            
    # Default: Create "data/RaagaName" in CWD if nothing found
    if not data_dir:
        data_dir = Path.cwd() / "data" / raaga_name
        
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "raaga_name": raaga_name,
        "crepe_csv": str(data_dir / f"crepe_{raaga_name}.csv"),
        "clean_crepe_csv": str(data_dir / f"crepe_{raaga_name}_clean.csv"),
        "carva_csv": str(data_dir / f"carva_{raaga_name}.csv"),
        "log_csv": str(data_dir / f"log_{raaga_name}.csv")
    }

def clean_audio(audio_dir: str, window_size: float = 0.5, hop_size: float = 0.25, 
                amp_thresh_ratio: float = 0.2, freq_thresh_ratio: float = 3.0, verbose: bool = True):
    """
    Preprocessing: Filters CREPE data based on Audio Amplitude and Frequency Outliers (Per Song).
    
    Args:
        window_size (float): Window size in seconds for amplitude check (default 0.5)
        hop_size (float): Hop size in seconds for amplitude check (default 0.25)
        amp_thresh_ratio (float): Reject windows where median amp < ratio * song median amp (default 0.2)
        freq_thresh_ratio (float): Reject frequencies > ratio * song median frequency (default 3.0)
    
    Creates a new 'clean' CREPE CSV file.
    """
    import librosa
    
    context = get_raaga_context(audio_dir)
    input_csv = context["crepe_csv"]
    output_csv = context["clean_crepe_csv"]
    
    if not os.path.exists(input_csv):
        print(f"❌ Input CSV not found: {input_csv}")
        return None

    if verbose: print(f"🧹 Starting Audio Cleaning for {context['raaga_name']} (Per Song Filter)...")
    
    df = pd.read_csv(input_csv)
    unique_songs = df['Index'].unique()
    
    for song_idx in tqdm(unique_songs, desc="Cleaning Songs"):
        # Select rows for this song
        # Using a boolean mask on the main DF to easily apply updates later
        song_mask = df['Index'] == song_idx
        song_df = df[song_mask]
        
        if song_df.empty: continue
        
        # --- 1. Frequency Filter (Per Song) ---
        valid_freqs = song_df['Frequency'][song_df['Frequency'] > 10]
        if not valid_freqs.empty:
            song_median_freq = valid_freqs.median()
            freq_cutoff = song_median_freq * freq_thresh_ratio
            
            # Identify outliers in this song
            # We use the index from song_df which matches df's index
            high_freq_indices = song_df[song_df['Frequency'] > freq_cutoff].index
            if not high_freq_indices.empty:
                df.loc[high_freq_indices, ['Frequency', 'Tonic_Normalized_Frequency']] = np.nan
        
        # --- 2. Amplitude Filter (Per Song) ---
        audio_path_raw = song_df.iloc[0]['AudioPath']
        audio_path = None
        
        candidates = [
            audio_path_raw,
            os.path.join(audio_dir, os.path.basename(audio_path_raw)),
        ]
        
        for c in candidates:
            if os.path.exists(c):
                audio_path = c
                break
        
        if not audio_path:
            if verbose: print(f"   ⚠️ Audio not found for Song {song_idx}: {os.path.basename(audio_path_raw)}")
            continue
            
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"   ❌ Error loading audio {audio_path}: {e}")
            continue
            
        rms_hop_ms = 10 
        rms_hop_len = int(sr * rms_hop_ms / 1000)
        rms_frame_len = int(sr * 0.02) 
        
        envelope = librosa.feature.rms(y=y, frame_length=rms_frame_len, hop_length=rms_hop_len)[0]
        times_env = librosa.frames_to_time(np.arange(len(envelope)), sr=sr, hop_length=rms_hop_len)
        
        # Song Median Amplitude
        song_median_amp = np.median(envelope)
        amp_cutoff = song_median_amp * amp_thresh_ratio
        
        if verbose:
            # Use tqdm.write if available to avoid breaking progress bar, else print
            msg = f"   [Song {song_idx}] Median Freq: {song_median_freq:.2f}, Median Amp: {song_median_amp:.4f}"
            try:
                tqdm.write(msg)
            except:
                print(msg)

        duration = len(y) / sr
        curr_time = 0
        quiet_ranges = []
        
        while curr_time + window_size <= duration:
            idx_start = np.searchsorted(times_env, curr_time)
            idx_end = np.searchsorted(times_env, curr_time + window_size)
            chunk = envelope[idx_start:idx_end]
            
            if len(chunk) > 0:
                win_median = np.median(chunk)
                if win_median < amp_cutoff:
                    quiet_ranges.append((curr_time, curr_time + window_size))
            curr_time += hop_size
            
        if quiet_ranges:
            s_indices = song_df.index.values # Explicit numpy array
            s_times = song_df['Time'].values
            silence_mask = np.zeros(len(s_times), dtype=bool)
            
            for q_start, q_end in quiet_ranges:
                i1 = np.searchsorted(s_times, q_start)
                i2 = np.searchsorted(s_times, q_end)
                silence_mask[i1:i2] = True
            
            # Update main dataframe using the masked indices
            df.loc[s_indices[silence_mask], ['Frequency', 'Tonic_Normalized_Frequency']] = np.nan

    df.to_csv(output_csv, index=False)
    if verbose: print(f"✅ Cleaned data saved to {output_csv}")
    return output_csv

def clean_np_float_list(seg_str: str) -> np.ndarray:
    """Convert a stringified list with np.float64 entries into a proper list of floats."""
    cleaned = re.sub(r'np\\.float64\\(([^)]+)\\)', r'\\1', seg_str)
    return np.array(ast.literal_eval(cleaned), dtype=float)

def interpolate_list(lst: Union[list, np.ndarray], target_len: int) -> np.ndarray:
    """Standardized interpolation of a pitch segment to a fixed length."""
    if lst is None or len(lst) == 0:
        return np.zeros(target_len)
    
    # Ensure numpy array
    arr = np.array(lst)
    original_len = len(arr)
    
    # Linear interpolation across normalized [0, original_len-1] grid
    return np.interp(np.linspace(0, original_len - 1, target_len),
                    np.arange(original_len), arr)


# ============================================================================
# UTILITY: FREQUENCY & NOTE CONVERSIONS
# ============================================================================

def hz_to_semitones(hz: float, ref_hz: float = 440.0) -> float:
    """Converts frequency to absolute semitones (MIDI-standard, A4=69)."""
    if hz <= 0: return -1.0
    return 12 * np.log2(hz / ref_hz) + 69.0

def semitones_to_hz(semitones: float, ref_hz: float = 440.0) -> float:
    """Converts absolute semitones back to frequency (Hz)."""
    return ref_hz * (2.0 ** ((semitones - 69.0) / 12.0))

def hz_to_ratio(hz: float, tonic: float) -> float:
    """Converts frequency to a ratio relative to the tonic frequency."""
    return hz / tonic if (tonic and tonic > 0) else 0.0

def ratio_to_semitones(ratio: float) -> float:
    """Converts a frequency ratio to relative semitones (e.g., 1.5 -> 7.02)."""
    if ratio <= 0: return 0.0
    return 12 * np.log2(ratio)

def hz_to_note_cents(hz: float) -> Tuple[int, int]:
    """Helper for UI: Returns (MIDI_Note_Number, Cents_Offset)."""
    st = hz_to_semitones(hz)
    if st < 0: return -1, 0
    note = int(round(st))
    cents = int((st - note) * 100)
    return note, cents


# ============================================================================

# PHASE 1: DATA PREPARATION & INITIAL CLUSTERING

# ============================================================================

def primary_clustering_pca(audio_dir: str, config: Dict, song_index: Optional[int] = None, write_to_file: bool = True, verbose: bool = True, **kwargs):
    """
    Optimized clustering function using a distance matrix modification approach.
    Supports dry-run (write_to_file=False) and safe single-song updates.
    """
    # Merge kwargs into config (priority to kwargs)
    config = {**config, **kwargs}
    
    context = get_raaga_context(audio_dir)
    
    # Preprocessing / Cleaning Step
    clean_crepe_path = context.get("clean_crepe_csv")
    preprocess_needed = config.get("preprocess_audio", True) # Default to True per user request logic
    
    # If explicit preprocess requested OR clean file missing, run cleaning
    if preprocess_needed:
        # Check if we already have it? 
        # User said "makes a new crepe file". To update parameters we might need to rerun.
        # For now, if clean file is missing OR config["force_clean"] is true
        if (clean_crepe_path and not os.path.exists(clean_crepe_path)) or config.get("force_clean", False):
            print(f"🧹 Running Audio Preprocessing (Clean)...")
            clean_audio(
                audio_dir, 
                window_size=config.get("clean_window", 0.5),
                hop_size=config.get("clean_hop", 0.25),
                amp_thresh_ratio=config.get("clean_amp_thresh", 0.2),
                freq_thresh_ratio=config.get("clean_freq_thresh", 3.0),
                verbose=verbose
            )
        elif clean_crepe_path and os.path.exists(clean_crepe_path):
             if verbose: print(f"🧹 Cleaned audio file exists. Skipping cleaning. (Set 'force_clean': True to override)")
            
    # Select Input CSV
    if clean_crepe_path and os.path.exists(clean_crepe_path):
        crepe_csv_path = clean_crepe_path
        if verbose: print(f"📂 Using Cleaned CREPE Data: {os.path.basename(crepe_csv_path)}")
    else:
        crepe_csv_path = context["crepe_csv"]
        if verbose: print(f"⚠️ Cleaned data not found. Using Original CREPE Data: {os.path.basename(crepe_csv_path)}")

    carva_file_path = context["carva_csv"]
    raaga_name = context["raaga_name"]

    if not Path(crepe_csv_path).exists():

        print(f"❌ CREPE CSV file not found: {crepe_csv_path}.")

        return pd.DataFrame() # Return empty DF on failure

    df_master = pd.read_csv(crepe_csv_path)

    def extract_overlapping_segments(arr, window_size, hop_factor):

        # Optimized overlapping window extraction

        segments, starts = [], []

        hop_size = max(1, int(window_size / hop_factor))

        i = 0

        while i <= len(arr) - window_size:

            segment = arr[i:i + window_size]

            if not np.any(np.isnan(segment)):

                segments.append(segment)

                starts.append(i)

            i += hop_size

        return segments, starts

    def process_one_song_pca(song_idx):

        song_df = df_master[df_master["Index"] == song_idx].reset_index(drop=True)

        if song_df.empty: return []

        audio_path = song_df.loc[0, "AudioPath"]

        original_song_data = song_df["Tonic_Normalized_Frequency"].values

        remaining_data = original_song_data.copy()

        all_found_segments = []

        global_label_offset = 0

        window_size = config["initial_window_size"]

        while window_size >= config["min_window_size"]:

            segments, segment_starts = extract_overlapping_segments(remaining_data, window_size, config["hop_factor"])

            if len(segments) < config.get("outlier_threshold", 2):

                window_size -= config["decay_size"]

                continue

            X_abs = np.stack(segments)

            X_shape = X_abs - np.mean(X_abs, axis=1, keepdims=True)

            X_combined = np.concatenate([X_abs, X_shape], axis=1)

            n_samples = X_combined.shape[0]

            n_features = X_combined.shape[1]

            n_components = min(config["pca_components"], n_samples, n_features)

            pca = PCA(n_components=n_components)

            X_pca = pca.fit_transform(X_combined)

            dist_matrix = squareform(pdist(X_pca, metric='euclidean'))

            if dist_matrix.size > 0:

                large_distance = np.max(dist_matrix) * 10 + 1 

                for i in range(len(segments)):

                    for j in range(i + 1, len(segments)):

                        start_i, start_j = segment_starts[i], segment_starts[j]

                        if start_j < (start_i + window_size) and start_i < (start_j + window_size):

                            dist_matrix[i, j] = large_distance

                            dist_matrix[j, i] = large_distance

            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=config["similarity_threshold"], metric='precomputed', linkage='average')

            labels = clustering.fit_predict(dist_matrix)

            cluster_origins = defaultdict(list)

            for seg_idx, start_pos in enumerate(segment_starts):

                cluster_origins[labels[seg_idx]].append(start_pos)

            # Sort clusters by size (descending) to prioritize dominant motifs
            # This ensures that if Cluster A (big) overlaps Cluster B (small), we keep A.
            sorted_clusters = sorted(cluster_origins.items(), key=lambda x: len(x[1]), reverse=True)

            for label, starts in sorted_clusters:
                if len(starts) < config["outlier_threshold"]:
                    continue

                global_label = global_label_offset + label

                for start_frame in starts:
                    end_frame = start_frame + window_size
                    
                    # Check for overlap: If any part of this range is already claimed (NaN), skip it.
                    # This enforces the "Jump to end" behavior the user requested.
                    if np.isnan(remaining_data[start_frame : end_frame]).any():
                        continue 

                    segment_data = original_song_data[start_frame : end_frame]
                    all_found_segments.append([start_frame, end_frame, global_label, segment_data])
                    
                    # Mark this region as occupied
                    remaining_data[start_frame : end_frame] = np.nan

            global_label_offset += (len(set(labels)) + 1)

            window_size -= config["decay_size"]

        # Renumber labels

        all_labels = [lbl for _, _, lbl, _ in all_found_segments]

        if not all_labels: return []

        from collections import Counter

        counts = Counter(all_labels)

        sorted_labels = [lbl for lbl, _ in counts.most_common()]

        label_map = {old: new for new, old in enumerate(sorted_labels)}

        rows = []

        for start, end, lbl, seg_data in all_found_segments:

            rows.append({

                "Index": int(song_idx), "AudioPath": audio_path,

                "SegmentList": json.dumps(list(seg_data)),

                "StartFrame": int(start), "EndFrame": int(end),

                "Primary_Label": label_map.get(lbl, -1)

            })

        return rows

    # --- Execution Logic Refined ---

    if song_index is None:
        if verbose: print(f"🔍 Starting full clustering for {raaga_name}...")

        # If processing ALL, we wipe the file

        if write_to_file and Path(carva_file_path).exists():
            if verbose: print(f"   (Wiping old data for full run)")

            os.remove(carva_file_path)

        song_indices = sorted(df_master["Index"].unique())

        # Processing sequentially to avoid MemoryError with large datasets/pickling
        # Parallel(n_jobs=os.cpu_count())... caused BrokenProcessPool
        results = []
        for idx in tqdm(song_indices, desc="Processing all songs (Sequential)"):
             results.append(process_one_song_pca(idx))

        flat_rows = [row for song_rows in results for row in song_rows]

    else:

        # Single Song Mode
        if verbose: print(f"🎵 Single-Song Clustering: '{raaga_name}' (Index: {song_index})")

        flat_rows = process_one_song_pca(song_index)

    if not flat_rows:
        if verbose: print("⚠️ No motifs found with the current parameters.")

        return pd.DataFrame()

    carva_df = pd.DataFrame(flat_rows)

    if write_to_file:

        if song_index is None:

            # Full write



            carva_df.to_csv(carva_file_path, index=False)
            if verbose: print(f"✅ Clustering finished. Saved {len(carva_df)} segments to {carva_file_path} (Column: Primary_Label)")

        else:

            # Smart Update

            if Path(carva_file_path).exists():

                existing_df = pd.read_csv(carva_file_path)

                # Remove old entries for this song

                existing_df = existing_df[existing_df["Index"] != song_index]

                # Append new entries

                updated_df = pd.concat([existing_df, carva_df], ignore_index=True)

                updated_df.to_csv(carva_file_path, index=False)
                if verbose: print(f"✅ Updated Song {song_index} in {carva_file_path} (Total segments: {len(updated_df)})")

            else:

                carva_df.to_csv(carva_file_path, index=False)
                if verbose: print(f"✅ Created {carva_file_path} with Song {song_index}")

    else:
        if verbose: print("✅ (Dry Run) Clustering complete in memory.")

    return carva_df if verbose else None


# ============================================================================


def secondary_clustering_dtw(audio_dir: str, config: Dict, song_index: Optional[int] = None, write_to_file: bool = True):
    """
    Performs secondary clustering using Dynamic Time Warping (DTW) on interpolated segments.
    Supports single-song updates and dry runs.
    """
    context = get_raaga_context(audio_dir)
    raaga_name = context["raaga_name"]
    carva_path = context["carva_csv"]
    print(f"🔬 Starting DTW re-clustering for '{raaga_name}'...")

    if not Path(carva_path).exists():
        print(f"❌ ERROR: File not found at {carva_path}"); return pd.DataFrame()
    full_df = pd.read_csv(carva_path)

    # Target Selection

    if song_index is not None:

        target_df = full_df[full_df['Index'] == song_index].copy()

        print(f"   -> Targeting ONLY Song Index: {song_index}")

    else:

        target_df = full_df.copy()

        print("   -> Targeting ALL songs.")

    if target_df.empty:

        print("⚠️ No segments found for the specified criteria."); return pd.DataFrame()

    interpol_size = config.get("interpolation_size", config.get("second_phase_window_size", 50))

    similarity_threshold = config.get("dtw_similarity_threshold", 1.5)

    # Interpolation

    interpolated_segments, valid_original_indices = [], target_df.index.tolist()

    for seg_str in target_df['SegmentList']:

        try:

            segment = clean_np_float_list(seg_str)

            if len(segment) > 0:

                interpolated_segments.append(np.array(interpolate_list(segment, interpol_size)))

            else:

                interpolated_segments.append(None)

        except:

            interpolated_segments.append(None)

    final_segments = [seg for seg in interpolated_segments if seg is not None]

    final_indices = [idx for i, idx in enumerate(valid_original_indices) if interpolated_segments[i] is not None]

    print(f"   -> Found {len(final_segments)} valid segments.")

    if len(final_segments) < 2:

        print("⚠️ Not enough segments (< 2) for clustering."); return pd.DataFrame()

    # Distance Matrix & Clustering

    num_segments = len(final_segments)

    dist_matrix = np.zeros((num_segments, num_segments))

    for i in tqdm(range(num_segments), desc="Calculating DTW Matrix"):

        for j in range(i + 1, num_segments):

            d, _ = fastdtw(final_segments[i], final_segments[j])

            dist_matrix[i, j] = d

            dist_matrix[j, i] = d

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=similarity_threshold, metric='precomputed', linkage='average')

    labels = clustering.fit_predict(dist_matrix)

    # Update DataFrame

    target_df.loc[final_indices, 'DTW_Label'] = labels

    if not write_to_file:

        print("✅ (Dry Run) DTW Coloring complete in memory.")

        return target_df

    # Smart File Update

    if 'DTW_Label' not in full_df.columns: full_df['DTW_Label'] = -1

    # Update the rows in the main DF with the new labels

    full_df.loc[target_df.index, 'DTW_Label'] = target_df['DTW_Label']

    full_df.to_csv(carva_path, index=False)

    print(f"✅ Updated {carva_path} with {len(set(labels))} new DTW clusters.")

    return target_df

def secondary_clustering_pca(audio_dir: str, config: Dict, song_index: Optional[int] = None, write_to_file: bool = True, verbose: bool = True):
    """
    Performs secondary clustering using PCA on interpolated segments.
    Supports single-song updates and dry runs.
    """
    context = get_raaga_context(audio_dir)
    raaga_name = context["raaga_name"]
    carva_path = context["carva_csv"]
    if verbose: print(f"🔬 Starting PCA re-clustering for '{raaga_name}'...")

    if not Path(carva_path).exists():
        print(f"❌ ERROR: File not found at {carva_path}"); return pd.DataFrame()

    full_df = pd.read_csv(carva_path)
    full_df = normalize_carva_df(full_df)

    if song_index is not None:
        target_df = full_df[full_df['Index'] == song_index].copy()
        if verbose: print(f"   -> Targeting ONLY Song Index: {song_index}")
    else:
        target_df = full_df.copy()
        if verbose: print("   -> Targeting ALL songs.")

    if target_df.empty:
        print("⚠️ No segments found."); return pd.DataFrame()

    interpolation_size = config.get("interpolation_size", config.get("second_phase_window_size", 50))
    similarity_threshold = config.get("pca_similarity_threshold", config.get("similarity_threshold_secondary", 0.7))
    pca_components = config.get("pca_components", config.get("pca_components_secondary", 10))
    interpolated_segments, valid_original_indices = [], target_df.index.tolist()

    for seg_str in target_df['SegmentList']:
        try:
            segment = clean_np_float_list(seg_str)
            if len(segment) > 0:
                interpolated_segments.append(np.array(interpolate_list(segment, interpolation_size)))
            else:
                interpolated_segments.append(None)
        except:
            interpolated_segments.append(None)

    final_segments = [seg for seg in interpolated_segments if seg is not None]
    final_indices = [idx for i, idx in enumerate(valid_original_indices) if interpolated_segments[i] is not None]
    if verbose: print(f"   -> Found {len(final_segments)} valid segments.")

    if len(final_segments) < 2:
        if verbose: print("⚠️ Not enough segments (< 2).")
        return pd.DataFrame()

    # PCA & Clustering
    X_abs = np.stack(final_segments)
    X_shape = X_abs - np.mean(X_abs, axis=1, keepdims=True)
    X_combined = np.concatenate([X_abs, X_shape], axis=1)
    
    n_components = min(pca_components, X_combined.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_combined)
    
    # HYBRID CLUSTERING STRATEGY
    # For large N, calculating a full Distance Matrix is O(N^2) memory and time.
    # We switch to BIRCH for N > 3000 to keep it interactive.
    
    if len(X_pca) > 3000:
        if verbose: print(f"   -> Large dataset ({len(X_pca)} segments). Switching to BIRCH clustering for speed.")
        # Birch threshold is roughly equivalent to radius. 
        # We assume similarity_threshold matches somewhat, but might need tuning.
        # Adjusted heuristic: Birch threshold is usually smaller than Agglomerative distance_threshold.
        birch_thresh = similarity_threshold * 0.5 
        
        brc = Birch(n_clusters=None, threshold=birch_thresh, branching_factor=50)
        brc.fit(X_pca)
        labels = brc.predict(X_pca)
    else:
        # Standard Hierarchical for small datasets (high precision)
        dist_matrix = squareform(pdist(X_pca, metric='euclidean'))      
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=similarity_threshold, metric='precomputed', linkage='average')
        labels = clustering.fit_predict(dist_matrix)

    # Update DataFrame
    # Reorder labels by cluster size (cluster 1 = largest, 2 = second largest, etc.)
    from collections import Counter
    label_counts = Counter(labels)
    sorted_labels = [lbl for lbl, _ in label_counts.most_common()]
    # Create mapping: old label -> new label (1-indexed)
    label_map = {old: (new + 1) for new, old in enumerate(sorted_labels)}
    # Apply remapping
    remapped_labels = [label_map[lbl] for lbl in labels]
    
    target_df.loc[final_indices, 'Secondary_Label'] = remapped_labels   
    if not write_to_file:
        if verbose: print("✅ (Dry Run) PCA Coloring complete in memory.")
        return target_df

    # Smart File Update

    if 'Secondary_Label' not in full_df.columns: full_df['Secondary_Label'] = -1

    full_df.loc[target_df.index, 'Secondary_Label'] = target_df['Secondary_Label']

    full_df.to_csv(carva_path, index=False)

    
    if verbose: print(f"✅ Updated {carva_path} with {len(set(remapped_labels))} new PCA clusters (ordered by size).")

    return target_df if verbose else None

def tertiary_clustering_pca(audio_dir: str, config: Dict, verbose: bool = True):
    """
    Groups Secondary Clusters based on their SHAPE only (Meta-Clustering).
    Generates meaningful labels like 'Pa_Constant' or 'Ri_Slide'.
    
    Logic:
    1. Compute Average Shape for each Secondary Cluster.
    2. Cluster these Averages (Shape Only).
    3. Assign 'ShapeType' (0, 1, 2...)
    4. For each Secondary Cluster, find its 'Anchor Note' (Mean Height).
    5. Final Label = {Anchor}_{ShapeType}
    """
    context = get_raaga_context(audio_dir)
    carva_path = context["carva_csv"]
    
    if verbose: print(f"🔬 Starting Tertiary (Meta) Clustering...")
    full_df = pd.read_csv(carva_path)
    full_df = normalize_carva_df(full_df)
    
    if 'Secondary_Label' not in full_df.columns:
        if verbose: print("❌ Run Secondary Clustering first.")
        return

    # Filter Valid
    valid_df = full_df[full_df['Secondary_Label'] != -1]
    unique_clusters = sorted(valid_df['Secondary_Label'].unique())
    
    if verbose: print(f"   -> Analyzing {len(unique_clusters)} Secondary Clusters...")
    
    # 1. Compute Centroids (Average Shape & Average Height)
    cluster_centroids = []
    cluster_heights = []
    valid_cluster_ids = []
    
    interpolation_size = config.get("interpolation_size", 64)
    
    for cid in unique_clusters:
        c_rows = valid_df[valid_df['Secondary_Label'] == cid]
        
        # Collect interpolated segments
        segments = []
        raw_values = [] # For height
        
        for _, row in c_rows.iterrows():
            try:
                seg = np.array(json.loads(row["SegmentList"]))
                seg_interp = interpolate_list(seg, interpolation_size)
                segments.append(seg_interp)
                raw_values.extend(seg)
            except: pass
            
        if not segments: continue
        
        # Mean Shape
        avg_shape = np.mean(segments, axis=0) # shape (64,)
        # Mean Height (scalar)
        avg_height = np.nanmean(raw_values)
        
        cluster_centroids.append(avg_shape)
        cluster_heights.append(avg_height)
        valid_cluster_ids.append(cid)

    # 2. Cluster the SHAPES (Shape Only)
    if not valid_cluster_ids: return

    X_abs = np.stack(cluster_centroids)
    X_shape = X_abs - np.mean(X_abs, axis=1, keepdims=True) # Normalize Height to 0
    
    similarity_thresh = config.get("similarity_threshold_tertiary", 0.5)
    
    if verbose: print(f"   -> Grouping {len(valid_cluster_ids)} clusters by Shape Similarity (Threshold={similarity_thresh})...")
    
    # PCA on Shapes
    n_components = min(5, len(valid_cluster_ids))
    pca = PCA(n_components=n_components) 
    X_pca = pca.fit_transform(X_shape)
    
    # Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=similarity_thresh, 
        linkage='average',
        metric='euclidean' # explicit metric often safer
    )
    shape_labels = clustering.fit_predict(X_pca)
    
    n_meta_clusters = len(set(shape_labels))
    
    # 3. Map to Swaras
    # Need CARNATIC_RATIOS
    # Assuming standard map: S=1, R1=1.059 etc.
    # Need to verify normalization. 'SegmentList' is usually ratios?
    
    def get_anchor_name(ratio):
        best_n, min_d = "?", 99
        for name, val in CARNATIC_RATIOS.items():
            if abs(val - ratio) < min_d:
                min_d = abs(val - ratio)
                best_n = name
        return best_n
        
    # 4. Generate Dictionary {SecondaryID: TertiaryLabel}
    # Reorder shape labels by cluster size and assign simple numeric labels
    from collections import Counter
    shape_counts = Counter(shape_labels)
    sorted_shapes = [shp for shp, _ in shape_counts.most_common()]
    # Create mapping: old shape -> new shape (1-indexed numeric)
    shape_remap = {old: (new + 1) for new, old in enumerate(sorted_shapes)}
    
    mapping = {}
    
    for idx, cid in enumerate(valid_cluster_ids):
        old_shp_id = shape_labels[idx]
        new_shp_id = shape_remap[old_shp_id]  # Remapped shape ID (just a number)
        
        # Simple numeric label (1, 2, 3, ...)
        mapping[cid] = new_shp_id
        
    # 5. Apply to DataFrame
    # Map from Secondary_Label -> Tertiary_Label
    full_df['Tertiary_Label'] = full_df['Secondary_Label'].map(mapping).fillna(-1)
    
    full_df.to_csv(carva_path, index=False)
    if verbose: print(f"✅ Tertiary Clustering Complete. Mapped {len(unique_clusters)} clusters -> {n_meta_clusters} shapes (ordered by size).")
    
    return full_df if verbose else None

def plot_tertiary_summary(audio_dir: str, degree: int = 2, extrema: bool = False):
    """
    Visualizes the 'Meta-Shapes' found by Tertiary Clustering.
    degree=1: Plots RAW SEGMENTS (original lengths) for each shape.
    degree=2: Plots INTERPOLATED Secondary Cluster Averages (normalized).
    extrema=True: Plots local maxima (Red) and minima (Green) on the Grand Curve.
    """
    context = get_raaga_context(audio_dir)
    try:
        df = pd.read_csv(context["carva_csv"])
        # df = normalize_carva_df(df) # Don't normalize here; we need the full 'Pa_Shape60' string to extract Anchor info
    except:
        print("Error reading data."); return

    if 'Tertiary_Label' not in df.columns:
        print("❌ Tertiary_Label not found. Run tertiary_clustering_pca first.")
        return

    # Parse Labels
    df['Meta_ShapeID'] = df['Tertiary_Label'].apply(lambda x: x.split('_')[-1].replace('Shape','') if '_' in str(x) else -1)
    df['Meta_Anchor'] = df['Tertiary_Label'].apply(lambda x: x.split('_')[0] if '_' in str(x) else "?")
    
    valid = df[df['Meta_ShapeID'] != -1]
    unique_shapes = sorted(valid['Meta_ShapeID'].unique())
    
    if not unique_shapes: print("No shapes found."); return
    
    # Plot Config
    cols = 5
    rows = (len(unique_shapes) // cols) + 1
    plt.style.use('dark_background')
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
    axes = axes.flatten()
    
    mode_str = f"Degree {degree}"
    print(f"📊 Visualizing {len(unique_shapes)} Meta-Shapes ({mode_str})...")
    
    for i, shp_id in enumerate(unique_shapes):
        ax = axes[i]
        shape_rows = valid[valid['Meta_ShapeID'] == shp_id]
        grand_avg_list = []
        
        # DEGREE 1: Raw Original Segments (Original Lengths)
        if degree == 1:
            count = 0
            for _, row in shape_rows.head(200).iterrows():
                try:
                    seg = np.array(json.loads(row['SegmentList']))
                    seg_norm = seg - np.mean(seg) # Center for shape
                    ax.plot(np.arange(len(seg)), seg_norm, color='cyan', alpha=0.1, linewidth=1)
                    count += 1
                except: pass
            ax.set_title(f"Shape {shp_id} (n={count})\nRaw Lengths", fontsize=8)

        # DEGREE 2: ALL Raw Segments (Interpolated to 64)
        elif degree == 2:
            count = 0
            for _, row in shape_rows.head(200).iterrows():
                try:
                    seg = np.array(json.loads(row['SegmentList']))
                    # Interpolate
                    seg_interp = interpolate_list(seg, 64)
                    
                    seg_norm = seg_interp - np.mean(seg_interp)
                    ax.plot(seg_norm, color='gray', alpha=0.1, linewidth=1)
                    grand_avg_list.append(seg_norm)
                    count += 1
                except: pass
            
            # Plot Grand Average of ALL segments
            if grand_avg_list:
                grand_avg = np.mean(grand_avg_list, axis=0)
                ax.plot(grand_avg, color='cyan', linewidth=3, label='Avg')
                # Extrema logic works on this grand average
                if extrema:
                    diff = np.diff(grand_avg)
                    max_indices = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
                    min_indices = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0] + 1
                    ax.scatter(max_indices, grand_avg[max_indices], color='red', s=30, zorder=5)
                    ax.scatter(min_indices, grand_avg[min_indices], color='lime', s=30, zorder=5)

            ax.set_title(f"Shape {shp_id} (n={count})\nRaw Interpolated", fontsize=8)

        # DEGREE 3: Secondary Cluster Averages (Interpolated)
        else:
            sec_clusters = shape_rows['Secondary_Label'].unique()
            
            for sec_id in sec_clusters:
                sec_rows = shape_rows[shape_rows['Secondary_Label'] == sec_id]
                segs = []
                for _, row in sec_rows.iterrows():
                    try: segs.append(clean_np_float_list(row['SegmentList']))
                    except: pass
                if not segs: continue
                
                # Interpolate & Average Secondary Cluster
                interp_segs = [np.interp(np.linspace(0,1,64), np.linspace(0,1,len(s)), s) for s in segs]
                sec_avg = np.mean(interp_segs, axis=0)
                sec_avg_norm = sec_avg - np.mean(sec_avg) # Shape only
                
                ax.plot(sec_avg_norm, color='gray', alpha=0.6, linewidth=1.5)
                grand_avg_list.append(sec_avg_norm) # Grand avg of averages

            # Plot Grand Average of Secondary Averages
            if grand_avg_list:
                grand_avg = np.mean(grand_avg_list, axis=0)
                ax.plot(grand_avg, color='cyan', linewidth=3, label='Grand Avg')
                if extrema:
                    diff = np.diff(grand_avg)
                    max_indices = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
                    min_indices = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0] + 1
                    ax.scatter(max_indices, grand_avg[max_indices], color='red', s=30, zorder=5)
                    ax.scatter(min_indices, grand_avg[min_indices], color='lime', s=30, zorder=5)

            anchors = shape_rows['Meta_Anchor'].unique()
            ax.set_title(f"Shape {shp_id}\nAnchors: {', '.join(anchors)}", fontsize=9)
            
        ax.axis('off')

        ax.axis('off')
        
    for j in range(i+1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.show()

def get_note_sequence(seg_interp, allowed_ratios):
    """
    Extracts a sequence of notes (one per 8-frame bin) from a 64-point segment.
    Uses Global Extrema Detection and maps to 'allowed_ratios'.
    """
    note_sequence = []
    
    # 1. Global Extrema Detection (Full Array)
    seg_grad = np.diff(seg_interp)
    g_peaks = np.where((seg_grad[:-1] > 0) & (seg_grad[1:] < 0))[0] + 1
    g_valleys = np.where((seg_grad[:-1] < 0) & (seg_grad[1:] > 0))[0] + 1
    
    # Combine Peaks, Valleys AND Endpoints (0, 63)
    # Ensure they are within bounds 0-63
    all_extrema = np.unique(np.concatenate([g_peaks, g_valleys, [0, 63]]))
    
    for bin_idx in range(8):
        b_start = bin_idx * 8
        b_end = (bin_idx + 1) * 8
        
        # Filter extrema relevant to this bin
        bin_extrema = all_extrema[(all_extrema >= b_start) & (all_extrema < b_end)]
        
        found_note = "_"
        if len(bin_extrema) > 0:
            # Find the extrema CLOSEST to an allowed bar
            best_match_note = "_"
            min_global_dist = 0.1 # Threshold (strict)
            
            for idx_global in bin_extrema:
                 # Safety check
                 if idx_global >= len(seg_interp): continue
                 
                 val = seg_interp[idx_global]
                 
                 # Find distance to nearest allowed note
                 current_best_note = "_"
                 current_min_dist = 999
                 
                 for r, n in allowed_ratios.items(): # Search Allowed Ratios only
                     dist = abs(r - val)
                     if dist < current_min_dist:
                         current_min_dist = dist
                         current_best_note = n
                 
                 # Is this extrema better than previous ones in this bin?
                 if current_min_dist < min_global_dist:
                     min_global_dist = current_min_dist
                     best_match_note = current_best_note
                     
            note_sequence.append(best_match_note)
        else:
            note_sequence.append("_")
            
    return note_sequence, all_extrema

def normalize_carva_df(df):
    """
    Standardizes CSV columns based on user request:
    1. Label -> Primary_Label
    2. PCA_Label -> Secondary_Label
    3. Tertiary_Label is already numeric (1, 2, 3, ...)
    """
    # 1. Rename
    rename_map = {}
    if 'Label' in df.columns and 'Primary_Label' not in df.columns:
        rename_map['Label'] = 'Primary_Label'
    if 'PCA_Label' in df.columns and 'Secondary_Label' not in df.columns:
        rename_map['PCA_Label'] = 'Secondary_Label'
        
    if rename_map:
        df = df.rename(columns=rename_map)
        
    # 2. Ensure Tertiary_Label is numeric (already is from clustering)
    if 'Tertiary_Label' in df.columns:
        # Convert to int, handling any legacy string formats
        df['Tertiary_Label'] = pd.to_numeric(df['Tertiary_Label'], errors='coerce').fillna(-1).astype(int)
              
    return df

def plot_cluster(cluster_number: int, order: str = 'secondary', song_index: Optional[int] = None, 
                 raaga_dir: Optional[str] = None, density: float = 1.0, normalize: bool = False):
    """
    Plot all segments in a cluster on one graph.
    
    Args:
        cluster_number (int): Cluster ID to plot
        order (str): Clustering level - 'primary', 'secondary', or 'tertiary'
        song_index (int, optional): Required for primary clustering (song-specific)
        raaga_dir (str, optional): Path to raaga directory (e.g., 'C:/path/to/Mayamalavagowlai')
                                   If None, tries to infer from current directory.
        density (float): Fraction of segments to plot (0.0 to 1.0). Use < 1.0 for large clusters.
                        E.g., 0.1 will randomly sample 10% of segments.
        normalize (bool): For tertiary clustering only - if True, plots normalized segments
                         (mean-centered). Ignored for primary/secondary.
    
    Behavior:
        - Primary: Plots raw segments (original lengths) from specified song
        - Secondary: Plots interpolated segments (all songs)
        - Tertiary: Plots interpolated segments (all songs), optionally normalized
    
    Example:
        plot_cluster(5, order='secondary', raaga_dir=r'C:\\Desktop\\Python\\CarnaticAnnotater\\Mayamalavagowlai')
        plot_cluster(3, order='primary', song_index=0, raaga_dir=r'C:\\Desktop\\Python\\CarnaticAnnotater\\Mayamalavagowlai')
        plot_cluster(1, order='tertiary', raaga_dir=raaga_dir, density=0.1, normalize=True)  # 10% sample, normalized
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Validate order
    order = order.lower()
    if order not in ['primary', 'secondary', 'tertiary']:
        raise ValueError("order must be 'primary', 'secondary', or 'tertiary'")
    
    # Validate density
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be between 0.0 and 1.0")
    
    # Get audio_dir (vocal directory)
    if raaga_dir is None:
        # Try to infer from current working directory
        raaga_dir = os.getcwd()
    
    # Construct the vocal directory path
    raaga_name = os.path.basename(raaga_dir)
    audio_dir = os.path.join(raaga_dir, f"{raaga_name}_Vocals")
    
    # Get context using the vocal directory
    context = get_raaga_context(audio_dir)
    carva_path = context["carva_csv"]
    
    if not Path(carva_path).exists():
        print(f"❌ ERROR: File not found at {carva_path}")
        print(f"   Raaga dir: {raaga_dir}")
        print(f"   Audio dir: {audio_dir}")
        return
    
    # Load data
    df = pd.read_csv(carva_path)
    df = normalize_carva_df(df)
    
    # Filter by clustering level
    if order == 'primary':
        if song_index is None:
            raise ValueError("song_index is required for primary clustering")
        
        # Primary clustering is per-song
        cluster_df = df[(df['Index'] == song_index) & (df['Primary_Label'] == cluster_number)]
        label_col = 'Primary_Label'
        title = f"Primary Cluster {cluster_number} - Song {song_index}"
        
    elif order == 'secondary':
        # Secondary clustering is across all songs
        cluster_df = df[df['Secondary_Label'] == cluster_number]
        label_col = 'Secondary_Label'
        title = f"Secondary Cluster {cluster_number}"
        
    else:  # tertiary
        # Tertiary clustering is across all songs
        cluster_df = df[df['Tertiary_Label'] == cluster_number]
        label_col = 'Tertiary_Label'
        title = f"Tertiary Cluster {cluster_number}"
    
    if cluster_df.empty:
        print(f"⚠️ No segments found for {order} cluster {cluster_number}")
        if order == 'primary':
            print(f"   (Song index: {song_index})")
        return
    
    # Apply density sampling if needed
    total_segments = len(cluster_df)
    if density < 1.0:
        sample_size = max(1, int(total_segments * density))
        cluster_df = cluster_df.sample(n=sample_size, random_state=42)
        print(f"📊 Plotting {len(cluster_df)} segments (sampled {density*100:.1f}% from {total_segments} total) from {order} cluster {cluster_number}")
    else:
        print(f"📊 Plotting {len(cluster_df)} segments from {order} cluster {cluster_number}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Audio collection for preview (first 8 segments)
    audio_items = [] 
    preview_count = 0
    max_previews = 8
    
    # Color map for distinct segments
    import matplotlib.colors as mcolors
    params = {'linewidth': 1.5, 'alpha': 0.6}
    colors = plt.cm.viridis(np.linspace(0, 1, min(len(cluster_df), 50))) # Cycle first 50 colors
    
    # Collect all segments for plotting
    all_segments = []
    
    if order == 'primary':
        # Primary: Raw segments
        for i, (idx, row) in enumerate(cluster_df.iterrows()):
            try:
                segment = clean_np_float_list(row['SegmentList'])
                if len(segment) > 0:
                    all_segments.append(segment)
                    
                    # Plot with variation
                    color_rgba = colors[i % len(colors)]
                    color_hex = mcolors.to_hex(color_rgba)
                    
                    ax.plot(segment, color=color_hex, **params)
                    
                    # Audio Preview Collection
                    if preview_count < max_previews:
                        try:
                            a_path = row.get('AudioPath')
                            s_frame = row.get('StartFrame')
                            e_frame = row.get('EndFrame')
                            
                            if a_path and os.path.exists(a_path) and s_frame is not None:
                                # Frame to Time (assuming 20ms step as per config)
                                start_s = s_frame * 0.02
                                end_s = e_frame * 0.02
                                duration = end_s - start_s
                                
                                y, sr = librosa.load(a_path, sr=None, offset=start_s, duration=duration)
                                audio_items.append({
                                    'y': y, 'sr': sr, 'color': color_hex, 'id': i+1,
                                    'info': f"Seg {idx} ({start_s:.1f}s - {end_s:.1f}s)"
                                })
                                preview_count += 1
                        except Exception as e:
                            print(f"Error loading audio for segment {idx}: {e}")
            except:
                continue
    
    else:  # secondary or tertiary
        # Interpolated segments
        interpolation_size = 64
        
        for i, (idx, row) in enumerate(cluster_df.iterrows()):
            try:
                segment = clean_np_float_list(row['SegmentList'])
                if len(segment) > 0:
                    interpolated = np.array(interpolate_list(segment, interpolation_size))
                    
                    if order == 'tertiary' and normalize:
                        interpolated = interpolated - np.mean(interpolated)
                    
                    all_segments.append(interpolated)
                    
                    # Plot
                    color_rgba = colors[i % len(colors)]
                    color_hex = mcolors.to_hex(color_rgba)
                    
                    ax.plot(interpolated, color=color_hex, **params)
                    
                    # Audio Preview
                    if preview_count < max_previews:
                        try:
                            a_path_raw = row.get('AudioPath')
                            if not os.path.exists(a_path_raw):
                                a_path = os.path.join(audio_dir, os.path.basename(a_path_raw))
                            else:
                                a_path = a_path_raw
                                
                            s_frame = row.get('StartFrame')
                            e_frame = row.get('EndFrame')
                            
                            if os.path.exists(a_path) and s_frame is not None:
                                start_s = s_frame * 0.02
                                end_s = e_frame * 0.02
                                duration = end_s - start_s
                                
                                y, sr = librosa.load(a_path, sr=None, offset=start_s, duration=duration)
                                audio_items.append({
                                    'y': y, 'sr': sr, 'color': color_hex, 'id': i+1,
                                    'info': f"Seg {idx} ({start_s:.1f}s - {end_s:.1f}s)"
                                })
                                preview_count += 1
                        except: pass
            except:
                continue
    
    # Add Carnatic swara grid overlay
    if all_segments:
        all_values = np.concatenate(all_segments)
        y_min, y_max = np.nanmin(all_values), np.nanmax(all_values)
        for swara_name, ratio in CARNATIC_RATIOS.items():
            if y_min <= ratio <= y_max:
                ax.axhline(y=ratio, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
                ax.text(ax.get_xlim()[1] * 1.01, ratio, swara_name, 
                       fontsize=9, va='center', color='darkred', fontweight='bold')
    
    # Update title
    if order == 'tertiary' and normalize:
        title += " (Normalized)"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Frame' if order == 'primary' else 'Interpolated Frame', fontsize=12)
    ax.set_ylabel('Frequency Ratio (Tonic-Normalized)' if not normalize else 'Normalized Frequency Ratio (Mean-Centered)', fontsize=12)
    ax.grid(True, alpha=0.2)
    
    # Add cluster info
    info_text = f"Total segments: {len(cluster_df)}"
    if order != 'primary':
        songs = cluster_df['Index'].nunique()
        info_text += f" | Songs: {songs}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # Display Individual Audio Widgets with Color Linking
    if audio_items:
        print(f"🔊 Audio Previews (Top {len(audio_items)}):")
        
        # Create a VBox of HBoxes
        rows = []
        for item in audio_items:
            # Color Badge + Info
            badge_html = widgets.HTML(
                value=f"<div style='width:20px; height:20px; background-color:{item['color']}; "
                      f"border: 1px solid #777; border-radius: 50%; display:inline-block; vertical-align:middle; margin-right:10px;'></div>"
                      f"<span style='font-family:monospace; font-size:12px;'>{item['info']}</span>"
            )
            
            # Use Output widget to capture the Audio display
            out_audio = widgets.Output()
            with out_audio:
                display(Audio(data=item['y'], rate=item['sr']))
            
            # Combine Badge and Audio Player
            row = widgets.HBox([badge_html, out_audio], layout=widgets.Layout(align_items='center', margin='2px 0'))
            rows.append(row)
            
        display(widgets.VBox(rows))
            
    else:
        print("🔇 No audio preview available.")
    
    return cluster_df

def play_segment(song_index: int, start_time: float, end_time: float, raaga_dir: Optional[str] = None):
    """
    Play a specific segment of a song.
    
    Args:
        song_index (int): Index of the song to play
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        raaga_dir (str, optional): Directory containing audio files
    """
    context = get_raaga_context(raaga_dir)
    try:
        master_df = pd.read_csv(context["crepe_csv"])
    except FileNotFoundError:
        print("❌ Data files not found.")
        return None

    song_data = master_df[master_df["Index"] == song_index]
    if song_data.empty:
        print(f"⚠️ No data for song index {song_index}")
        return None
        
    # Get audio path
    audio_path = song_data['AudioPath'].iloc[0] if 'AudioPath' in song_data.columns else None
    
    if not audio_path or not os.path.exists(audio_path):
        # Construct path if missing
        file_name = song_data['AudioFile'].iloc[0] if 'AudioFile' in song_data.columns else f"song_{song_index}.wav"
        
        # Try finding it in RAAGA_Vocals
        possible_paths = [
            os.path.join(raaga_dir if raaga_dir else "", file_name),
            os.path.join(context['vocals'], file_name)
        ]
        
        found = False
        for p in possible_paths:
            if os.path.exists(p):
                audio_path = p
                found = True
                break
        
        if not found:
            print(f"❌ Audio file not found: {file_name}")
            return None

    # Load and slice
    try:
        y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=(end_time - start_time))
        return Audio(y, rate=sr)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None


def play_song_with_breaks(song_index: int, raaga_dir: str, method: str = 'clustering',
                          break_duration: float = 0.3, max_segments: Optional[int] = None,
                          db_thresh: float = 10, top_n: int = 9, wait: int = 10, delta: float = 0.2,
                          beep: bool = True):
    """
    Play a song with breaks at segment boundaries.
    
    Args:
        song_index (int): Index of the song to play
        raaga_dir (str): Path to raaga directory
        method (str): Segmentation method - 'clustering' or 'offset'
                     - 'clustering': Uses segment boundaries from clustering (StartFrame/EndFrame)
                     - 'offset': Uses onset detection to find consonants (spectral energy changes)
        break_duration (float): Duration of silence/beep between segments in seconds (default: 0.3)
        max_segments (int, optional): Maximum number of segments to play. If None, plays all.
        db_thresh (float): NOT USED in current implementation (reserved for future)
        top_n (int): NOT USED in current implementation (reserved for future)
        wait (int): For offset method - minimum frames between onsets (default: 10)
                   - Lower (5-8): More splits, catches rapid syllables
                   - Higher (15-20): Fewer splits, only major consonants
        delta (float): For offset method - onset strength threshold (default: 0.2)
                      - Lower (0.1-0.15): More sensitive, catches subtle consonants
                      - Higher (0.3-0.5): Less sensitive, only strong consonants
        beep (bool): If True, plays a beep after each syllable. If False, uses silence (default: True)
    
    Returns:
        IPython.display.Audio object
    
    Example:
        # Play with clustering-based breaks and beeps
        play_song_with_breaks(0, raaga_dir=r'C:\\path\\to\\Mayamalavagowlai', method='clustering')
        
        # Play with onset detection (catches consonants) - DEFAULT SETTINGS
        play_song_with_breaks(0, raaga_dir=r'C:\\\\path\\\\to\\\\Mayamalavagowlai', 
                             method='offset', wait=10, delta=0.2, beep=True)
        
        # More sensitive (catches more syllables)
        play_song_with_breaks(0, raaga_dir=r'C:\\\\path\\\\to\\\\Mayamalavagowlai', 
                             method='offset', wait=5, delta=0.1)
        
        # Less sensitive (only major consonants)
        play_song_with_breaks(0, raaga_dir=r'C:\\\\path\\\\to\\\\Mayamalavagowlai', 
                             method='offset', wait=15, delta=0.3)
    """
    import numpy as np
    import librosa
    from IPython.display import Audio
    from scipy.signal import find_peaks
    
    # Validate method
    if method not in ['clustering', 'offset']:
        raise ValueError("method must be 'clustering' or 'offset'")
    
    # Construct the vocal directory path
    raaga_name = os.path.basename(raaga_dir)
    audio_dir = os.path.join(raaga_dir, f"{raaga_name}_Vocals")
    
    # Get context
    context = get_raaga_context(audio_dir)
    
    if method == 'clustering':
        # Use clustering-based segmentation
        carva_path = context["carva_csv"]
        
        if not Path(carva_path).exists():
            print(f"❌ ERROR: File not found at {carva_path}")
            return
        
        # Load data
        df = pd.read_csv(carva_path)
        df = normalize_carva_df(df)
        
        # Filter by song index
        song_df = df[df['Index'] == song_index].sort_values('StartFrame')
        
        if song_df.empty:
            print(f"⚠️ No segments found for song index {song_index}")
            return
        
        # Get audio path from CSV
        csv_audio_path = song_df.iloc[0]['AudioPath']
        
        # Fix the path - audio is in RAAGA_Vocals, not in CSVs folder
        audio_filename = os.path.basename(csv_audio_path)
        audio_path = os.path.join(audio_dir, audio_filename)
        
    else:  # offset method
        # Use onset detection like explore_syllables
        crepe_csv_path = context["crepe_csv"]
        
        if not Path(crepe_csv_path).exists():
            print(f"❌ ERROR: File not found at {crepe_csv_path}")
            return
        
        # Load CREPE data
        df_master = pd.read_csv(crepe_csv_path)
        song_df_full = df_master[df_master['Index'] == song_index]
        
        if song_df_full.empty:
            print(f"⚠️ No data found for song index {song_index}")
            return
        
        # Get audio path from CSV
        csv_audio_path = song_df_full.iloc[0]['AudioPath']
        
        # Fix the path - audio is in RAAGA_Vocals, not in CSVs folder
        audio_filename = os.path.basename(csv_audio_path)
        audio_path = os.path.join(audio_dir, audio_filename)
    
    # Check audio file exists
    if not Path(audio_path).exists():
        print(f"❌ ERROR: Audio file not found at {audio_path}")
        return
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['sample_rate'])
    
    if method == 'offset':
        # Use librosa onset detection (better for consonants)
        # This detects sudden changes in spectral energy, which happens at consonants
        
        # Onset detection on the audio signal
        onset_frames = librosa.onset.onset_detect(
            y=y, 
            sr=sr,
            wait=wait,           # Minimum frames between onsets
            delta=delta,         # Threshold for peak picking
            units='frames',
            hop_length=512,      # Standard hop length
            backtrack=True       # Backtrack to find precise onset
        )
        
        # Convert frames to time
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        
        # Create syllable boundaries from onsets
        syllables = []
        if len(onset_times) > 0:
            # First segment: from start to first onset
            if onset_times[0] > 0.1:  # Only if there's significant audio before first onset
                syllables.append((0.0, onset_times[0]))
            
            # Middle segments: between consecutive onsets
            for i in range(len(onset_times) - 1):
                syllables.append((onset_times[i], onset_times[i + 1]))
            
            # Last segment: from last onset to end
            syllables.append((onset_times[-1], len(y) / sr))
        else:
            # No onsets found, use entire song as one segment
            syllables = [(0.0, len(y) / sr)]
        
        # Limit segments if requested
        if max_segments is not None:
            syllables = syllables[:max_segments]
        
        print(f"🎵 Playing {len(syllables)} segments (onset detection) with {'beeps' if beep else 'silence'}")
        
    else:  # clustering method
        # Limit segments if requested
        if max_segments is not None:
            song_df = song_df.head(max_segments)
        
        print(f"🎵 Playing {len(song_df)} segments (clustering method) with {'beeps' if beep else 'silence'}")
    
    # Create beep or silence for breaks
    break_samples = int(break_duration * sr)
    
    if beep:
        # Generate a beep sound (1000 Hz sine wave)
        beep_freq = 1000  # Hz
        t = np.linspace(0, break_duration, break_samples, False)
        beep_sound = 0.3 * np.sin(2 * np.pi * beep_freq * t)  # 0.3 amplitude to not be too loud
    else:
        beep_sound = np.zeros(break_samples)
    
    # Build audio with breaks
    audio_segments = []
    
    if method == 'offset':
        # Use time-based syllables
        for start_time, end_time in syllables:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Ensure bounds
            start_sample = max(0, min(len(y), start_sample))
            end_sample = max(0, min(len(y), end_sample))
            
            # Extract segment
            segment = y[start_sample:end_sample]
            
            # Add segment and beep/silence
            audio_segments.append(segment)
            audio_segments.append(beep_sound)
    
    else:  # clustering method
        # Use frame-based segments
        for idx, row in song_df.iterrows():
            start_frame = int(row['StartFrame'])
            end_frame = int(row['EndFrame'])
            
            # Convert frames to samples (assuming CREPE step_size of 20ms)
            step_size_ms = CREPE_CONFIG['step_size']
            start_sample = int(start_frame * step_size_ms * sr / 1000)
            end_sample = int(end_frame * step_size_ms * sr / 1000)
            
            # Extract segment
            segment = y[start_sample:end_sample]
            
            # Add segment and beep/silence
            audio_segments.append(segment)
            audio_segments.append(beep_sound)
    
    # Concatenate all segments
    final_audio = np.concatenate(audio_segments)
    
    # Plot distribution of segment lengths in CREPE frames
    import matplotlib.pyplot as plt
    
    # Calculate segment lengths in CREPE frames (20ms per frame)
    step_size_ms = CREPE_CONFIG['step_size']  # 20ms
    
    if method == 'offset':
        # For offset method, calculate from time-based syllables
        segment_lengths_frames = []
        for start_time, end_time in syllables:
            duration_ms = (end_time - start_time) * 1000  # Convert to ms
            num_frames = int(duration_ms / step_size_ms)
            segment_lengths_frames.append(num_frames)
    else:
        # For clustering method, use frame-based segments
        segment_lengths_frames = []
        for idx, row in song_df.iterrows():
            num_frames = int(row['EndFrame']) - int(row['StartFrame'])
            segment_lengths_frames.append(num_frames)
    
    # Create histogram
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(segment_lengths_frames, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Segment Length (CREPE frames)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of Segment Lengths\n({method} method)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_len = np.mean(segment_lengths_frames)
    median_len = np.median(segment_lengths_frames)
    min_len = np.min(segment_lengths_frames)
    max_len = np.max(segment_lengths_frames)
    
    stats_text = f'Mean: {mean_len:.1f} frames\nMedian: {median_len:.1f} frames\nMin: {min_len} frames\nMax: {max_len} frames'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(segment_lengths_frames, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel('Segment Length (CREPE frames)', fontsize=12)
    plt.title('Box Plot of Segment Lengths', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add frame duration reference
    duration_text = f'1 frame = {step_size_ms}ms\n{mean_len:.1f} frames ≈ {mean_len * step_size_ms:.0f}ms'
    plt.text(0.5, -0.15, duration_text, transform=plt.gca().transAxes,
             fontsize=9, ha='center', style='italic', color='gray')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Segment Statistics:")
    print(f"   Total segments: {len(segment_lengths_frames)}")
    print(f"   Mean length: {mean_len:.1f} frames ({mean_len * step_size_ms:.0f}ms)")
    print(f"   Median length: {median_len:.1f} frames ({median_len * step_size_ms:.0f}ms)")
    print(f"   Range: {min_len}-{max_len} frames ({min_len * step_size_ms:.0f}-{max_len * step_size_ms:.0f}ms)")
    
    # Display and return audio player
    return Audio(final_audio, rate=sr)

def plot_single_tertiary_cluster(audio_dir: str, shape_id: str, extrema: bool = True, density: float = 1.0, synthesize: bool = False, tonic: Optional[str] = None, raaga_mask: Optional[list] = None):
    """
    Plots ALL interpolated segments for a SPECIFIC Tertiary Shape ID.
    Uses Plotly for INTERACTIVE visualization (Hover shows Song Name & Frames).
    Plots absolute values (No Normalization) to show pitch height.
    Overlays 'Carnatic Bars' (Swara Grid).
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        import plotly.colors as pcolors
        import ipywidgets as widgets
        from IPython.display import display
        pio.renderers.default = 'notebook'
    except ImportError:
        print("❌ Plotly/Widgets not found. Please install: pip install plotly ipywidgets nbformat")
        return

    context = get_raaga_context(audio_dir)
    try: 
        df = pd.read_csv(context["carva_csv"])
        master_df = pd.read_csv(context["crepe_csv"]) # Need Master for Metadata lookup if not in Carva
    except: 
        print("Error reading data."); return

    df = normalize_carva_df(df) # Ensure columns are consistent
    
    # Parse (Now using standardized 'Tertiary_Label')
    # It should already be numeric strings after normalization, but safe to cast
    df['Meta_ShapeID'] = df['Tertiary_Label'].astype(str)
    
    # Filter
    target_rows = df[df['Meta_ShapeID'] == str(shape_id)]
    if target_rows.empty:
        print(f"❌ Shape ID {shape_id} not found."); return

    # Density Sampling
    if density < 1.0:
        n_sample = int(len(target_rows) * density)
        target_rows = target_rows.sample(n=max(1, n_sample), random_state=None)
        
    print(f"📊 Plotting Shape {shape_id} (n={len(target_rows)} segments, Using Plotly)...")
    
    
    # Initialize Figure
    fig = go.Figure()
    
    # ---------------- RAGA SCALES \u0026 MASKING ----------------
    # Full Chromatic Scale (Index 0-11)
    # Ratios approximated for Just Intonation / Carnatic
    chromatic_ratios = [
        ("Sa", 1.0), ("Ri1", 1.059), ("Ri2", 1.118), ("Ga2", 1.189), ("Ga3", 1.260),
        ("Ma1", 1.335), ("Ma2", 1.414), ("Pa", 1.500), ("Da1", 1.587), ("Da2", 1.682),
        ("Ni2", 1.782), ("Ni3", 1.888)
    ]
    
    # Build Allowed Ratios based on Mask
    allowed_ratios = {} # mapping value -> name
    
    if raaga_mask and len(raaga_mask) >= 12:
        # User Provided Mask
        for i, bit in enumerate(raaga_mask[:12]):
            if bit == 1:
                name, rat = chromatic_ratios[i]
                allowed_ratios[rat] = name
                # Add 2nd Octave
                allowed_ratios[rat * 2] = name + "'"
                
        # Always add Octave Sa
        allowed_ratios[2.0] = "Sa'"
        allowed_ratios[4.0] = "Sa''"
    else:
        # Default: Show All Common Carnatic Notes (Mayamalavagowla-ish fallback or All)
        # Fallback to the dictionary we had, expanded
        base_map = {
            "Sa": 1.0, "Ri1": 1.059, "Ri2": 1.122, "Ga2": 1.189, "Ga3": 1.260,
            "Ma1": 1.335, "Ma2": 1.414, "Pa": 1.500, "Da1": 1.587, "Da2": 1.682,
            "Ni2": 1.782, "Ni3": 1.888
        }
        for n, r in base_map.items():
            allowed_ratios[r] = n
            allowed_ratios[r*2] = n + "'"
        allowed_ratios[2.0] = "Sa'" # Ensure overlap is fine
        allowed_ratios[4.0] = "Sa''"

    # Add Horizontal Grid Lines (Allowed Notes)
    for ratio, swara in allowed_ratios.items():
        if ratio <= 4.0: # Limit to 2 octaves
            fig.add_hline(y=ratio, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text=swara, annotation_position="top left")

    # Add Vertical Grid Lines (8 Divisions)
    for v in range(0, 65, 8):
        fig.add_vline(x=v, line_dash="dot", line_color="rgba(255,255,0,0.4)", annotation_text=f"{v}", annotation_position="bottom right")

    # Colors
    colors = pcolors.qualitative.Plotly * 10
    
    # Widgets Setup
    out_audio = widgets.Output() 
    buttons_list = []
    collected_sequences = [] # Store sequence data for HTML display

    # Add Segments
    for i, (_, row) in enumerate(target_rows.iterrows()):
        try:
            # Metadata
            song_idx = row['Index']
            start_f = row['StartFrame']
            end_f = row['EndFrame']
            seg_color = colors[i % len(colors)]
            
            song_name = "Unknown"
            lookup = master_df[master_df['Index'] == song_idx]
            if not lookup.empty: song_name = lookup.iloc[0]['SongName']
            
            # Interpolate (0 to 64)
            seg = np.array(json.loads(row['SegmentList']))
            x_old = np.linspace(0, 64, len(seg))
            x_new = np.linspace(0, 64, 64) 
            seg_interp = np.interp(x_new, x_old, seg)
            
            # --- NOTE EXTRACTION (Per 8-Division) ---
            # Use Helper
            note_sequence, all_extrema_idxs = get_note_sequence(seg_interp, allowed_ratios)
            seq_str = " ".join(note_sequence)
            
            hover_text = (
                f"<b>Song:</b> {song_name}<br>"
                f"<b>Frames:</b> {start_f}-{end_f}<br>"
                f"<b>Time:</b> {start_f*0.02:.2f}s<br>"
                f"<b>Seq:</b> {seq_str}"
            )
            
            
            # Collect for HTML Display
            dur_frames = end_f - start_f
            dur_sec = dur_frames * 0.02
            
            collected_sequences.append({
                "name": f"Seg {i}",
                "seq": seq_str,
                "color": seg_color,
                "dur": f"{dur_frames}f / {dur_sec:.2f}s"
            })
            
            # Line Trace
            fig.add_trace(go.Scatter(
                x=x_new,
                y=seg_interp,
                mode='lines', # Removed +text to avoid overcrowding
                line=dict(color=seg_color, width=2),
                name=f"Seg {i}: {seq_str}", 
                text=[hover_text]*64, 
                hovertemplate=hover_text,
                showlegend=True
            ))
            
            # Add Sequence Annotation Text (At top of plot or near line?)
            # User said "print that in the same color".
            # Adding it to the legend name is good.
            # Adding annotations to the plot might get crowded.
            # Let's add markers for the detected notes.
            
            # Extrema Markers
            if extrema:
                # Use global extrema returned by helper
                if len(all_extrema_idxs) > 0:
                     ex_x = x_new[all_extrema_idxs]
                     ex_y = seg_interp[all_extrema_idxs]
                     
                     # Extrema Hover Text
                     ex_texts = []
                     for ey in ex_y:
                         note = get_closest_allowed_note(ey, allowed_ratios)
                         ex_texts.append(f"<b>{note}</b> ({ey:.2f})")
                     
                     fig.add_trace(go.Scatter(
                        x=ex_x,
                        y=ex_y,
                        mode='markers',
                        marker=dict(color=seg_color, size=6, symbol='circle-open', line=dict(width=2)),
                        text=ex_texts,
                        hoverinfo="text",
                        showlegend=False
                     ))
            
            # Create Play Button for this Segment
            # We must capture variables in closure
            def make_callback(idxs, sf, ef, s_name, seg_i):
                return lambda b: play_in_output(idxs, sf, ef, s_name, seg_i)
                
            btn = widgets.Button(
                description=f"▶ {i}", 
                layout=widgets.Layout(width='60px'),
                style=dict(button_color=seg_color, font_weight='bold') # Colored button!
            )
            
            # Define what happens on click
            def play_in_output(s_idx, s_frame, e_frame, s_name, seg_id):
                with out_audio:
                    out_audio.clear_output(wait=True)
                    print(f"🎵 Segment {seg_id} ({s_name})")
                    # Read current synth/tonic values from widget (defined below) or args
                    # For simplicity, we use the arguments passed to function initially OR global state?
                    # Better: Read from the toggle widgets we will create
                    is_synth = synth_checkbox.value
                    tgt_tonic = tonic_text.value if tonic_text.value.strip() else None
                    
                    play_segment(
                        audio_dir=audio_dir, 
                        song_index=s_idx, 
                        start_frame=s_frame, 
                        end_frame=e_frame, 
                        plot=False, 
                        synthesize=is_synth, 
                        tonic=tgt_tonic
                    )
                    
            btn.on_click(make_callback(song_idx, start_f, end_f, song_name, i))
            buttons_list.append(btn)
            
        except Exception as e: 
            print(f"Error plotting segment {i}: {e}")
            pass
            
    fig.update_layout(
        title=f"Tertiary Shape {shape_id}: Pitch Distribution (Interactive)",
        xaxis_title="Segment Index (0-64)",
        yaxis_title="Pitch Ratio",
        template="plotly_dark",
        showlegend=False,
        height=600,
        hovermode="closest",
        xaxis=dict(range=[0, 64])
    )
    
    # Controls UI
    synth_checkbox = widgets.Checkbox(value=synthesize, description='Synthesize')
    tonic_text = widgets.Text(value=tonic if tonic else "", placeholder='Tonic (e.g. D3)', description='Tonic:')
    
    controls = widgets.HBox([synth_checkbox, tonic_text])
    button_grid = widgets.HBox(buttons_list, layout=widgets.Layout(flex_flow='row wrap'))

    # Sequence Info Widget
    html_str = "<div style='max-height: 200px; overflow-y: auto; border: 1px solid #444; padding: 10px; margin-bottom: 10px;'>"
    html_str += "<h4 style='margin-top:0;'>Extracted Sequences:</h4>"
    for item in collected_sequences:
        html_str += f"<div style='color:{item['color']}; font-family: monospace; font-size: 14px;'><b>{item['name']}</b> ({item['dur']}): {item['seq']}</div>"
    html_str += "</div>"
    
    seq_widget = widgets.HTML(html_str)

    # Display Everything
    display(go.FigureWidget(fig))
    display(seq_widget)
    print("------- Playback Controls -------")
    display(controls)
    display(button_grid)
    display(out_audio)
    
def annotate_clusters_widget(audio_dir: str, raaga_mask: Optional[list] = None):
    """
    Full UI Widget for Annotating Clusters.
    - Shape Selector
    - Plot Visualization
    - Editable Note Sequences (Bio-Input)
    - Save to CSV
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        import plotly.colors as pcolors
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        pio.renderers.default = 'notebook'
    except ImportError:
        print("❌ Plotly/Widgets not found.")
        return

    context = get_raaga_context(audio_dir)
    try: 
        carva_df = pd.read_csv(context["carva_csv"])
        master_df = pd.read_csv(context["crepe_csv"]) 
    except: 
        print("Error reading data."); return

    carva_df = normalize_carva_df(carva_df)
    
    if 'Tertiary_Label' not in carva_df.columns: return
    
    # Pre-processing (Already handled by normalize, but ensure string for dropdown)
    carva_df['Meta_ShapeID'] = carva_df['Tertiary_Label'].astype(str)
    
    # Get Unique Shapes and sort by frequency
    shape_counts = carva_df['Meta_ShapeID'].value_counts()
    unique_shapes = sorted(shape_counts.index.tolist(), key=lambda x: int(x) if x.isdigit() else 9999)
    
    # --- BOOKMARKING ---
    bookmark_path = context["carva_csv"].replace(".csv", "_bookmark.json")
    last_shape = unique_shapes[0] if unique_shapes else "-1"
    
    if os.path.exists(bookmark_path):
        try:
            with open(bookmark_path, 'r') as f:
                data = json.load(f)
                saved_shape = str(data.get("last_shape", ""))
                if saved_shape in unique_shapes:
                    last_shape = saved_shape
        except Exception as e:
            print(f"⚠️ Could not load bookmark: {e}")

    # ---------------- UI COMPONENTS ----------------
    
    # 1. Header
    dropdown = widgets.Dropdown(options=unique_shapes, value=last_shape, description='Shape ID:')
    btn_prev = widgets.Button(description='< Prev', layout=widgets.Layout(width='80px'))
    btn_next = widgets.Button(description='Next >', layout=widgets.Layout(width='80px'))
    header = widgets.HBox([btn_prev, dropdown, btn_next])
    
    # 2. Controls (Global)
    chk_synth = widgets.Checkbox(value=False, description='Synthesize', layout=widgets.Layout(width='100px'))
    txt_tonic = widgets.Text(value="", placeholder='Tonic', description='Tonic:', layout=widgets.Layout(width='120px'))
    
    # Batch Labeling
    txt_batch = widgets.Text(value="", placeholder='Global Label (e.g. Sa Ri Ga)', layout=widgets.Layout(width='200px'))
    btn_batch_apply = widgets.Button(description='Apply All', icon='arrow-down', layout=widgets.Layout(width='100px'))
    
    controls_box = widgets.HBox([chk_synth, txt_tonic, widgets.Label(" | "), txt_batch, btn_batch_apply])

    # 3. Output Areas
    out_plot = widgets.Output()
    out_audio = widgets.Output() # Dedicated audio output to prevent piling up
    out_editors = widgets.Output() # Scrollable area for editors
    status_label = widgets.Label(value="Ready.")
    
    # 4. State
    current_text_widgets = [] # Tuples of (index, widget)
    
    # 5. Save Button
    btn_save = widgets.Button(description='SAVE ANNOTATIONS', button_style='success', icon='check', layout=widgets.Layout(width='200px'))
    
    # ---------------- LOGIC ----------------
    
    def on_shape_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            new_shape = change['new']
            render_shape(new_shape)
            
    dropdown.observe(on_shape_change)
    
    def go_prev(b):
        opts = dropdown.options
        curr_idx = opts.index(dropdown.value)
        if curr_idx > 0: dropdown.value = opts[curr_idx - 1]
            
    def go_next(b):
        opts = dropdown.options
        curr_idx = opts.index(dropdown.value)
        if curr_idx < len(opts) - 1: dropdown.value = opts[curr_idx + 1]
            
    btn_prev.on_click(go_prev)
    btn_next.on_click(go_next)
    
    # Batch Apply Logic for 8-Box System
    def apply_batch(b):
        val = txt_batch.value
        if not val: return
        
        # Split input by space to distribute across 8 boxes
        parts = val.split()
        # Pad with underscores or empty strings? User probably wants underscores from extraction
        # Let's just use what they provided, cycle or stop?
        # Usually user provides "S R G M" -> fills first 4, rest empty? or "_"
        
        for _, boxes in current_text_widgets:
            for k in range(8):
                if k < len(parts): boxes[k].value = parts[k]
                else: boxes[k].value = "_" # Default to underscore if not provided
        
        status_label.value = f"Applied '{val}' to all visible segments."
        
    btn_batch_apply.on_click(apply_batch)
    
    def save_action(b):
        status_label.value = "Saving..."
        count = 0
        try:
            # Check if column exists
            if 'Verified_Sequence' not in carva_df.columns:
                carva_df['Verified_Sequence'] = "" 
                
            for idx, box_list in current_text_widgets:
                # Join 8 boxes with space
                notes = [w.value.strip() if w.value.strip() else "_" for w in box_list]
                user_seq = " ".join(notes)
                carva_df.at[idx, 'Verified_Sequence'] = user_seq
                count += 1
                
            # Save File
            carva_df.to_csv(context["carva_csv"], index=False)
            
            # Save Bookmark (User requested: Only save bookmark on Save Click)
            try:
                with open(bookmark_path, 'w') as f:
                    json.dump({"last_shape": str(dropdown.value)}, f)
            except: pass
            
            status_label.value = f"✅ Saved {count} rows & Bookmarked Shape {dropdown.value}!"
            
            # Optional: Refresh to show saved state?
            # render_shape(dropdown.value) 
        except Exception as e:
            status_label.value = f"❌ Error Saving: {e}"

    btn_save.on_click(save_action)

    def render_shape(shape_id):
        nonlocal current_text_widgets
        current_text_widgets = []
        
        out_plot.clear_output(wait=True)
        out_editors.clear_output(wait=True)
        status_label.value = f"Loading Shape {shape_id}..."
        
        # Filter Data
        target_rows = carva_df[carva_df['Meta_ShapeID'] == str(shape_id)]
        
        if target_rows.empty:
            with out_plot: print("No data for this shape.")
            return

        # --- RE-USE PLOTTING LOGIC (Simplified) ---
        # We need to construct the figure manually here or refactor plot function to return fig.
        # Copying simplified logic for robustness inside the widget.
        
        with out_plot:
            fig = go.Figure()
            
            # Grid Setup
            chromatic_ratios = [
                ("Sa", 1.0), ("Ri1", 1.059), ("Ri2", 1.118), ("Ga2", 1.189), ("Ga3", 1.260),
                ("Ma1", 1.335), ("Ma2", 1.414), ("Pa", 1.500), ("Da1", 1.587), ("Da2", 1.682),
                ("Ni2", 1.782), ("Ni3", 1.888)
            ]
            allowed_ratios = {}
            if raaga_mask:
                for i, bit in enumerate(raaga_mask[:12]):
                    if bit == 1:
                        n, r = chromatic_ratios[i]
                        allowed_ratios[r] = n
                        allowed_ratios[r*2] = n + "'"
                allowed_ratios[2.0] = "Sa'"
                allowed_ratios[4.0] = "Sa''"
            else:
                 # Default
                 for n, r in chromatic_ratios: 
                     allowed_ratios[r] = n
                     allowed_ratios[r*2] = n + "'"
                 allowed_ratios[2.0] = "Sa'"

            # Add Horizontal Grid
            for r, n in allowed_ratios.items():
                if r <= 4.0: fig.add_hline(y=r, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text=n, annotation_position="top left")

             # Add Vertical Grid
            for v in range(0, 65, 8):
                fig.add_vline(x=v, line_dash="dot", line_color="rgba(255,255,0,0.4)", annotation_text=f"{v}", annotation_position="bottom right")

            colors = pcolors.qualitative.Plotly * 10
            
            editor_rows = []
            
            # --- HEADER ROW ---
            # Align with row items: [Play (50px)] [Info (60px)] [Dup (30px)] [8 Boxes (40px each)]
            h_play = widgets.Label("Play", layout=widgets.Layout(width='50px'))
            h_info = widgets.Label("Seg", layout=widgets.Layout(width='60px'))
            h_dup = widgets.Label("Cp", layout=widgets.Layout(width='30px'))
            h_boxes = [widgets.Label(str(k+1), layout=widgets.Layout(width='40px')) for k in range(8)]
            editor_rows.append(widgets.HBox([h_play, h_info, h_dup] + h_boxes))
            
            for i, (idx, row) in enumerate(target_rows.iterrows()): # idx is DF Index
                try:
                    seg_color = colors[i % len(colors)]
                    
                    # Data
                    seg = np.array(json.loads(row['SegmentList']))
                    x_new = np.linspace(0, 64, 64) 
                    seg_interp = np.interp(x_new, np.linspace(0, 64, len(seg)), seg)
                    
                    # Auto Extract
                    note_seq, all_extrema_idxs = get_note_sequence(seg_interp, allowed_ratios)
                    auto_str = " ".join(note_seq)
                    
                    # Verify if existing annotation exists
                    current_val = auto_str
                    if 'Verified_Sequence' in carva_df.columns:
                        saved_val = str(row['Verified_Sequence'])
                        if saved_val and saved_val != "nan" and saved_val.strip():
                             current_val = saved_val
                    
                    # Plot Trace
                    fig.add_trace(go.Scatter(x=x_new, y=seg_interp, mode='lines', line=dict(color=seg_color, width=2), name=f"Seg {i}"))
                    
                             # Plot Extrema Markers
                    if len(all_extrema_idxs) > 0:
                        ex_x = x_new[all_extrema_idxs]
                        ex_y = seg_interp[all_extrema_idxs]
                        ex_texts = []
                        for ey in ex_y:
                             note = get_closest_allowed_note(ey, allowed_ratios)
                             ex_texts.append(f"<b>{note}</b> ({ey:.2f})")
                             
                        fig.add_trace(go.Scatter(
                            x=ex_x, y=ex_y, mode='markers',
                            marker=dict(color=seg_color, size=6, symbol='circle-open', line=dict(width=2)),
                            text=ex_texts, hoverinfo="text", showlegend=False
                        ))
                    
                    # Create Editor Row
                    # Layout: [Play Btn] [Info Label] [Text Input]
                    
                    # Play Button
                    btn_play = widgets.Button(description=f'▶ {i}', layout=widgets.Layout(width='50px'), style=dict(button_color=seg_color, font_weight='bold'))
                    
                    # Play Callback
                    # Play Callback
                    def mk_play(s_idx, frames, color_hex):
                        def on_click(b):
                            with out_audio:
                                out_audio.clear_output(wait=True)
                                # Colored HTML Label
                                display(widgets.HTML(f"<b style='color:{color_hex}; font-size:14px'>🎵 Playing Segment {i}</b>"))
                                
                                play_segment(
                                    audio_dir, s_idx, frames[0], frames[1], 
                                    plot=False, 
                                    synthesize=chk_synth.value, 
                                    tonic=txt_tonic.value if txt_tonic.value else None
                                )
                        return on_click
                    
                    btn_play.on_click(mk_play(row['Index'], (row['StartFrame'], row['EndFrame']), seg_color))
                    
                    # Info Label
                    dur = f"{(row['EndFrame']-row['StartFrame'])*0.02:.2f}s"
                    lbl_info = widgets.Label(value=f"{i} ({dur})", layout=widgets.Layout(width='60px'))
                    
                    # 8-Box Text Input
                    # Split current val (space separated)
                    curr_parts = current_val.split()
                    # Ensure 8 parts
                    while len(curr_parts) < 8: curr_parts.append("_")
                    curr_parts = curr_parts[:8]
                    
                    box_list = []
                    for k in range(8):
                        # Small text box
                        tb = widgets.Text(value=curr_parts[k], layout=widgets.Layout(width='40px'))
                        box_list.append(tb)
                    
                    # Store list of widgets for saving
                    current_text_widgets.append((idx, box_list))
                    
                    # Duplicate Button (Copies from Previous Row)
                    btn_dup = widgets.Button(description='〃', layout=widgets.Layout(width='30px'))
                    
                    if i > 0:
                        def mk_dup(curr_boxes, prev_boxes):
                            def copy_prev(b):
                                for k in range(8):
                                    curr_boxes[k].value = prev_boxes[k].value
                            return copy_prev
                        
                        # Get prev widget list. Access via current_text_widgets[-2] because we just appended current
                        prev_widgets = current_text_widgets[-2][1]
                        btn_dup.on_click(mk_dup(box_list, prev_widgets))
                    else:
                        btn_dup.disabled = True
                    
                    # Assemble Row: Play | Info | Dup | 8 Boxes
                    row_items = [btn_play, lbl_info, btn_dup] + box_list
                    row_box = widgets.HBox(row_items)
                    editor_rows.append(row_box)
                    
                except: pass
                
            fig.update_layout(template="plotly_dark", height=400, showlegend=False, margin=dict(l=0,r=0,t=30,b=0))
            display(go.FigureWidget(fig))
            
        with out_editors:
            # --- TABBED INTERFACE ---
            
            # Tab 1: Interactive Grid
            vbox_grid = widgets.VBox(editor_rows)
            
            # Tab 2: Spreadsheet / Bulk Text
            txt_sheet = widgets.Textarea(placeholder="Paste from Excel here (rows of 8 columns)...", layout=widgets.Layout(width='95%', height='400px'))
            btn_sync_to_sheet = widgets.Button(description="⬇️ Load from Grid", button_style='info')
            btn_sync_from_sheet = widgets.Button(description="⬆️ Apply to Grid", button_style='warning')
            lbl_sheet_status = widgets.Label()
            
            def load_sheet(b):
                # Dump current widgets to text (Tab separated)
                lines = []
                for idx, boxes in current_text_widgets:
                    vals = [b.value.strip() if b.value.strip() else "_" for b in boxes]
                    lines.append("\t".join(vals))
                txt_sheet.value = "\n".join(lines)
                lbl_sheet_status.value = "Loaded data from Grid."
                
            def apply_sheet(b):
                # Parse text to widgets
                text = txt_sheet.value
                if not text: return
                lines = text.strip().split('\n')
                
                # Check length match
                if len(lines) > len(current_text_widgets):
                    lbl_sheet_status.value = f"⚠️ Input has {len(lines)} rows, but Grid has {len(current_text_widgets)}. Truncating/Ignoring extras."
                
                count = 0
                for i, line in enumerate(lines):
                    if i >= len(current_text_widgets): break
                    
                    # Split by tab or multiple spaces
                    # Use regex or simple split? Tab is standard for Excel copy. Space might be ambiguous if notes have spaces (rare).
                    # Let's try tab first, fall back to whitespace if no tabs?
                    if '\t' in line: parts = line.split('\t')
                    else: parts = line.split() # any whitespace
                    
                    # Fill boxes
                    _, boxes = current_text_widgets[i]
                    for k in range(8):
                        val = parts[k].strip() if k < len(parts) else "_"
                        boxes[k].value = val if val else "_"
                    count += 1
                
                lbl_sheet_status.value = f"✅ Updated {count} rows from Sheet."
                
            btn_sync_to_sheet.on_click(load_sheet)
            btn_sync_from_sheet.on_click(apply_sheet)
            
            vbox_sheet = widgets.VBox([
                widgets.HBox([btn_sync_to_sheet, btn_sync_from_sheet, lbl_sheet_status]),
                txt_sheet
            ])
            
            # Assemble Tabs
            tabs = widgets.Tab(children=[vbox_grid, vbox_sheet])
            tabs.set_title(0, "Interactive Grid")
            tabs.set_title(1, "Spreadsheet (Bulk Copy/Paste)")
            
            display(tabs)
            
        status_label.value = f"Loaded {len(target_rows)} segments for Shape {shape_id}."

    # Initial Render
    display(widgets.VBox([header, controls_box])) # Header + Controls combined
    display(out_plot)
    display(out_audio) # Playback appears here
    display(widgets.Label("--- Note Editors ---"))
    display(out_editors)
    display(widgets.Label("--------------------"))
    display(widgets.HBox([status_label, btn_save]))
    
    # Trigger first load (Use loaded bookmark)
    render_shape(last_shape)

# ============================================================================

# ANALYSIS & VISUALIZATION

# ============================================================================

def plot_carnatic_segment(song_data, start_frame, end_frame, title_suffix=""):
    print(f"   (Plotting segment {start_frame}-{end_frame} {title_suffix} - Placeholder)")


# ============================================================================

# STANDALONE UTILITIES (Extracted for use by Widget and Notebook)

# ============================================================================

def synthesize_tone(ratios, sr=44100, base_freq=261.63):

    """Synthesize a rich tone from pitch ratios."""

    freqs = np.array(ratios) * base_freq

    phases = np.cumsum(2 * np.pi * freqs / sr)

    # Fundamental + Harmonics

    y = 0.6 * np.sin(phases) + 0.3 * np.sin(2 * phases) + 0.1 * np.sin(3 * phases)

    # Envelope

    env = np.ones_like(y)

    fade_len = int(0.01 * sr)

    if len(env) > 2 * fade_len:

        env[:fade_len] = np.linspace(0, 1, fade_len)

        env[-fade_len:] = np.linspace(1, 0, fade_len)

    else:

        env = np.hanning(len(env))

    return y * env

def generate_drone(duration_samples, sr=44100, base_freq=261.63):

    """Generate C4 + G4 Tambura-style drone."""

    t = np.arange(duration_samples) / sr

    sa = base_freq

    pa = sa * 1.5

    y_drone = 0.4 * np.sin(2 * np.pi * sa * t) + \
              0.3 * np.sin(2 * np.pi * pa * t) + \
              0.1 * np.sin(2 * np.pi * sa * 2 * t)

    return y_drone * 0.15


class SongReconstructionWidget:
    """
    Reconstructs a song by playing a RANDOM segment from the assigned cluster 
    for each time step. 
    
    This verifies 'Cluster Consistency'. If the reconstruction sounds like the 
    original melody (in C4), then the clusters are pure.
    """
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self.context = get_raaga_context(audio_dir)
        try:
            self.carva_df = pd.read_csv(self.context["carva_csv"])
            self.master_df = pd.read_csv(self.context["crepe_csv"])
        except:
            print("Error loading data."); return

        # UI Elements
        self.song_indices = sorted(self.carva_df['Index'].unique())
        self.dropdown_song = widgets.Dropdown(options=self.song_indices, description="Song Index:")
        self.btn_play = widgets.Button(description="▶️ Reconstruct & Play", button_style="success", icon="play")
        self.out = widgets.Output()
        
        self.btn_play.on_click(self.on_play)
        
        self.ui = widgets.VBox([
            widgets.HTML("<h3>🎵 Cluster Reconstruction (C4)</h3>"),
            widgets.HBox([self.dropdown_song, self.btn_play]),
            self.out
        ])
        
    def show(self):
        display(self.ui)
        
    def on_play(self, b):
        with self.out:
            clear_output(wait=True)
            song_idx = self.dropdown_song.value
            print(f"Reconstructing Song {song_idx} using Cluster Clones...")
            
            # 1. Get Sequence
            song_segments = self.carva_df[self.carva_df['Index'] == song_idx].sort_values('StartFrame')
            
            if song_segments.empty:
                print("No segments found for this song.")
                return

            label_col = next((col for col in ['PCA_Label', 'DTW_Label', 'Label'] if col in song_segments.columns), 'Label')
            print(f"Using Clustering Column: {label_col}")
            
            # 2. Build Audio
            sr = 44100
            full_audio = []
            
            # Pre-fetch all segments grouped by label to speed up random sampling
            # (Optimization: Don't query DF inside loop)
            # We need segments from ALL songs to pick random samples effectively?
            # User said "play a random segment from all clusters". Yes, global pool.
            global_clusters = self.carva_df.groupby(label_col)
            
            # Cache segments for fast access
            cluster_cache = {} # {label: [list_of_json_strings]}
            
            # 2. Build Audio Buffer (Non-Overlapping Sequence)
            sr = 44100
            step_sec = 0.02 
            full_audio = []
            
            # Cache segments
            global_clusters = self.carva_df.groupby(label_col)
            cluster_cache = {} 
            
            # State for tiling
            last_end_frame = 0 # Begin at 0
            
            # Filter Loop: Greedy Tiling
            # We iterate sorted segments. 
            # If a segment starts BEFORE the previous one ended, we SKIP it (Overlap).
            # If it starts AFTER, we add SILENCE, then Play it.
            
            count_played = 0
            
            for i, row in tqdm(song_segments.iterrows(), total=len(song_segments), desc="Synthesizing"):
                 start_frame = int(row['StartFrame'])
                 
                 # 1. Skip Overlaps
                 if start_frame < last_end_frame:
                     continue
                 
                 # 2. Handle Gap (Silence/Noise)
                 if start_frame > last_end_frame:
                     gap_frames = start_frame - last_end_frame
                     gap_samples = int(gap_frames * step_sec * sr)
                     if gap_samples > 0:
                         full_audio.append(np.zeros(gap_samples))
                 
                 # 3. Process This Segment
                 label = row[label_col]
                 
                 # If "Noise" (-1), we treat it as silence (already handled by gap if we skip?)
                 # Actually, if the best tiling segment is labelled -1, we should probably play silence 
                 # for its duration to maintain sync?
                 # User said: "parts that aren't coloured... should be silent".
                 
                 end_frame = int(row['EndFrame'])
                 duration_frames = end_frame - start_frame
                 duration_samples = int(duration_frames * step_sec * sr)
                 
                 if label == -1:
                     # Play Silence for this segment's duration
                     full_audio.append(np.zeros(duration_samples))
                     last_end_frame = end_frame
                     continue

                 # Fetch candidates
                 if label not in cluster_cache:
                     if label in global_clusters.groups:
                         cluster_cache[label] = global_clusters.get_group(label)['SegmentList'].values
                     else: cluster_cache[label] = []
                 
                 candidates = cluster_cache[label]
                 if len(candidates) == 0: 
                      # Fallback silence
                      full_audio.append(np.zeros(duration_samples))
                      last_end_frame = end_frame
                      continue
                      
                 chosen_seg_str = random.choice(candidates)
                 
                 try:
                     ratios = json.loads(chosen_seg_str)
                     
                     if duration_samples < 10: 
                         # Too short
                         full_audio.append(np.zeros(duration_samples))
                         last_end_frame = end_frame
                         continue
                     
                     # Interpolate Random Clone (len 64) -> Target Duration (len 50)
                     current_ratios = np.array(ratios)
                     x_old = np.linspace(0, 1, len(current_ratios))
                     x_new = np.linspace(0, 1, duration_samples)
                     ratios_stretched = np.interp(x_new, x_old, current_ratios)
                     
                     freqs = ratios_stretched * 261.63
                     phases = np.cumsum(2 * np.pi * freqs / sr)
                     tone = 0.5 * np.sin(phases)
                     
                     # Enveloping for smooth concatenation
                     fade_len = min(200, len(tone)//2)
                     tone[:fade_len] *= np.linspace(0, 1, fade_len)
                     tone[-fade_len:] *= np.linspace(1, 0, fade_len)
                     
                     full_audio.append(tone)
                     count_played += 1
                     
                 except:
                     full_audio.append(np.zeros(duration_samples))
                 
                 # Update Pointer
                 last_end_frame = end_frame
            
            if not full_audio:
                print("Failed to generate audio.")
                return
                
            final_mix = np.concatenate(full_audio)
            print(f"Playback Ready ({len(final_mix)/sr:.2f}s, {count_played} segments used)...")
            display(ipd.Audio(final_mix, rate=sr))

# ============================================================================
# SYLLABLE EXPLORATION
# ============================================================================
class SyllableExplorer:
    def __init__(self, audio_dir: str, song_index: int, wait: int = 10, delta: float = 0.2):
        self.audio_dir = audio_dir
        self.song_index = song_index
        self.wait = wait
        self.delta = delta
        
        self.context = get_raaga_context(audio_dir)
        try:
            # 1. Load Master CREPE Data
            self.master_df = pd.read_csv(self.context["crepe_csv"])
            self.song_df = self.master_df[self.master_df["Index"] == song_index]
            
            if self.song_df.empty:
                print("❌ Song not found in CREPE data.")
                return
                
            # Print Tonic
            tonic = self.song_df.iloc[0].get('Tonic', 'Unknown')
            self.tonic = float(tonic) if tonic != 'Unknown' else 200.0
            print(f"🎵 Song: {self.song_df.iloc[0]['SongName']} (Tonic: {tonic} Hz)")
                
            self.audio_path = self.song_df.iloc[0]["AudioPath"]
            self.song_name = self.song_df.iloc[0]["SongName"]
            
            # --- ROBUST AUDIO SELECTION (Discovery + Matching) ---
            import difflib
            
            # 1. Identify where the WAVs actually are
            valid_audio_dir = None
            candidate_dirs = [
                Path(self.audio_dir),
                Path(self.audio_dir).parent / f"{self.song_name}_Vocals",
                Path(self.audio_dir).parent.parent / self.song_name / f"{self.song_name}_Vocals",
                Path(self.audio_dir).parent / "Vocals"
            ]
            
            for c_dir in candidate_dirs:
                if c_dir.exists() and any(f.endswith('.wav') for f in os.listdir(c_dir)):
                     valid_audio_dir = c_dir
                     print(f"📂 Found Audio Directory: {valid_audio_dir}")
                     break
            
            if valid_audio_dir:
                # 2. Find the SPECIFIC file for this song
                target_fname = os.path.basename(self.audio_path) # Name from CSV
                available_files = [f for f in os.listdir(valid_audio_dir) if f.endswith('.wav')]
                
                # A. Exact Match
                if target_fname in available_files:
                     self.audio_path = str(valid_audio_dir / target_fname)
                     print(f"✅ Found Exact Match: {target_fname}")
                
                # B. Fuzzy Match (The user's original issue)
                else:
                     matches = difflib.get_close_matches(target_fname, available_files, n=1, cutoff=0.4)
                     if matches:
                         self.audio_path = str(valid_audio_dir / matches[0])
                         print(f"✨ Fuzzy Match: {target_fname} -> {matches[0]}")
                     else:
                         # C. Last Resort: First File (Only if requested, but dangerous for specific song analysis)
                         # Based on user "Using one audiofile and using another", we should avoid random picking.
                         print(f"❌ Could not match {target_fname} in directory. \n   Example available: {available_files[0] if available_files else 'None'}")
            else:
                 print(f"⚠️ No .wav files found in candidates. Using CSV path: {self.audio_path}")
            # ----------------------------------

            # 2. Load Cluster Data (CARVA)
            try:
                self.carva_df = pd.read_csv(self.context["carva_csv"])
                self.song_carva = self.carva_df[self.carva_df["Index"] == song_index]
                # Normalize columns if needed
                self.song_carva = normalize_carva_df(self.song_carva)
            except FileNotFoundError:
                print("⚠️ Clusters not found (carva.csv). Plotting raw pitch only.")
                self.song_carva = pd.DataFrame()

            # 3. Load Audio
            self.y, self.sr = librosa.load(self.audio_path, sr=None)
            self.duration = len(self.y) / self.sr
            
            # 4. Detect Onsets (Initial)
            self.detect_onsets()
                
            self.setup_ui()
            
        except Exception as e:
            print(f"Error initializing explorer: {e}")
            import traceback
            traceback.print_exc()

    def detect_onsets(self):
        print(f"🎵 Analying Syllables for: {self.song_name} (Delta={self.delta}, Wait={self.wait})...")
        self.onsets = librosa.onset.onset_detect(y=self.y, sr=self.sr, backtrack=True, units='time', wait=self.wait, delta=self.delta)
        print(f"✨ Detected {len(self.onsets)} syllables.")
        
        # 5. Build Syllable Segments
        self.syllables = []
        for i in range(len(self.onsets)):
            start = self.onsets[i]
            end = self.onsets[i+1] if i < len(self.onsets) - 1 else self.duration
            if end - start > 5.0: end = start + 5.0 
            self.syllables.append((start, end))

    def setup_ui(self):
        self.is_ready = False # Flag to prevent ghost triggers
        
        # 2. Controls
        self.options = [(f"{i+1}. {s:.2f}s - {e:.2f}s", i) for i, (s, e) in enumerate(self.syllables)]
        self.dropdown = widgets.Dropdown(options=self.options, description="Syllable:")
        self.btn_prev = widgets.Button(description="< Prev", layout=widgets.Layout(width='80px'))
        self.btn_next = widgets.Button(description="Next >", layout=widgets.Layout(width='80px'))
        
        # Segmentation Controls
        self.slider_onset_delta = widgets.FloatSlider(value=self.delta, min=0.01, max=1.0, step=0.01, description="Threshold:", layout=widgets.Layout(width='200px'))
        self.slider_onset_wait = widgets.IntSlider(value=self.wait, min=1, max=50, description="Min Dist:", layout=widgets.Layout(width='200px'))
        self.btn_resegment = widgets.Button(description="Re-Segment", button_style='info')
        self.btn_resegment.on_click(self.on_resegment)
        
        self.btn_play_breaks = widgets.Button(description="Play w/ Breaks", button_style='success', icon='play', layout=widgets.Layout(width='150px'))
        self.btn_play_breaks.on_click(self.on_play_breaks)
        
        # Motif Search UI (Restored)
        self.txt_motif = widgets.Text(description="Motif:", placeholder="e.g. Sa Ri1 Sa")
        self.btn_search_motif = widgets.Button(description="Fit Pattern")
        self.slider_weight = widgets.FloatSlider(value=0.05, min=0.0, max=1.0, step=0.01, description="Penalty:", readout_format='.2f')
        self.btn_auto_motif = widgets.Button(description="Auto-Detect", button_style='info')
        
        self.btn_prev.on_click(self.on_prev)
        self.btn_next.on_click(self.on_next)
        
        # Attach observer. It might fire during init, but is_ready=False shields us.
        self.dropdown.observe(self.on_select, names='value')
        
        self.btn_search_motif.on_click(self.on_search_motif)
        self.btn_auto_motif.on_click(self.on_auto_detect)
        
        # Whisper Transcribe
        self.btn_transcribe = widgets.Button(description="Transcribe (Whisper)", button_style='warning', layout=widgets.Layout(width='150px'))
        self.btn_transcribe.on_click(self.on_transcribe)
        
        # 3. Detail Plot & Audio Output
        # Use a Container to swap output widgets dynamically (Nuclear Option)
        self.plot_container = widgets.VBox([])
        
        # 4. Transcription Output
        self.out_transcribe = widgets.Output()
        
        # Layout
        self.ui = widgets.VBox([
            widgets.HTML(f"<h3>Syllable Explorer: {self.song_name}</h3>"),
            
            # Row 1: Nav
            widgets.HBox([self.btn_prev, self.dropdown, self.btn_next, self.btn_transcribe]),
            
            # Row 2: Segmentation Tuning
            widgets.Label("Segmentation Tuning:"),
            widgets.HBox([self.slider_onset_delta, self.slider_onset_wait, self.btn_resegment, self.btn_play_breaks]),
            
            # Row 3: Motif
            widgets.HBox([self.txt_motif, self.btn_search_motif, self.slider_weight, self.btn_auto_motif]),
            
            self.plot_container,
            self.out_transcribe
        ])
        
        # Guard against double display
        if not getattr(self, 'ui_displayed', False):
             display(self.ui)
             self.ui_displayed = True
            
        # Enable rendering
        self.is_ready = True
        
        # Trigger first selection
        if self.options:
            if self.dropdown.value == 0:
                 # Already 0 (default), so Trigger won't fire. Call manually.
                 self.render_syllable(0)
            else:
                 # Change to 0. Trigger fires.
                 self.dropdown.value = 0

    def plot_overview_plotly(self):
        """Plots the full song pitch contour using Plotly, colored by clusters."""
        fig = go.Figure()

        # Fetch Tonic for Normalization
        tonic_hz = 0
        if 'Tonic' in self.song_df.columns:
            try:
                tonic_hz = float(self.song_df.iloc[0]['Tonic'])
            except: pass
            
        # Fallback if no tonic found (plot raw, but warn?)
        if tonic_hz <= 0:
            tonic_hz = 1.0 # No normalization
            y_label = "Frequency (Hz)"
            print("⚠️ Tonic not found, plotting raw Hz.")
        else:
            y_label = "Frequency Ratio (F/Tonic)"

        # Prepare Data: Normalize Frequencies
        # We'll use a helper to get normalized values easily
        def get_y(freq_series):
            return freq_series / tonic_hz

        # A. Plot Un-clustered Background (Gray)
        fig.add_trace(go.Scatter(
            x=self.song_df['Time'], 
            y=get_y(self.song_df['Frequency']),
            mode='lines',
            line=dict(color='gray', width=1),
            opacity=0.5,
            name='Raw Pitch'
        ))

        # B. Plot Clusters (if available)
        if not self.song_carva.empty:
            label_col = 'Primary_Label' if 'Primary_Label' in self.song_carva.columns else 'Label'
            
            if label_col in self.song_carva.columns:
                unique_labels = self.song_carva[label_col].unique()
                unique_labels = [l for l in unique_labels if l != -1]
                import plotly.colors as pc
                colors = pc.qualitative.Plotly * 10
                
                for i, lbl in enumerate(unique_labels):
                    cluster_segments = self.song_carva[self.song_carva[label_col] == lbl]
                    x_list = []
                    y_list = []
                    
                    for _, row in cluster_segments.iterrows():
                        s_f = int(row['StartFrame'])
                        e_f = int(row['EndFrame'])
                        seg_data = self.song_df.iloc[s_f:e_f]
                        
                        x_list.extend(seg_data['Time'].tolist())
                        x_list.append(None)
                        y_list.extend(get_y(seg_data['Frequency']).tolist())
                        y_list.append(None)
                    
                    fig.add_trace(go.Scatter(
                        x=x_list,
                        y=y_list,
                        mode='lines',
                        line=dict(color=colors[i % len(colors)], width=2),
                        name=f'Cluster {lbl}'
                    ))

        # C. Plot Syllable Lines
        shapes = []
        for o in self.onsets:
            shapes.append(dict(
                type="line",
                x0=o, x1=o,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="cyan", width=1, dash="dash")
            ))
            
        fig.update_layout(
            title=f"Normalized Pitch Contour (Tonic={tonic_hz:.1f}Hz) & Clusters",
            xaxis_title="Time (s)",
            yaxis_title=y_label,
            template="plotly_dark",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            shapes=shapes
        )
        
        # Add Range Slider
        fig.update_xaxes(rangeslider_visible=True)
        
        fig.show()

    def on_prev(self, b):
        idx = self.dropdown.value
        if idx > 0:
            self.dropdown.value = idx - 1

    def on_next(self, b):
        idx = self.dropdown.value
        if (idx is not None) and (idx < len(self.syllables) - 1):
            self.dropdown.value = idx + 1
            
    def on_search_motif(self, b):
        # Just re-render, the render method will pick up the text
        if self.dropdown.value is not None:
            self.render_syllable(self.dropdown.value)

    def on_auto_detect(self, b):
        if self.dropdown.value is None: return
        self.btn_auto_motif.description = "Searching..."
        try:
             # Run detection logic
             self.auto_detect_logic(self.dropdown.value)
        except Exception as e:
             print(f"Auto-detect error: {e}")
        finally:
             self.btn_auto_motif.description = "Auto-Detect"

    def on_resegment(self, b):
        self.btn_resegment.disabled = True
        self.btn_resegment.description = "Working..."
        
        try:
            # 1. Update params
            self.delta = self.slider_onset_delta.value
            self.wait = self.slider_onset_wait.value
            
            # 2. Re-run detection
            # Capture output to avoid flooding cell if possible, or just let it print
            with self.out_transcribe:
                print(f"\n🔄 Re-segmenting with Delta={self.delta}, Wait={self.wait}...")
                self.detect_onsets()
            
            # 3. Update Dropdown
            self.options = [(f"{i+1}. {s:.2f}s - {e:.2f}s", i) for i, (s, e) in enumerate(self.syllables)]
            self.dropdown.options = self.options
            
            # 4. Reset to first
            if self.options:
                self.dropdown.value = 0
                # Use a small delay or direct call? Direct call is fine.
                self.render_syllable(0)
            
            with self.out_transcribe:
                 print("✅ Segmentation Updated.")

        except Exception as e:
            with self.out_transcribe:
                print(f"❌ Resegment error: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self.btn_resegment.disabled = False
            self.btn_resegment.description = "Re-Segment"

    def on_play_breaks(self, b):
        self.btn_play_breaks.disabled = True
        self.btn_play_breaks.description = "Generating..."
        
        try:
             # Logic: Insert silence between syllables
             import numpy as np
             from IPython.display import Audio, display, clear_output
             
             gap_duration = 0.5 # seconds
             gap_samples = int(gap_duration * self.sr)
             silence = np.zeros(gap_samples, dtype=self.y.dtype)
             
             chunks = []
             for s, e in self.syllables:
                 s_idx = int(s * self.sr)
                 e_idx = int(e * self.sr)
                 # Ensure bounds
                 s_idx = max(0, min(len(self.y), s_idx))
                 e_idx = max(0, min(len(self.y), e_idx))
                 
                 chunks.append(self.y[s_idx:e_idx])
                 chunks.append(silence)
             
             if chunks:
                 print("   Concatenating segments...")
                 final_mix = np.concatenate(chunks)
                 
                 # Play in Transcribe Output area to avoid cluttering main plot
                 with self.out_transcribe:
                     clear_output()
                     print(f"▶️ Playing {len(self.syllables)} segments with 0.5s gaps...")
                     display(Audio(final_mix, rate=self.sr, autoplay=True))
             else:
                 print("⚠️ No syllables to play.")
                 
        except Exception as e:
             with self.out_transcribe:
                 print(f"❌ Playback error: {e}")
                 import traceback
                 traceback.print_exc()
        finally:
             self.btn_play_breaks.disabled = False
             self.btn_play_breaks.description = "Play w/ Breaks"

    def on_transcribe(self, b):
        from IPython.display import clear_output
        import os
        
        # Direct output to dedicated widget to ensure visibility
        with self.out_transcribe:
            clear_output()
            print("\n🎙️ Starting Whisper Transcription...")
            
            # 0. Ensure FFmpeg is in PATH (Critical for Whisper)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ffmpeg_dir = os.path.abspath(os.path.join(current_dir, "..", "demucs_env", "Scripts"))
            
            if os.path.exists(os.path.join(ffmpeg_dir, "ffmpeg.exe")):
                if ffmpeg_dir not in os.environ["PATH"]:
                    os.environ["PATH"] += os.pathsep + ffmpeg_dir
                    print(f"✅ Added FFmpeg to PATH: {ffmpeg_dir}")
                else:
                    print(f"✅ FFmpeg found in PATH: {ffmpeg_dir}")
            else:
                print(f"⚠️ FFmpeg not found at expected path: {ffmpeg_dir}")
                print("    Whisper might fail if ffmpeg is not in system PATH.")

            self.btn_transcribe.disabled = True
            self.btn_transcribe.description = "Loading Model..."
            
            try:
                 import whisper
                 # Check for GPU
                 import torch
                 device = "cuda" if torch.cuda.is_available() else "cpu"
                 print(f"   Using device: {device}")
                 
                 model = whisper.load_model("base", device=device)
                 
                 self.btn_transcribe.description = "Transcribing..."
                 print(f"   Processing: {self.audio_path}")
                 
                 # Transcribe with language='en' to force Roman/English characters (Phonetic approximation)
                 # fp16=False suppresses the CPU warning
                 result = model.transcribe(self.audio_path, language="en", fp16=False)
                 
                 print("\n✅ Transcription Complete (English Phonetics):\n" + "="*40)
                 for seg in result['segments']:
                      start = seg['start']
                      end = seg['end']
                      text = seg['text'].strip()
                      print(f"[{start:6.2f}s - {end:6.2f}s]  {text}")
                 print("="*40)
                 
                 # Save to file
                 t_path = self.audio_path.replace('.wav', '_whisper.txt')
                 with open(t_path, 'w', encoding='utf-8') as f:
                      f.write(result['text'])
                 print(f"📄 Saved to: {t_path}")
                 
            except Exception as e:
                 print(f"❌ Transcription Failed: {e}")
                 import traceback
                 traceback.print_exc()
            finally:
                 self.btn_transcribe.disabled = False
                 self.btn_transcribe.description = "Transcribe (Whisper)"

    def auto_detect_logic(self, idx):
        start_t, end_t = self.syllables[idx]
        mask = (self.song_df['Time'] >= start_t) & (self.song_df['Time'] <= end_t)
        syl_df = self.song_df[mask]
        
        if syl_df.empty: return

        # 1. Get raw Y
        y_v = syl_df['Frequency'].values
        t_v = syl_df['Time'].values
        
        tonic_hz = 200.0
        if 'Tonic' in self.song_df.columns:
             try: tonic_hz = float(self.song_df.iloc[0]['Tonic'])
             except: pass
             
        # Use Shared Logic
        w = 0.05
        if hasattr(self, 'slider_weight'):
            w = self.slider_weight.value
            
        print("🔍 Scanning patterns (Fixed Start/End)...")
        best_motif_str, best_spline_y, best_knots, score = find_best_spline_fit(t_v, y_v, tonic_hz, w)
        
        if best_motif_str is not None:
            print(f"✅ Found Best Fit: '{best_motif_str}' (Score: {score:.5f})")
            self.txt_motif.value = best_motif_str
            
            # Update Render? The render function re-calculates or should we store it?
            # Ideally render_syllable should visualize the best fit.
            # But render_syllable currently does its own thing or manual entry.
            # We can re-trigger render which might need to know about the valid spline.
            # For now, just setting the text value is enough, the user can click 'Render' or we call it.
            self.render_syllable(idx)
        else:
            print("⚠️ Could not find a valid spline fit.")

    def on_select(self, change):
        if not getattr(self, 'is_ready', False): return
        idx = change['new']
        if idx is None: return
        self.render_syllable(idx)

    def render_syllable(self, idx):
        start_t, end_t = self.syllables[idx]
        motif_txt = self.txt_motif.value.strip()
        
        # Create FRESH output widget and swap it in
        # This guarantees previous content is destroyed
        out = widgets.Output()
        self.plot_container.children = (out,)
        
        with out:
            plt.ioff()   # turn off interactive auto-render
            # No need for clear_output() since it's a fresh widget
            
            # 1. Play Audio
            start_sample = int(start_t * self.sr)
            end_sample = int(end_t * self.sr)
            segment_audio = self.y[start_sample:end_sample]
            
            print(f"🔊 Playing Syllable {idx+1}: {start_t:.2f}s to {end_t:.2f}s")
            display(ipd.Audio(segment_audio, rate=self.sr, autoplay=True))
            
            # 2. Plot Zoomed Pitch & Waveform
            # Use explicit figure management to avoid duplicate display in Output widget
            fig = plt.figure(figsize=(10, 4))
            
            
            # Waveform
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(segment_audio, sr=self.sr, alpha=0.6)
            plt.title(f"Audio Waveform (Syl {idx+1})")
            
            # ... (Logic remains similar, just target current figure context) ...
            
            # Pitch (from CREPE DF) with Carnatic Context
            plt.subplot(2, 1, 2)
            
            # Filter DF for this time range
            mask = (self.song_df['Time'] >= start_t) & (self.song_df['Time'] <= end_t)
            syl_df = self.song_df[mask]
            
            if not syl_df.empty:
                # Fetch Tonic
                tonic_hz = 0
                if 'Tonic' in self.song_df.columns:
                    try: tonic_hz = float(self.song_df.iloc[0]['Tonic'])
                    except: pass
                
                # Plot
                if tonic_hz > 0:
                     y_vals = syl_df['Frequency'] / tonic_hz
                     plt.plot(syl_df['Time'], y_vals, 'o-', color='#FFCC00', markersize=3, label='F/Tonic')
                     plt.ylabel("Freq Ratio")
                     
                     carnatic_map = {
                         "Sa": 1.0, "Ri1": 1.059, "Ri2": 1.118, "Ga2": 1.189, "Ga3": 1.260,
                         "Ma1": 1.335, "Ma2": 1.414, "Pa": 1.500, "Da1": 1.587, "Da2": 1.682,
                         "Ni2": 1.782, "Ni3": 1.888
                     }
                     
                     # Add Carnatic Bars
                     min_y, max_y = y_vals.min(), y_vals.max()
                     for mult in [0.5, 1.0, 2.0]:
                        for note, ratio in carnatic_map.items():
                            val = ratio * mult
                            if min_y * 0.9 <= val <= max_y * 1.1:
                                label = note
                                if mult == 0.5: label += "."
                                elif mult == 2.0: label += "'"
                                alpha = 0.5 if "Sa" in note or "Pa" in note else 0.3
                                color = "orange" if "Sa" in note or "Pa" in note else "yellow"
                                linestyle = '-' if "Sa" in note or "Pa" in note else ':'
                                plt.axhline(y=val, color=color, linestyle=linestyle, alpha=alpha, linewidth=1)
                                plt.text(end_t, val, label, color=color, fontsize=8, va='center')
                                
                     # Plot Extrema (Peaks & Troughs)
                     y_v = y_vals.values
                     t_v = syl_df['Time'].values
                     
                     # NEW: Plot Start/End & Print
                     if len(y_v) > 0:
                         # Plot Black Squares (Start/End)
                         plt.plot([t_v[0], t_v[-1]], [y_v[0], y_v[-1]], 's', color='black', markersize=5, label='Start/End')
                         
                         # Print Info
                         s_hz = y_v[0] * tonic_hz
                         e_hz = y_v[-1] * tonic_hz
                         print(f"🔹 Syllable Range: Start {y_v[0]:.2f}x ({s_hz:.1f} Hz) -> End {y_v[-1]:.2f}x ({e_hz:.1f} Hz)")
                     if len(y_v) > 2:
                         dy = np.diff(y_v)
                         peaks = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0] + 1
                         troughs = np.where((dy[:-1] < 0) & (dy[1:] > 0))[0] + 1
                         
                         if len(peaks) > 0:
                            plt.plot(t_v[peaks], y_v[peaks], 'o', color='red', markersize=5, label='Peak')
                         if len(troughs) > 0:
                            plt.plot(t_v[troughs], y_v[troughs], 'o', color='lime', markersize=5, label='Trough')

                     # --- MOTIF SEARCH LOGIC (Spline Best Fit) ---
                     if motif_txt:
                         targets = []
                         # Parse Swaras with Octaves
                         # Markers: "Sa." = Lower, "Sa'" = Upper
                         try:
                             # FIX: Replace commas to handle "Sa,Ri" input
                             clean_txt = motif_txt.replace(",", " ")
                             for token in clean_txt.split():
                                 token = token.strip()
                                 if not token: continue
                                 octave = 1.0
                                 
                                 # Check Lower
                                 if "." in token or "," in token:
                                     octave = 0.5
                                     token = token.replace(".", "").replace(",", "")
                                 
                                 # Check Upper
                                 elif "'" in token or '"' in token:
                                     octave = 2.0
                                     token = token.replace("'", "").replace('"', "")
                                 
                                 if token in carnatic_map: targets.append(carnatic_map[token] * octave)
                                 elif token+"1" in carnatic_map: targets.append(carnatic_map[token+"1"] * octave)
                                 elif token+"2" in carnatic_map: targets.append(carnatic_map[token+"2"] * octave)
                         except: pass

                         # Normalize Targets (Intermediate Notes)
                         # Note: User input is NOW treated as intermediates only
                         # Start/End are fixed to actual signal.

                         if len(targets) > 0:
                             # === MANUAL FIT: PEAK SEARCH ===
                             import itertools
                             from scipy.interpolate import CubicSpline
                             
                             # 1. Identify Candidate X-locations (Extrema)
                             indices = []
                             if len(y_v) > 2:
                                 dy = np.diff(y_v)
                                 peaks_idx = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0] + 1
                                 troughs_idx = np.where((dy[:-1] < 0) & (dy[1:] > 0))[0] + 1
                                 indices.extend(peaks_idx)
                                 indices.extend(troughs_idx)
                                 
                             valid_indices = [i for i in sorted(list(set(indices))) if i > 2 and i < len(y_v)-3]
                             candidate_indices = np.array(valid_indices)
                             
                             if len(candidate_indices) >= len(targets):
                                 best_err = float('inf')
                                 best_spline_y = None
                                 best_knots = None
                                 
                                 if len(candidate_indices) < 25:
                                     # Try all sorted combinations of length len(targets)
                                     for mid_indices_combo in itertools.combinations(candidate_indices, len(targets)):
                                         mid_indices = list(mid_indices_combo)
                                         
                                         # Construct Full Knot Set
                                         k_times = t_v[[0] + mid_indices + [len(t_v)-1]]
                                         
                                         start_val = y_v[0]
                                         end_val = y_v[-1]
                                         k_vals = np.array([start_val] + targets + [end_val])
                                         
                                         try:
                                             cs = CubicSpline(k_times, k_vals, bc_type='natural')
                                             spline_y = cs(t_v)
                                             mse = np.mean((y_v - spline_y) ** 2)
                                             
                                             # Score = MSE (Simplify for manual fit)
                                             if mse < best_err:
                                                 best_err = mse
                                                 best_spline_y = (t_v, spline_y)
                                                 best_knots = (k_times, k_vals)
                                         except: pass
                                         
                                 if best_knots:
                                     bt, by = best_spline_y
                                     kt, kv = best_knots
                                     plt.plot(bt, by, '--', color='cyan', linewidth=2, label='Manual Fit')
                                     plt.plot(kt, kv, 'D', color='magenta', markersize=6, label='Knots')
                                     print(f"✨ Manual Fit (MSE: {best_err:.5f})")
                                 else:
                                     print("⚠️ Could not search for fit.")
                             else:
                                 print("⚠️ Not enough extrema candidates for this motif (Need more peaks).")
                         
                         elif "Linear" in motif_txt:
                             # Explicit Linear Plot
                             start_t, end_t = t_v[0], t_v[-1]
                             start_r, end_r = y_v[0], y_v[-1]
                             plt.plot([start_t, end_t], [start_r, end_r], '--', color='cyan', linewidth=2, label='Linear Fit')
                             print("✨ Linear Fit")

                else:
                     plt.plot(syl_df['Time'], syl_df['Frequency'], 'o-', color='#FFCC00', markersize=3, label='Hz')
                     plt.ylabel("Hz")
            else:
                plt.text(0.5, 0.5, "No Pitch Data", ha='center')
                
            plt.xlabel("Time (s)")
            plt.tight_layout()
            # Explicit display to prevent duplicate output in Widgets
            plt.show()
            plt.ion() 

            plt.close(fig)

def explore_syllables(audio_dir: str, song_index: int, db_thresh: float = 10, top_n: int = 9, wait: int = 10, delta: float = 0.2):
    """
    Splits the song into syllables using Onset Detection and creates an interactive explorer widget.
    
    Args:
        wait (int): Window size (in frames) for peak picking. Lower values allow closer onsets. Default=10.
        delta (float): Threshold offset. Lower values increase sensitivity (more splits). Default=0.2.
    """
    SyllableExplorer(audio_dir, song_index, wait, delta)
    return None

# ============================================================================
# ALIASES
# ============================================================================
def primary_clustering(*args, **kwargs):
    """Alias for primary_clustering_pca"""
    return primary_clustering_pca(*args, **kwargs)

def secondary_clustering(*args, **kwargs):
    """Alias for secondary_clustering_pca"""
    return secondary_clustering_pca(*args, **kwargs)

# ============================================================================
# SHARED LOGIC: SPLINE FITTING
# ============================================================================


# ============================================================================
# TONIC ESTIMATION & ANALYSIS
# ============================================================================


def estimate_tonic_for_song(csv_path, song_index, threshold=0.1, plot=True):
    """
    Estimates tonic for a specific song from the Master CREPE CSV.
    Extracts the 'Frequency' column for the given Index and runs stability analysis.
    
    Args:
        csv_path (str): Path to the Master CSV (e.g. Master_Crepe.csv).
        song_index (int): The unique Song Index to filter by.
        threshold (float): Stability threshold in semitones.
        plot (bool): Whether to show the histogram.
        
    Returns:
        float: Estimated Tonic (Hz).
    """
    import pandas as pd
    import numpy as np
    
    print(f"🔍 Loading frequencies for Song Index {song_index} from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Filter by Song Index
        song_df = df[df['Index'] == song_index]
        
        if song_df.empty:
            print(f"❌ No data found for Song Index: {song_index}")
            return 0.0
            
        freqs = song_df['Frequency'].values
        
        # Get Reference Tonic if available
        ref_tonic = 0.0
        if 'Tonic' in song_df.columns:
            try: ref_tonic = float(song_df.iloc[0]['Tonic'])
            except: pass
            
        if len(freqs) < 100:
            print(f"⚠️ Song has very few points ({len(freqs)}). Result may be unstable.")
            
        tonic_est = get_stable_tonic(freqs, threshold=threshold, plot=plot)
        
        print("\n📊 Verification Results:")
        print(f"   🎵 Reference Tonic (CSV): {ref_tonic:.2f} Hz")
        print(f"   ✨ Estimated Tonic (Alg): {tonic_est:.2f} Hz")
        
        if ref_tonic > 0 and tonic_est > 0:
            diff = abs(ref_tonic - tonic_est)
            err_st = 12 * np.log2(tonic_est / ref_tonic)
            print(f"   📉 Difference: {diff:.2f} Hz ({err_st:+.2f} semitones)")
            
        return tonic_est
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0.0


def scan_tonic_for_song(csv_path, song_idx, step_size=0.5, window=0.5, plot=True):
    """
    Wrapper for get_tonic_scan that loads data for a specific song from the Master CSV.
    """
    import pandas as pd
    import numpy as np
    
    print(f"🔍 Loading frequencies for Song Index {song_idx} from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   Shape: {df.shape}, Columns: {list(df.columns)}")
        print(f"   Unique Indices in CSV: {sorted(df['Index'].unique())[:5]} ...")
        
        song_df = df[df['Index'] == song_idx]
        
        if song_df.empty:
            print(f"❌ No data found for Song Index: {song_idx}")
            return 0.0
            
        freqs = song_df['Frequency'].values
        print(f"   Found {len(freqs)} frequency points.")
        
        # Helper: Print Reference if available
        if 'Tonic' in song_df.columns:
            try:
                ref = float(song_df.iloc[0]['Tonic'])
                print(f"   🎵 Reference Tonic (CSV): {ref:.2f} Hz")
            except: pass
            
        return get_tonic_scan(freqs, step_size=step_size, window=window, plot=plot)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0.0



# ============================================================================
# VISUALIZATION & ANALYSIS FUNCTIONS
# ============================================================================

def cluster_curve(
    audio_dir: str, 
    song_index: int, 
    mode: str = 'primary',
    start_time: Optional[float] = None, 
    end_time: Optional[float] = None,
    db_thresh: float = 10,
    top_n: int = 9,
    wait: int = 10,
    delta: float = 0.2
    ):
    """
    Visualizes the F0 contour colored by Cluster Label.
    Modes: 'primary', 'secondary', 'tertiary'.
    
    Args:
        audio_dir (str): Path to audio directory (e.g., Mayamalavagowlai_Vocals)
        song_index (int): Song index to plot
        mode (str): Clustering level - 'primary', 'secondary', or 'tertiary'
        start_time (float, optional): Start time in seconds
        end_time (float, optional): End time in seconds
    """
    from scipy.signal import find_peaks
    
    context = get_raaga_context(audio_dir)
    try:
        master_df = pd.read_csv(context["crepe_csv"])
        carva_df = pd.read_csv(context["carva_csv"])
    except FileNotFoundError:
        print("❌ Data files not found."); return

    song_data = master_df[master_df["Index"] == song_index].reset_index(drop=True)
    if song_data.empty: print("⚠️ No song data."); return
    
    song_carva = carva_df[carva_df["Index"] == song_index]
    # Mode -> Column Mapping
    if mode == 'primary': label_col = 'Primary_Label'
    elif mode == 'secondary': label_col = 'Secondary_Label'
    elif mode == 'tertiary': label_col = 'Tertiary_Label'
    else: label_col = 'Primary_Label'

    if label_col not in song_carva.columns:
        print(f"❌ Column '{label_col}' not found. Run relevant clustering first."); return

    # Frame Slicing
    total_frames = len(song_data)
    time_array = song_data["Time"].values
    s_frame, e_frame = 0, total_frames
    
    if start_time is not None: s_frame = np.searchsorted(time_array, start_time)
    if end_time is not None: e_frame = np.searchsorted(time_array, end_time)
    s_frame = max(0, min(s_frame, total_frames - 1))
    e_frame = max(0, min(e_frame, total_frames))
    
    # Filter Visible
    visible = song_carva[(song_carva['EndFrame'] > s_frame) & (song_carva['StartFrame'] < e_frame)].copy()
    
    # Color Map
    label_to_color = {}
    if not visible.empty:
        # Fill NA labels with -1
        visible[label_col] = visible[label_col].fillna(-1)
        unique_labels = sorted(visible[label_col].astype(str).unique())
        cmap = plt.get_cmap('tab20', len(unique_labels)) if len(unique_labels) <= 20 else plt.get_cmap('gist_ncar', len(unique_labels))
        label_to_color = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}
        
    plt.style.use('dark_background')
    plt.figure(figsize=(18, 6))
    
    # 1. Background Gray
    sub = song_data.iloc[s_frame:e_frame]
    plt.plot(sub["Time"], sub["Frequency"], color='gray', alpha=0.3)
    
    # Colored Segments
    for _, row in visible.iterrows():
        s, e = int(row['StartFrame']), int(row['EndFrame'])
        ps, pe = max(s_frame, s), min(e_frame, e)
        if pe > ps:
            lbl = str(row[label_col])
            c = label_to_color.get(lbl, 'white')
            seg_t = song_data["Time"].iloc[ps:pe]
            seg_f = song_data["Frequency"].iloc[ps:pe]
            plt.plot(seg_t, seg_f, color=c, linewidth=2)
    
    # Title based on mode
    title_suffix = ""
    
    plt.title(f"Clustering: {mode.upper()} | Song {song_index}{title_suffix}")
    plt.ylabel("Hz")
    plt.xlabel("Time (s)")
    plt.show()

def evaluate_cluster(
    audio_dir: str, 
    song_index: int, 
    mode: str = 'primary'
    ):
    """
    Evaluates clustering Quality and provides Interactive Inspection Widget.
    """
    context = get_raaga_context(audio_dir)
    try:
        master_df = pd.read_csv(context["crepe_csv"])
        carva_df = pd.read_csv(context["carva_csv"])
    except: return

    song_carva = carva_df[carva_df["Index"] == song_index]
    if song_carva.empty: return

    if mode == 'primary': label_col = 'Primary_Label'
    elif mode == 'secondary': label_col = 'Secondary_Label'
    elif mode == 'tertiary': label_col = 'Tertiary_Label'
    else: label_col = 'Primary_Label'
    
    if label_col not in song_carva.columns: print(f"Col {label_col} missing"); return
    
    valid = song_carva[song_carva[label_col].astype(str) != '-1']
    
    # Bar Chart
    vc = valid[label_col].value_counts()
    plt.figure(figsize=(10,4))
    vc.plot(kind='bar')
    plt.title(f"{mode} Cluster Counts")
    plt.show()
    
    # Widget
    ids = sorted(valid[label_col].unique())
    dd = widgets.Dropdown(options=ids, description="Cluster:")
    btn = widgets.Button(description="Plot Overlay")
    out = widgets.Output()
    
    def on_clk(b):
        with out:
            clear_output(wait=True)
            cid = dd.value
            subset = valid[valid[label_col] == cid]
            plt.figure(figsize=(10,5)); plt.style.use('dark_background')
            
            interp_len = 64
            Ys = []
            for _, r in subset.head(50).iterrows():
                try:
                    seg = np.array(json.loads(r["SegmentList"]))
                    if len(seg)<2: continue
                    # Always interpolate for overlay consistency
                    x_new = np.linspace(0, 1, interp_len)
                    x_old = np.linspace(0, 1, len(seg))
                    y_i = np.interp(x_new, x_old, seg)
                    Ys.append(y_i)
                    plt.plot(x_new, y_i, color='cyan', alpha=0.3)
                except: pass
            
            if Ys:
                avg = np.mean(Ys, axis=0)
                plt.plot(np.linspace(0,1,interp_len), avg, 'w--', linewidth=2)
                
            plt.title(f"Cluster {cid} Overlay")
    btn.on_click(on_clk)
    display(widgets.VBox([dd, btn, out]))


# ============================================================================
# HELPER: AUDIO SYNTHESIS
# ============================================================================
def synthesize_tone(ratios, sr=44100, base_freq=261.63):
    """
    Synthesize a rich tone from pitch ratios.
    Args:
        ratios (list/array): Pitch ratios (F/Base).
        sr (int): Sampling rate.
        base_freq (float): Base frequency (Tonic) in Hz. Defaults to C4 (261.63).
    """
    import numpy as np
    
    freqs = np.array(ratios) * base_freq
    
    # Integration for phase (handling varying frequency)
    phases = np.cumsum(2 * np.pi * freqs / sr)
    
    # Fundamental + Harmonics (Rich Tone)
    y = 0.6 * np.sin(phases) + 0.3 * np.sin(2 * phases) + 0.1 * np.sin(3 * phases)
    
    # Envelope (Fade In/Out)
    env = np.ones_like(y)
    fade_len = int(0.01 * sr) # 10ms fade
    
    if len(env) > 2 * fade_len:
        env[:fade_len] = np.linspace(0, 1, fade_len)
        env[-fade_len:] = np.linspace(1, 0, fade_len)
    else:
        env = np.hanning(len(env))
        
    return y * env

# Removed secondary_cluster_widget (Deprecated)

# ============================================================================
# SHARED LOGIC: DETECT SWARAS (Refactored)
# ============================================================================
def detect_swaras_from_values(y_values, target_points=16, raga=None):
    if len(y_values) == 0: return "", []
    
    # Prepare Valid Candidates
    valid_candidates = list(CARNATIC_RATIOS.items())
    if raga and raga != 'None':
        allowed = get_allowed_swaras(raga)
        if allowed:
            # print(f"DEBUG: Raga={raga}, Applying Mask: {sorted(list(allowed))}")
            filtered = [x for x in valid_candidates if x[0] in allowed]
            if filtered: valid_candidates = filtered
        else:
            # print(f"DEBUG: Raga={raga}, No allowed notes found.")
            pass

    detected_events = [] 
    
    # Helper: Snap Y-value to nearest Carnatic ratio
    def snap_to_carnatic(y_val):
        return min(valid_candidates, key=lambda x: abs(x[1] - y_val))[1]

    # 1. INITIAL DETECTION: Capture EVERYTHING (Max Sensitivity)
    peaks, _ = find_peaks(y_values, prominence=0.01)
    valleys, _ = find_peaks(-y_values, prominence=0.01)
    
    for p in peaks:
        y_snapped = snap_to_carnatic(y_values[p])
        detected_events.append((float(p), y_snapped, 'peak'))
    for v in valleys:
        y_snapped = snap_to_carnatic(y_values[v])
        detected_events.append((float(v), y_snapped, 'valley'))
        
    # Add basic plateaus
    dy = np.diff(y_values)
    is_flat = np.abs(dy) < 0.05 
    run_start = -1
    min_plateau_len = 3 
    
    for i in range(len(is_flat)):
        if is_flat[i]:
            if run_start == -1: run_start = i
        else:
            if run_start != -1:
                run_end = i 
                if run_end - run_start >= min_plateau_len:
                    p_val = np.median(y_values[run_start : run_end+1])
                    p_val_snapped = snap_to_carnatic(p_val)
                    detected_events.append((float(run_start), p_val_snapped, 'plateau_start'))
                    detected_events.append((float(run_end), p_val_snapped, 'plateau_end'))
                run_start = -1
    
    if run_start != -1:
            p_val = np.median(y_values[run_start : ])
            p_val_snapped = snap_to_carnatic(p_val)
            detected_events.append((float(run_start), p_val_snapped, 'plateau_start'))
            detected_events.append((float(len(is_flat)), p_val_snapped, 'plateau_end'))

    # Ensure Boundaries (Float) - USE RAW VALUES (NOT SNAPPED!)
    if not any(e[0] == 0.0 for e in detected_events):
        detected_events.append((0.0, float(y_values[0]), 'boundary'))
    last_idx = float(len(y_values)-1)
    if not any(e[0] == last_idx for e in detected_events):
        detected_events.append((last_idx, float(y_values[-1]), 'boundary'))

    # Sort and Deduplicate (Float tolerance?)
    detected_events.sort(key=lambda x: x[0])
    unique_events = []
    last_x = -1.0
    for e in detected_events:
        if abs(e[0] - last_x) > 0.01: # 0.01 tolerance
            unique_events.append(e)
            last_x = e[0]
    detected_events = unique_events

    # Interpretation: "Target Points" = Intermediate Points.
    # So Total Target = Target Points + 2 (Start + End)
    total_target = target_points + 2

    # 2. SIMPLIFY: Remove "Useless" points until count == total_target
    while len(detected_events) > total_target:
        min_area = np.inf
        remove_idx = -1
        
        # Check internal points only (preserve start/end)
        for i in range(1, len(detected_events) - 1):
            p_prev = detected_events[i-1]
            p_curr = detected_events[i]
            p_next = detected_events[i+1]
            x1, y1 = p_prev[0], p_prev[1]
            x2, y2 = p_curr[0], p_curr[1]
            x3, y3 = p_next[0], p_next[1]
            area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            if area < min_area:
                min_area = area
                remove_idx = i
        
        if remove_idx != -1:
            detected_events.pop(remove_idx)
        else:
            break 

    # 3. FILL: Add points if too few (Decimal Support)
    while len(detected_events) < total_target:
        detected_events.sort(key=lambda x: x[0])
        max_gap = 0
        best_mid = -1
        
        for i in range(len(detected_events) - 1):
            gap = detected_events[i+1][0] - detected_events[i][0]
            if gap > max_gap:
                max_gap = gap
                best_mid = (detected_events[i][0] + detected_events[i+1][0]) / 2.0
        
        if max_gap < 0.001: break # Prevent infinite splitting of tiny gaps
        
        # Interpolate Y
        val = np.interp(best_mid, np.arange(len(y_values)), y_values)
        detected_events.append((best_mid, val, 'forced'))

    detected_events.sort(key=lambda x: x[0])

    detected_names = []
    for idx_t, val, ev_type in detected_events:
        # Skip extreme edges to avoid artifacts
        if idx_t < 2 or idx_t > len(y_values)-3: 
            continue
        
        best_swara = min(valid_candidates, key=lambda x: abs(x[1] - val))
        detected_names.append(best_swara[0])
        
    # Ultimate Fallback
    if not detected_names:
        median_val = np.median(y_values)
        best_swara = min(valid_candidates, key=lambda x: abs(x[1] - median_val))
        return best_swara[0], detected_events 
        
    return " ".join(detected_names), detected_events


def extract_melodic_sequence(y_raw: np.ndarray, target_points: int = 8, target_frames: int = 60) -> List[List[float]]:
    """
    Standard pipeline for Melodic Parameterization via Spline Fitting (Manuscript Sec 3.3).
    
    Args:
        y_raw: Raw pitch contour array.
        target_points: Number of anchor points (knots) to extract. Default 8 matches RagaLSTM.
        target_frames: Resolution for landmark detection. Default 60.
        
    Returns:
        List of [note_id, delta_t] vectors.
    """
    if len(y_raw) < 5:
        return []

    # 1. Linear interpolation to fixed time-grid for landmark detection
    x_old = np.linspace(0, 1, len(y_raw))
    x_new = np.linspace(0, 1, target_frames)
    y_interp = np.interp(x_new, x_old, y_raw)
    
    # 2. Detect and Prune Landmarks (Peaks, Valleys, Plateaus + Visvalingam-Whyatt)
    # detect_swaras_from_values returns target_points + 2 (boundaries), 
    # so we adjust the target to match the requested output length.
    _, events = detect_swaras_from_values(y_interp, target_points=max(2, target_points - 2))
    
    # 3. Encode into Sequence
    sequence = []
    prev_t = events[0][0] / target_frames
    
    for t_frame, y_val, _ in events:
        if y_val <= 0: continue
        t = t_frame / target_frames
        delta_t = t - prev_t
        prev_t = t
        
        # Note ID logic (Distance-based mapping to 12 semitones)
        semitone = 12 * np.log2(y_val)
        note_id = int(round(semitone)) % 12
        
        sequence.append([float(note_id), float(delta_t)])
        
    return sequence
