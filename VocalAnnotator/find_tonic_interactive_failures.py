import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
import glob
import scipy.io.wavfile
from scipy.signal import find_peaks
import tempfile
import re
import librosa

# Try importing pygame
try:
    import pygame
    HAS_PYGAME = True
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
except ImportError as e:
    HAS_PYGAME = False
    print(f"WARNING: 'pygame' import failed: {e}")
    print("Advanced audio features disabled.")
except Exception as e:
    HAS_PYGAME = False
    print(f"WARNING: 'pygame' initialization failed: {e}")
    print("Advanced audio features disabled.")

# ==========================
# CONFIGURATION
# ==========================
BASE_PROJECT_DIR = r"C:\Desktop\Python\CarnaticAnnotater"

# ---- ONLY CHANGE FROM find_tonic_interactive.py ----
# Hardcoded list of REAL failure case CSVs identified by the scanner.
# Audio files are searched dynamically within BASE_PROJECT_DIR.
FAILURE_PATHS = [
    # Keeravani (1)
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Keeravani\Keeravani_CSVs\Keeravani Maya_Vithai_01_vocals_147_03_D3_pitch.csv",
    # Kharaharapriya (6)
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Kharaharapriya\Kharaharapriya_CSVs\Kharaharapriya_Nadachi01_vocals_157_81_D#3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Kharaharapriya\Kharaharapriya_CSVs\Kharaharapriya_Nidhamoru01_vocals_151_72_D#3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Kharaharapriya\Kharaharapriya_CSVs\Kharaharapriya_Prakallanilabadi01_vocals_252_50_B3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Kharaharapriya\Kharaharapriya_CSVs\Kharaharapriya_Prakallanilabadi02_vocals_251_56_B3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Kharaharapriya\Kharaharapriya_CSVs\Kharaharapriya_Ramaneesamana01_vocals_133_91_C3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Kharaharapriya\Kharaharapriya_CSVs\Kharaharapriya_Ramaneesamana03_vocals_133_91_C3_pitch.csv",
    # Mayamalavagowlai (6)
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Mayamalavagowlai\Mayamalavagowlai_CSVs\Mayamalavagowla_DevadevaNeraval3_vocals_131_78_C3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Mayamalavagowlai\Mayamalavagowlai_CSVs\Mayamalavagowla_DevadevaNeraval4_vocals_117_77_A#2_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Mayamalavagowlai\Mayamalavagowlai_CSVs\Mayamalavagowla_DevadevaNeraval5_vocals_127_60_C3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Mayamalavagowlai\Mayamalavagowlai_CSVs\Mayamalavagowla_DevadevaSwarakalpana3_vocals_116_54_A#2_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Mayamalavagowlai\Mayamalavagowlai_CSVs\Mayamalavagowla_DevadevaSwarakalpana4_vocals_124_52_B2_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Mayamalavagowlai\Mayamalavagowlai_CSVs\Mayamalavagowla_MerusamanaSwarakalpan_vocals_103_00_G#2_pitch.csv",
    # Shankarabharanam (10)
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Shankarabharanam\Shankarabharanam_CSVs\Shankarabharanam Sita_Kalyana_Vibhogame_01_vocals_184_06_F#3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Shankarabharanam\Shankarabharanam_CSVs\Shankarabharanam_BhakthaParayana02_vocals_203_28_G#3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Shankarabharanam\Shankarabharanam_CSVs\Shankarabharanam_Dakshinamurthe02_vocals_118_91_A#2_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Shankarabharanam\Shankarabharanam_CSVs\Shankarabharanam_Dakshinamurthe03_vocals_133_44_C3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Shankarabharanam\Shankarabharanam_CSVs\Shankarabharanam_EndukuPeddala02_vocals_129_69_C3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Shankarabharanam\Shankarabharanam_CSVs\Shankarabharanam_Nannubrovakanu01_vocals_129_69_C3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Shankarabharanam\Shankarabharanam_CSVs\Shankarabharanam_Pallavi02_vocals_130_16_C3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Shankarabharanam\Shankarabharanam_CSVs\Shankarabharanam_Pallavi03_vocals_224_38_A3_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Shankarabharanam\Shankarabharanam_CSVs\Shankarabharanam_Pallavi_vocals_119_84_A#2_pitch.csv",
    r"C:\Desktop\Python\CarnaticAnnotater\Raagas\Shankarabharanam\Shankarabharanam_CSVs\Shankarabharanam_Swaragasudha01_vocals_136_72_C#3_pitch.csv",
]
# -----------------------------------------------------

# --- SIGNAL PROCESSING HYPERPARAMETERS ---
CONF_THRESHOLD = 0.1          # Min confidence
AMP_THRESHOLD = 0.1           # Min normalized amplitude (0.0 to 1.0)
DEBRIS_THRESHOLD = 10         # Min contiguous frames
FREQ_MEDIAN_FACTOR = 3.0      # Max freq = Median * Factor
AMP_WINDOW = 0.33             # Seconds for rolling amplitude window
AMP_HOP = 0.125               # Seconds (used for window calc)
ONLY_EXTREMA = True           # Use both Peaks AND Troughs (Local Extrema)
ONLY_PEAKS = False            # Deprecated flag, keeping for ref if needed (replaced by ONLY_EXTREMA logic)

# --- SEARCH RANGE ---
MIN_TONIC = 50.0             # Lower bound for tonic search
MAX_TONIC = 500.0            # Upper bound for tonic search

# RAAGA DEFINITION
# The user would manually change ACTIVE_NOTES per raga folder.
# Here we auto-detect from the file path.
NOTE_NAMES = ['S', 'r1', 'r2', 'g2', 'g3', 'm1', 'm2', 'P', 'd1', 'd2', 'n2', 'n3']
SA_PA_WEIGHT = 3.0

RAGA_MASKS = {
    "Kalyani":          np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]),
    "Mayamalavagowlai": np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1]),
    "Shankarabharanam": np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]),
    "Kharaharapriya":   np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]),
    "Thodi":            np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]),
    "Keeravani":        np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1]),
    "Shanmukhapriya":   np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0]),
    "Panthuvarali":     np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1]),
}

def get_active_and_weighted(path):
    for raga_name, mask in RAGA_MASKS.items():
        if raga_name.lower() in path.lower():
            w = mask.astype(float).copy()
            w[0] = SA_PA_WEIGHT; w[7] = SA_PA_WEIGHT
            return mask, w
    # Fallback: all chromatic
    mask = np.ones(12, dtype=int)
    w = np.ones(12); w[0] = SA_PA_WEIGHT; w[7] = SA_PA_WEIGHT
    return mask, w

# Global defaults (overwritten per file in load_current_file)
ACTIVE_NOTES = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
WEIGHTED_NOTES = ACTIVE_NOTES.astype(float).copy()
WEIGHTED_NOTES[0] = SA_PA_WEIGHT
WEIGHTED_NOTES[7] = SA_PA_WEIGHT

RATIOS = {
    "Sa": 1.0,
    "Ri1": 16/15, "Ri2": 9/8,
    "Ga2": 6/5, "Ga3": 5/4,
    "Ma1": 4/3, "Ma2": 45/32,
    "Pa": 1.5,
    "Da1": 8/5, "Da2": 5/3,
    "Ni2": 9/5, "Ni3": 15/8
}
SCALE_CENTS = np.array([1200 * np.log2(list(RATIOS.values())[i]) for i in range(len(RATIOS))])
RATIO_VALUES = np.array(list(RATIOS.values()))


def hz_to_note_cents(freq):
    if freq <= 0: return "N/A", "--"
    A4 = 440.0
    midi = 69 + 12 * np.log2(freq / A4)
    midi_round = int(round(midi))
    cents_diff = int(round((midi - midi_round) * 100))
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = notes[midi_round % 12]
    octave = (midi_round // 12) - 1
    sign = "+" if cents_diff >= 0 else ""
    return f"{note_name}{octave} {sign}{cents_diff}c", f"{note_name}{octave}"

def note_to_hz(note_str):
    note_str = note_str.strip().upper()
    if not note_str: return None
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_str = note_str.replace('DB', 'C#').replace('EB', 'D#').replace('GB', 'F#').replace('AB', 'G#').replace('BB', 'A#')

    alpha = ""
    numeric = ""
    for char in note_str:
        if char.isdigit() or char == '-': numeric += char
        else: alpha += char

    if not alpha in notes: return None
    if not numeric: octave = 4
    else: octave = int(numeric)

    semitone_idx = notes.index(alpha)
    midi = 12 * (octave + 1) + semitone_idx
    freq = 440.0 * (2 ** ((midi - 69) / 12.0))
    return freq

def compute_histogram(pitch_data, tonic_freq):
    if len(pitch_data) == 0: return np.zeros(12), 1
    rel_cents = 1200 * np.log2(pitch_data / tonic_freq)
    folded_cents = rel_cents % 1200
    counts = np.zeros(12)
    window = 30
    scale_cents_arr = 1200 * np.log2(RATIO_VALUES)

    for i, target_cent in enumerate(scale_cents_arr):
        diff = np.abs(folded_cents - target_cent)
        diff = np.minimum(diff, 1200 - diff)
        counts[i] = np.sum(diff < window)
    return counts, len(pitch_data)

def compute_range_weight(pitch_data, tonic_freq):
    if len(pitch_data) == 0: return 0
    lower = tonic_freq * 0.7
    upper = tonic_freq * 3.0
    in_range = np.sum((pitch_data >= lower) & (pitch_data <= upper))
    return in_range / len(pitch_data)

def analyze_best_tonic(pitch_data):
    if len(pitch_data) == 0: return 0, 0
    tonics = np.linspace(MIN_TONIC, MAX_TONIC, 1000)
    best_score = -1
    best_tonic = 0
    for t in tonics:
        counts, total = compute_histogram(pitch_data, t)
        chroma = np.sum(counts * WEIGHTED_NOTES) / total if total > 0 else 0
        range_w = compute_range_weight(pitch_data, t)
        score = chroma * range_w
        if score > best_score:
            best_score = score
            best_tonic = t
    return best_tonic, best_score

def get_filtered_pitch(csv_path, wav_path):
    """
    Applies robust filtering:
    1. Confidence Threshold
    2. Amplitude (RMS) Threshold
    3. Debris Removal
    4. Median Frequency Filter
    5. Peak & Trough Picking (Extrema)
    """
    try:
        df = pd.read_csv(csv_path)

        # Normalize Column Names
        df.columns = [c.lower() for c in df.columns]
        col_map = {'frequency': 'Frequency', 'time': 'Time', 'confidence': 'Confidence'}
        df = df.rename(columns=col_map)

        if len(df) == 0: return np.array([100]), None, None

        # 1. Confidence Filter
        if 'Confidence' in df.columns:
            df['ConfSmooth'] = df['Confidence'].rolling(window=5, center=True).median().fillna(0)
            mask_conf = df['ConfSmooth'] >= CONF_THRESHOLD
        else:
            mask_conf = pd.Series(True, index=df.index)

        # 2. Amplitude Filter (Needs Audio)
        mask_amp = pd.Series(True, index=df.index)
        if wav_path and os.path.exists(wav_path):
            try:
                y, sr = librosa.load(wav_path, sr=None)
                rmse = librosa.feature.rms(y=y)[0]
                times = librosa.times_like(rmse, sr=sr)
                peak_amp = np.max(rmse) if len(rmse) > 0 else 1.0

                if 'Time' in df.columns:
                    crepe_times = df['Time'].values
                    interp_rmse = np.interp(crepe_times, times, rmse)
                    norm_rmse = interp_rmse / peak_amp if peak_amp > 0 else interp_rmse

                    step_size = df['Time'].diff().median()
                    if pd.isna(step_size) or step_size <= 0: step_size = 0.02
                    win_samples = int(AMP_WINDOW / step_size)
                    if win_samples < 1: win_samples = 1

                    rolling_amp = pd.Series(norm_rmse).rolling(window=win_samples, center=True).mean().fillna(0)
                    mask_amp = rolling_amp >= AMP_THRESHOLD

            except Exception as e:
                print(f"Warning: Audio load failed for filtering: {e}")

        # 3. Frequency Median Filter
        mask_freq = pd.Series(True, index=df.index)
        valid_freqs = df[mask_conf]['Frequency']
        if len(valid_freqs) > 0:
            median_freq = valid_freqs.median()
            max_thresh = median_freq * FREQ_MEDIAN_FACTOR
            mask_freq = df['Frequency'] <= max_thresh

        # Combine Masks
        mask = mask_conf & mask_amp & mask_freq

        # 4. Debris Filter
        if DEBRIS_THRESHOLD > 0:
            blocks = (mask != mask.shift()).cumsum()
            sizes = mask.groupby(blocks).transform('count')
            mask = mask & ~((mask) & (sizes < DEBRIS_THRESHOLD))

        # Extract Valid Frequencies
        valid_df = df[mask].copy()

        if len(valid_df) == 0:
            print("Warning: Filtering removed all data points.")
            return np.array([100]), None, None

        # 5. Extrema Picking (Peaks AND Troughs)
        if ONLY_EXTREMA:
            freq_curve = valid_df['Frequency'].values

            # Smoothing
            freq_curve_smooth = np.convolve(freq_curve, [0.2, 0.6, 0.2], mode='same')

            peaks, _ = find_peaks(freq_curve_smooth)
            troughs, _ = find_peaks(-freq_curve_smooth)

            extrema_indices = np.sort(np.concatenate((peaks, troughs)))
            valid_extrema = freq_curve[extrema_indices]

            print(f"Filtering: {len(df)} -> {len(valid_df)} (Valid) -> {len(valid_extrema)} (Extrema)")

            if len(valid_extrema) > 5:
                return valid_extrema, df, mask
            else:
                print("Not enough extrema found, falling back to all valid points.")
                return freq_curve, df, mask
        else:
            print(f"Filtering: {len(df)} -> {len(valid_df)} (Valid)")
            return valid_df['Frequency'].values, df, mask

    except Exception as e:
        print(f"Error in filtering: {e}")
        return np.array([100]), None, None

def plot_carnatic_pitch_standalone(ax, times, freqs, mask, tonic, title="Carnatic Pitch Plot"):
    ax.set_facecolor('black')
    ax.set_title(title, color='black', fontsize=12)

    if tonic and tonic > 0:
        valid_freqs = freqs[mask] if mask is not None else freqs
        valid_freqs = valid_freqs[~np.isnan(valid_freqs)]

        if len(valid_freqs) > 0:
            min_f, max_f = np.min(valid_freqs), np.max(valid_freqs)
        else:
            min_f, max_f = tonic, tonic * 2

        multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]

        for mult in multipliers:
            base = tonic * mult
            if base * 2 < min_f and base * RATIOS["Ni3"] < min_f: continue
            if base > max_f: continue

            for swara_name, ratio in RATIOS.items():
                y = base * ratio
                color = "#FFD700"

                if swara_name in ["Sa", "Pa"]:
                    lw = 1.5; alpha = 0.9; linestyle = '--'
                else:
                    lw = 0.8; alpha = 0.5; linestyle = ':'

                ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=lw, alpha=alpha)

    ax.plot(times, freqs, color='red', linewidth=1.0, alpha=0.5, label='Filtered')

    valid_freqs_plot = freqs.copy()
    if mask is not None:
        valid_freqs_plot[~mask] = np.nan
    ax.plot(times, valid_freqs_plot, color='#00FA9A', linewidth=1.5, label='Valid')

    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    if tonic:
        ax.axhline(y=tonic, color='darkblue', linewidth=2.5, linestyle='-', alpha=1.0, label='Tonic (Sa)')


class InteractiveTonicApp:
    def __init__(self, file_paths):
        self.file_list = file_paths
        self.current_idx = 0
        self.current_wav_path = None
        self.audio_duration = 0
        self.full_df = None
        self.full_mask = None

        self.fig = plt.figure(figsize=(13, 13))
        gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 3, 2.5])
        self.ax_score = self.fig.add_subplot(gs[0])
        self.ax_bars = self.fig.add_subplot(gs[1])

        plt.subplots_adjust(bottom=0.40, hspace=0.4)

        self.setup_widgets()
        self.load_current_file()

    def generate_rich_tone(self, freq, duration=1.0):
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        wave = 0.6 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t) + 0.15 * np.sin(2 * np.pi * freq * 3 * t)
        env = np.ones_like(wave)
        attack = int(0.1 * sr)
        env[:attack] = np.linspace(0, 1, attack)
        env[-attack:] = np.linspace(1, 0, attack)
        wave = wave * env / np.max(np.abs(wave))
        wave = (wave * 32767).astype(np.int16)
        return sr, wave

    def load_current_file(self):
        if not self.file_list: return
        filepath = self.file_list[self.current_idx]
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            return

        filename = os.path.basename(filepath)
        print(f"Loading FAILURE {self.current_idx+1}/{len(self.file_list)}: {filename}")

        if HAS_PYGAME: pygame.mixer.music.stop()

        # Auto-detect raga and set correct mask for this file
        global ACTIVE_NOTES, WEIGHTED_NOTES
        ACTIVE_NOTES, WEIGHTED_NOTES = get_active_and_weighted(filepath)
        colors = ['green' if a else 'gray' for a in ACTIVE_NOTES]
        print(f"Raga mask applied for: {filepath.split(chr(92))[-3] if chr(92) in filepath else 'Unknown'}")

        # ---- CHANGED: Search whole project for the WAV ----
        self.current_wav_path = None
        wav_name = filename.replace('_pitch.csv', '.wav')
        for root, dirs, files in os.walk(BASE_PROJECT_DIR):
            if wav_name in files:
                self.current_wav_path = os.path.join(root, wav_name)
                break
        # ---------------------------------------------------

        self.audio_duration = 0
        if self.current_wav_path and os.path.exists(self.current_wav_path):
            print(f"Found audio: {os.path.basename(self.current_wav_path)}")
            if HAS_PYGAME:
                try:
                    pygame.mixer.music.load(self.current_wav_path)
                    sound = pygame.mixer.Sound(self.current_wav_path)
                    self.audio_duration = sound.get_length()
                except:
                    self.audio_duration = 0
        else:
            print(f"No audio file found for {wav_name}")

        self.time_slider.eventson = False
        self.time_slider.set_val(0)
        self.time_slider.valmax = max(1, self.audio_duration)
        self.time_slider.ax.set_xlim(0, self.time_slider.valmax)
        self.time_slider.eventson = True

        print(f"Filtering pitch data...")
        self.pitch_data, self.full_df, self.full_mask = get_filtered_pitch(filepath, self.current_wav_path)

        print(f"Computing best tonic...")
        self.best_auto_tonic, _ = analyze_best_tonic(self.pitch_data)
        self.current_tonic = self.best_auto_tonic

        print(f"Computing profile...")
        self.compute_profile()

        print(f"Refreshing plots...")
        self.refresh_plots()

        # ---- CHANGED: Show "FAILURE CASE" in title ----
        self.fig.suptitle(f"FAILURE CASE: {filename} ({self.current_idx+1}/{len(self.file_list)})", fontsize=13, color='red')
        # ------------------------------------------------
        print(f"Ready!")

    def compute_profile(self):
        tonics = np.linspace(MIN_TONIC, MAX_TONIC, 500)
        scores = []
        for t in tonics:
            c, total = compute_histogram(self.pitch_data, t)
            chroma = np.sum(c * WEIGHTED_NOTES) / total if total > 0 else 0
            range_w = compute_range_weight(self.pitch_data, t)
            scores.append(chroma * range_w)
        self.profile_x = tonics
        self.profile_y = np.array(scores)

    def update_metrics(self, tonic):
        counts, total = compute_histogram(self.pitch_data, tonic)
        active_counts = np.sum(counts * WEIGHTED_NOTES)
        chroma = active_counts / total if total > 0 else 0
        range_w = compute_range_weight(self.pitch_data, tonic)
        return counts / (np.max(counts) if np.max(counts) > 0 else 1), chroma * range_w

    def refresh_plots(self):
        self.ax_score.clear()
        self.ax_bars.clear()

        self.ax_score.plot(self.profile_x, self.profile_y, color='blue')
        self.score_marker, = self.ax_score.plot([self.current_tonic], [0], 'r*', markersize=12)
        self.ax_score.set_title("Match Score vs Frequency")
        self.ax_score.grid(True)

        norm_counts, score = self.update_metrics(self.current_tonic)
        colors = ['green' if a else 'gray' for a in ACTIVE_NOTES]
        self.bars = self.ax_bars.bar(NOTE_NAMES, norm_counts, color=colors, alpha=0.7)
        self.ax_bars.set_ylim(0, 1.1)

        self.update_controls(self.current_tonic, score)
        self.tonic_slider.set_val(self.current_tonic)
        self.fig.canvas.draw_idle()

    def update_interaction(self, val):
        tonic = self.tonic_slider.val
        self.current_tonic = tonic
        norm_counts, score = self.update_metrics(tonic)
        for bar, h in zip(self.bars, norm_counts): bar.set_height(h)
        self.score_marker.set_data([tonic], [score])
        self.update_controls(tonic, score)
        self.fig.canvas.draw_idle()

    def update_controls(self, tonic, score):
        detailed_note, simple_note = hz_to_note_cents(tonic)
        self.txt_note.set_text(f"Closest: {detailed_note}")
        self.txt_score.set_text(f"Score: {score:.4f}")
        self.ax_bars.set_title(f"Tonic: {tonic:.2f} Hz")

        if self.box_freq.text != f"{tonic:.2f}":
            self.box_freq.set_val(f"{tonic:.2f}")

        if self.box_western.text != simple_note:
            self.box_western.set_val(simple_note)

    def submit_freq(self, text):
        try:
            val = float(text)
            self.current_tonic = val
            self.refresh_plots()
        except: pass

    def submit_western(self, text):
        freq = note_to_hz(text)
        if freq:
            self.current_tonic = freq
            self.refresh_plots()

    def save_annotation(self, event):
        print(f"SAVE clicked. WAV path: {self.current_wav_path}")

        if not self.current_wav_path:
            print("No WAV to rename.")
            return

        if not os.path.exists(self.current_wav_path):
            print(f"WAV file doesn't exist: {self.current_wav_path}")
            return

        freq_str = f"{self.current_tonic:.2f}".replace('.', '_')
        note_str = self.box_western.text.strip()
        dir_name = os.path.dirname(self.current_wav_path)
        filename = os.path.basename(self.current_wav_path)
        match = re.match(r"^(.*_vocals).*(\\.wav)$", filename, re.IGNORECASE)
        if not match: base_part = os.path.splitext(filename)[0]
        else: base_part = match.group(1)

        new_filename = f"{base_part}_{freq_str}_{note_str}.wav"
        new_path = os.path.join(dir_name, new_filename)

        print(f"Renaming: {filename} -> {new_filename}")

        try:
            if HAS_PYGAME:
                print("Stopping pygame audio...")
                pygame.mixer.music.stop()
                pygame.mixer.stop()
                try:
                    pygame.mixer.music.unload()
                except:
                    pass
                import time
                time.sleep(0.3)

            os.rename(self.current_wav_path, new_path)
            print(f"WAV Saved: {new_filename}")

            csv_path = self.file_list[self.current_idx]
            csv_dir = os.path.dirname(csv_path)
            new_csv_name = f"{base_part}_{freq_str}_{note_str}_pitch.csv"
            new_csv_path = os.path.join(csv_dir, new_csv_name)

            if os.path.exists(csv_path):
                os.rename(csv_path, new_csv_path)
                print(f"CSV Renamed: {new_csv_name}")
                self.file_list[self.current_idx] = new_csv_path

            self.current_wav_path = new_path

        except PermissionError as e:
            print(f"PERMISSION ERROR: File is locked or in use! {e}")
        except FileExistsError:
            print(f"ERROR: Target file already exists: {new_filename}")
        except Exception as e:
            print(f"SAVE ERROR: {e}")
            import traceback
            traceback.print_exc()

    def plot_full_song_view(self, event):
        if self.full_df is None:
            print("No data loaded to plot.")
            return

        fig_new, ax_new = plt.subplots(figsize=(14, 8))
        plot_carnatic_pitch_standalone(
            ax_new,
            self.full_df['Time'],
            self.full_df['Frequency'],
            self.full_mask,
            self.current_tonic,
            title=f"Full Pitch Analysis: {os.path.basename(self.file_list[self.current_idx])}"
        )
        plt.show()

    def play_song_file(self, event):
        if not HAS_PYGAME or not self.current_wav_path: return
        try: pygame.mixer.music.play(start=self.time_slider.val)
        except: pass

    def pause_song(self, event):
        if HAS_PYGAME: pygame.mixer.music.pause()

    def unpause_song(self, event):
        if HAS_PYGAME: pygame.mixer.music.unpause()

    def play_current_tonic(self, event):
        self.play_tone_hz(self.current_tonic)

    def play_tone_hz(self, freq):
        if not HAS_PYGAME: return
        sr, wave = self.generate_rich_tone(freq)
        sound = pygame.mixer.Sound(buffer=wave.tobytes())
        sound.play()

    def next_file(self, e):
        if self.current_idx < len(self.file_list)-1: self.current_idx += 1; self.load_current_file()

    def prev_file(self, e):
        if self.current_idx > 0: self.current_idx -= 1; self.load_current_file()

    def auto_snap(self, e):
        self.current_tonic = self.best_auto_tonic
        self.refresh_plots()

    def stop_all(self, e):
        if HAS_PYGAME: pygame.mixer.stop(); pygame.mixer.music.stop()

    def setup_widgets(self):
        ax_tonic = plt.axes([0.15, 0.30, 0.50, 0.03])
        self.tonic_slider = Slider(ax_tonic, 'Tonic (Hz)', MIN_TONIC, MAX_TONIC)
        self.tonic_slider.on_changed(self.update_interaction)

        ax_time = plt.axes([0.15, 0.25, 0.50, 0.03])
        self.time_slider = Slider(ax_time, 'Seek (s)', 0.0, 100.0)
        self.time_slider.on_changed(self.play_song_file)

        self.box_freq = TextBox(plt.axes([0.15, 0.15, 0.15, 0.04]), "Freq:", initial="")
        self.box_freq.on_submit(self.submit_freq)

        self.box_western = TextBox(plt.axes([0.40, 0.15, 0.15, 0.04]), "Note:", initial="")
        self.box_western.on_submit(self.submit_western)

        self.btn_save = Button(plt.axes([0.60, 0.15, 0.15, 0.04]), 'SAVE', color='lightgreen', hovercolor='lime')
        self.btn_save.on_clicked(self.save_annotation)

        y_row2 = 0.05
        h_btn = 0.04

        self.btn_play = Button(plt.axes([0.15, y_row2, 0.08, h_btn]), 'Play')
        self.btn_play.on_clicked(self.play_song_file)
        self.btn_pause = Button(plt.axes([0.24, y_row2, 0.08, h_btn]), 'Pause')
        self.btn_pause.on_clicked(self.pause_song)
        self.btn_stop = Button(plt.axes([0.33, y_row2, 0.08, h_btn]), 'Stop')
        self.btn_stop.on_clicked(self.stop_all)
        self.btn_tone = Button(plt.axes([0.42, y_row2, 0.12, h_btn]), 'Hear Tone')
        self.btn_tone.on_clicked(self.play_current_tonic)

        self.btn_prev = Button(plt.axes([0.60, y_row2, 0.08, h_btn]), '<  Prev')
        self.btn_prev.on_clicked(self.prev_file)
        self.btn_next = Button(plt.axes([0.69, y_row2, 0.08, h_btn]), 'Next  >')
        self.btn_next.on_clicked(self.next_file)
        self.btn_auto = Button(plt.axes([0.78, y_row2, 0.08, h_btn]), 'Auto')
        self.btn_auto.on_clicked(self.auto_snap)

        self.btn_fullplot = Button(plt.axes([0.80, 0.15, 0.15, 0.04]), 'Full Plot', color='skyblue', hovercolor='cyan')
        self.btn_fullplot.on_clicked(self.plot_full_song_view)

        self.txt_note = self.fig.text(0.15, 0.35, "--", fontsize=12, fontweight='bold', color='darkred')
        self.txt_score = self.fig.text(0.55, 0.35, "--", fontsize=12, fontweight='bold', color='green')


def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


if __name__ == "__main__":
    # Filter to only files that actually exist on disk
    existing = [p for p in FAILURE_PATHS if os.path.exists(p)]
    missing  = [p for p in FAILURE_PATHS if not os.path.exists(p)]

    if missing:
        print(f"\nWARNING: {len(missing)} files not found on disk:")
        for m in missing:
            print(f"  MISSING: {m}")

    if existing:
        print(f"\nLoading {len(existing)} REAL failure cases...")
        app = InteractiveTonicApp(existing)
        plt.show()
    else:
        print("No failure files found. Check FAILURE_PATHS.")
