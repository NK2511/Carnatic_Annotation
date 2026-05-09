import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, Audio, clear_output
import os
import sys
from scipy.interpolate import PchipInterpolator, CubicSpline
import librosa
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import sys
import os

# Add the new CNN folder to the path so it can find the dictionary!
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Method_Jaccard_CNN'))
from melakarta_signatures import MELAKARTA_RAGAS as COMMON_RAGAS, get_allowed_swaras

try:
    from carnatic_functions import get_raaga_context, normalize_carva_df, CARNATIC_RATIOS, interpolate_list
except ImportError:
    # Minimal fallback constants if import fails
    CARNATIC_RATIOS = {
        'S': 1.0, 'R1': 16/15, 'R2': 9/8, 'G2': 6/5, 'G3': 5/4, 'M1': 4/3, 'M2': 45/32,
        'P': 1.5, 'D1': 8/5, 'D2': 5/3, 'N2': 16/9, 'N3': 15/8, "S'": 2.0
    }
    def get_raaga_context(d): return {"carva_csv": os.path.join(d, "carva.csv")}
    def normalize_carva_df(df): return df

# NN components removed as requested

def view_primary_segments_interactive(song_index, audio_dir=None, csv_path=None):
    """
    Interactive widget to view segments with:
    - Carnatic Swara Background
    - Audio Playback
    - Spline Fitting (DP Alignment + Pchip)
    - Overlap Filtering (Default: Non-Overlapping)
    """
    if audio_dir is None:
        audio_dir = os.getcwd()
        
    # 1. Load Data
    carva_path = csv_path
    if not carva_path:
        try:
            context = get_raaga_context(audio_dir)
            carva_path = context["carva_csv"]
        except:
            carva_path = os.path.join(audio_dir, "carva.csv")
            
    if not os.path.exists(carva_path):
        print(f"Error: CARVA file not found at {carva_path}")
        return

    df = pd.read_csv(carva_path)
    df = normalize_carva_df(df)
    
    song_df = df[df['Index'] == song_index].copy()
    if song_df.empty:
        print(f"No segments found for Song Index {song_index}")
        return
        
    if 'StartFrame' in song_df.columns:
        song_df = song_df.sort_values('StartFrame')
    
    song_df = song_df.reset_index(drop=True)
    
    # FILTER OVERLAPS (Default Behavior)
    # Primary clustering typically wipes data after finding a segment,
    # so segments *should* be non-overlapping. However, we enforce this 
    # visually to ensure a clean timeline.
    
    non_overlapping_indices = []
    if not song_df.empty:
        last_end = -1
        # Sort by StartFrame just in case
        sorted_indices = song_df.sort_values('StartFrame').index.tolist()
        
        for idx in sorted_indices:
            start = song_df.at[idx, 'StartFrame']
            end = song_df.at[idx, 'EndFrame']
            
            # Simple greedy non-overlap
            if start >= last_end:
                non_overlapping_indices.append(idx)
                last_end = end
                
    # Create two views
    df_all = song_df.copy()
    df_no_overlap = song_df.loc[non_overlapping_indices].sort_values('StartFrame').reset_index(drop=True)
    
    # Default to No Overlap
    active_df = df_no_overlap
    total_segments = len(active_df)
    
    # 2. Audio Setup
    audio_data = None
    sr = 44100
    
    def load_audio():
        nonlocal audio_data, sr
        if audio_data is not None: return
        
        try:
            csv_path_raw = song_df.iloc[0]['AudioPath']
            fname = os.path.basename(csv_path_raw)
            csv_dir = os.path.dirname(carva_path)
            grandparent = os.path.dirname(csv_dir)
            
            candidates = [
                csv_path_raw,
                os.path.join(audio_dir, fname),
                os.path.join(csv_dir, fname),
            ]
            if os.path.exists(grandparent):
                for d in os.listdir(grandparent):
                    if "Vocals" in d:
                        candidates.append(os.path.join(grandparent, d, fname))
            
            final_audio_path = None
            for c in candidates:
                if os.path.exists(c):
                    final_audio_path = c
                    break
            
            if final_audio_path:
                print(f"Loading audio from: {final_audio_path}")
                audio_data, sr = librosa.load(final_audio_path, sr=None)
            else:
                print(f"Warning: Audio file '{fname}' not found.")
        except Exception as e:
            print(f"Audio load error: {e}")

    # 3. Visualization
    fig = go.FigureWidget(
        layout=go.Layout(
            title=dict(text=f"Song {song_index} - Segment Explorer"),
            xaxis=dict(title="Frame Index", zeroline=False),
            yaxis=dict(title="Ratio", range=[0.5, 2.5], gridcolor='rgba(128,128,128,0.2)'),
            hovermode="closest",
            template="plotly_dark",
            height=600,
            margin=dict(l=20, r=20, t=40, b=20)
        )
    )
    
    # Traces
    # 0: Segment
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Segment', line=dict(color='cyan', width=3)))
    # 1: Spline
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Spline', line=dict(color='yellow', width=2, dash='dash')))
    # 2: Control Points (NEW)
    # 2: Control Points (NEW)
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Control Points', 
                             marker=dict(color='magenta', size=10, symbol='x')))
    # 3: Stable Points (Red Squares)
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Stable Points', 
                             marker=dict(color='red', size=9, symbol='square')))

    # Background Lines
    shapes = []
    annotations = []
    sorted_ratios = sorted(CARNATIC_RATIOS.items(), key=lambda x: x[1])
    for name, ratio in sorted_ratios:
        if 0.5 <= ratio <= 2.5:
            shapes.append(dict(
                type="line", x0=0, x1=1, xref="paper", y0=ratio, y1=ratio,
                line=dict(color="rgba(100,100,100,0.5)", width=1, dash="dot")
            ))
            annotations.append(dict(
                x=1, xref="paper", y=ratio, text=name,
                showarrow=False, xanchor="left", font=dict(size=10, color="gray")
            ))
    # 4. Controls
    btn_prev = widgets.Button(description="<< Prev", icon='arrow-left')
    btn_next = widgets.Button(description="Next >>", icon='arrow-right')
    lbl_info = widgets.Label(value="Segment Info")
    
    txt_swaras = widgets.Text(placeholder="e.g. S R1", description="Swaras:")
    btn_fit = widgets.Button(description="Fit Spline", button_style='info')
    btn_auto = widgets.Button(description="Auto Detect", button_style='warning')
    
    out_audio = widgets.Output()
    btn_play = widgets.Button(description="Play Audio", icon='play')
    
    # State
    current_idx = 0
    current_clean_y = None
    
    # NN Cleaning Toggle
    chk_nn_clean = widgets.Checkbox(value=False, description='Apply NN Clean (Exp)')
    
    MAX_FRAMES = 60 # User requested fixed length

    def get_segment_data(idx):
        row = active_df.iloc[idx]
        try:
            seg_raw = row['SegmentList']
            data = json.loads(seg_raw) if isinstance(seg_raw, str) else seg_raw
            arr = np.array(data)
            
            # Interpolate to fixed frames (MAX_FRAMES=60)
            if len(arr) > 0:
                arr = interpolate_list(arr, MAX_FRAMES)
            else:
                arr = np.zeros(MAX_FRAMES) # Fallback
                
            return arr, row
        except:
            return np.zeros(MAX_FRAMES), row

    # Explicit Event Handlers (No Lambdas)
    def on_prev_click(b):
        update_state(-1)
        update_view()
        
    def on_next_click(b):
        update_state(1)
        update_view()
    
    def update_state(delta):
        nonlocal current_idx
        current_idx += delta
    
    btn_prev.on_click(on_prev_click)
    btn_next.on_click(on_next_click)      
        
    def update_view():
        nonlocal current_idx, total_segments
        
        # Safety bound check
        total_segments = len(active_df)
        
        if total_segments == 0:
            current_idx = 0
            # Clear plot
            with fig.batch_update():
               fig.data[0].x, fig.data[0].y = [], []
               fig.data[1].x, fig.data[1].y = [], []
               fig.data[2].x, fig.data[2].y = [], []
               fig.data[3].x, fig.data[3].y = [], []
               fig.layout.title.text = "No Segments Match Criteria"
            lbl_info.value = "No segments found"
            return

        current_idx = max(0, min(current_idx, total_segments - 1))
        
        try:
            y_data, row = get_segment_data(current_idx)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return

        x_data = np.arange(len(y_data))
        
        # Apply NN Cleaning if requested
        # Default Plot Vars
        y_plot = y_data
        x_plot = x_data
        
        with fig.batch_update():
            # Trace 0: Original Data (always show)
            fig.data[0].x = x_plot.tolist() if isinstance(x_plot, np.ndarray) else list(x_plot)
            fig.data[0].y = y_plot.tolist() if isinstance(y_plot, np.ndarray) else list(y_plot)
            
            # Update Axes to fixed 60
            fig.layout.xaxis.range = [0, MAX_FRAMES]
            
            if len(y_data) > 0:
                ymin, ymax = np.min(y_data), np.max(y_data)
                padding = (ymax - ymin)*0.2
                target_min = max(0.5, ymin - padding)
                target_max = min(3.0, ymax + padding)
                fig.layout.yaxis.range = [target_min, target_max]

            label = row.get('Primary_Label', 'N/A')
            start = row.get('StartFrame', '?')
            end = row.get('EndFrame', '?')
            mode_txt = "All" if chk_overlap.value else "Non-Overlap"
            fig.layout.title.text = f"Segment {current_idx+1}/{total_segments} ({mode_txt}) | Label: {label} | Frames: {start}-{end}"
            
            # Trace 1 & 2: Spline & Knots (Only if NN active)
            if chk_nn_clean.value and spline_cleaner is not None:
                 x_knots, y_knots = spline_cleaner.predict_knots(y_data)
                 
                 if x_knots is not None:
                     # Generate smooth curve from knots
                     cs = PchipInterpolator(x_knots, y_knots)
                     x_new = np.linspace(0, len(y_data)-1, MAX_FRAMES)
                     y_new = cs(x_new)
                     
                     fig.data[1].x = x_new.tolist()
                     fig.data[1].y = y_new.tolist()
                     fig.data[2].x = x_knots.tolist()
                     fig.data[2].y = y_knots.tolist()
                 else:
                     helper_clear_spline()
            else:
                 helper_clear_spline()
        
        lbl_info.value = f"ID: {current_idx} | Start: {row.get('StartFrame')} | End: {row.get('EndFrame')}"
        out_audio.clear_output()
        txt_swaras.value = ""

    def helper_clear_spline():
        with fig.batch_update():
            fig.data[1].x = []
            fig.data[1].y = [] 
            fig.data[2].x = []
            fig.data[2].y = [] 
            fig.data[3].x = [] # Clear new trace
            fig.data[3].y = [] # Clear new trace

    def spline_fit_logic(y_data, user_ratios):
        """
        Fits spline through Start + UserRatios + End.
        Uses Dynamic Programming to optimally align UserRatios to the signal
        while preserving temporal order and enforcing minimum spacing.
        """
        n_frames = len(y_data)
        start_y = y_data[0]
        end_y = y_data[-1]
        
        x_knots = [0]
        y_knots = [start_y]
        
        M = len(user_ratios)
        if M == 0:
             x_knots.append(n_frames - 1)
             y_knots.append(end_y)
             return x_knots, y_knots

        # Dynamic Programming to find indices t_0 < t_1 < ... < t_{M-1}
        # that minimize Sum(|signal[t_k] - user_ratio[k]|)
        
        min_dist = max(2, int(n_frames / (M * 4))) 
        
        dp = np.full((M, n_frames), np.inf)
        parent = np.full((M, n_frames), -1, dtype=int)
        
        r0 = user_ratios[0]
        for j in range(1, n_frames - 1):
             dp[0][j] = abs(y_data[j] - r0)

        for i in range(1, M):
             curr_ratio = user_ratios[i]
             min_prev_cost = np.inf
             best_prev_idx = -1
             k = 0 
             for j in range(1, n_frames - 1):
                 target_k = j - min_dist
                 while k <= target_k and k < n_frames:
                     if dp[i-1][k] < min_prev_cost:
                         min_prev_cost = dp[i-1][k]
                         best_prev_idx = k
                     k += 1
                     
                 if min_prev_cost != np.inf:
                     cost = abs(y_data[j] - curr_ratio) + min_prev_cost
                     dp[i][j] = cost
                     parent[i][j] = best_prev_idx

        best_end_val = np.inf
        best_end_idx = -1
        
        last_row = M - 1
        for j in range(1, n_frames - 1):
            if dp[last_row][j] < best_end_val:
                best_end_val = dp[last_row][j]
                best_end_idx = j
                
        if best_end_idx == -1:
            return np.linspace(0, n_frames-1, M+2).astype(int), [start_y] + user_ratios + [end_y]
            
        path_indices = [0] * M
        curr = best_end_idx
        for i in range(M - 1, -1, -1):
            path_indices[i] = curr
            curr = parent[i][curr]
            
        x_knots.extend(path_indices)
        y_knots.extend(user_ratios)
        x_knots.append(n_frames - 1)
        y_knots.append(end_y)
        
        return x_knots, y_knots

    def on_fit_click(_):
        swara_str = txt_swaras.value.strip()
        y_data, _ = get_segment_data(current_idx)
        if len(y_data) == 0: return

        user_ratios = []
        if swara_str:
            for s in swara_str.split():
                val = CARNATIC_RATIOS.get(s)
                if not val: val = CARNATIC_RATIOS.get(s + "'")
                if not val: val = CARNATIC_RATIOS.get(s + "_")
                if not val: val = CARNATIC_RATIOS.get(s[:1]) 
                if val: user_ratios.append(val)
        
        x_knots, y_knots = spline_fit_logic(y_data, user_ratios)
        
        if len(x_knots) < 2: return
        
        x_knots = sorted(list(set(x_knots))) 
        if len(x_knots) != len(y_knots):
             x_knots = np.linspace(0, len(y_data)-1, len(y_knots))
        
        cs = PchipInterpolator(x_knots, y_knots)
        x_new = np.arange(len(y_data))
        y_new = cs(x_new)
        
        with fig.batch_update():
            # Show Control points 
            if len(x_knots) > 2:
                fig.data[2].x = x_knots[1:-1]
                fig.data[2].y = y_knots[1:-1]
            else:
                fig.data[2].x = []
                fig.data[2].y = [] 
            
            fig.data[1].x = x_new
            fig.data[1].y = y_new

    # --- Refactored Swara Detection ---
    def detect_swaras_from_values(y_values, target_points=16, raga=None):
        if len(y_values) == 0: return ""
        
        # Prepare Valid Candidates
        valid_candidates = list(CARNATIC_RATIOS.items())
        if raga and raga != 'None':
            allowed = get_allowed_swaras(raga)
            if allowed:
                print(f"DEBUG: Raga={raga}, Applying Mask: {sorted(list(allowed))}")
                filtered = [x for x in valid_candidates if x[0] in allowed]
                if filtered: valid_candidates = filtered
            else:
                print(f"DEBUG: Raga={raga}, No allowed notes found.")

        detected_events = [] 

        # 1. INITIAL DETECTION: Capture EVERYTHING (Max Sensitivity)
        peaks, _ = find_peaks(y_values, prominence=0.01)
        valleys, _ = find_peaks(-y_values, prominence=0.01)
        
        for p in peaks:
            detected_events.append((float(p), y_values[p], 'peak'))
        for v in valleys:
            detected_events.append((float(v), y_values[v], 'valley'))
            
        # Add basic plateaus
        # Two-rule check:
        #   Rule 1 (per-frame): |dy/dt| < 0.05 for >= 3 consecutive frames
        #   Rule 2 (range):     max(region) - min(region) <= 0.05
        #   Rule 2 prevents slow gradual glides being misclassified as held notes.
        PLATEAU_RANGE_THRESH = 0.05
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
                    region = y_values[run_start : run_end+1]
                    if run_end - run_start >= min_plateau_len and (np.max(region) - np.min(region)) <= PLATEAU_RANGE_THRESH:
                        p_val = np.median(region)
                        detected_events.append((float(run_start), p_val, 'plateau_start'))
                        detected_events.append((float(run_end), p_val, 'plateau_end'))
                    run_start = -1
        
        if run_start != -1:
            region = y_values[run_start:]
            if (np.max(region) - np.min(region)) <= PLATEAU_RANGE_THRESH:
                p_val = np.median(region)
                detected_events.append((float(run_start), p_val, 'plateau_start'))
                detected_events.append((float(len(is_flat)), p_val, 'plateau_end'))


        # Ensure Boundaries (Float)
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
        
        # DEBUG: Print final points
        print(f"Segment Len: {len(y_values)}, Requested Intermediate: {target_points} (Total: {total_target}), Final Count: {len(detected_events)}")
        pts_coords = [ (round(e[0], 2), round(e[1], 2)) for e in detected_events ]
        print(f"Points: {pts_coords}")

        detected_names = []
        for idx_t, val, ev_type in detected_events:
            # Skip extreme edges to avoid artifacts
            if idx_t < 2 or idx_t > len(y_values)-3: 
                continue
            
            best_swara = min(valid_candidates, key=lambda x: abs(x[1] - val))
            detected_names.append(best_swara[0])
            
        # Ultimate Fallback: If filtering removed everything (or nothing found),
        # use the median of the entire segment.
        if not detected_names:
            median_val = np.median(y_values)
            best_swara = min(valid_candidates, key=lambda x: abs(x[1] - median_val))
            return best_swara[0]
            return best_swara[0], detected_events
            
        return " ".join(detected_names), detected_events

    def detect_swaras_for_segment(idx_to_check, target_points=16, raga=None):
        y_data_seg, _ = get_segment_data(idx_to_check)
        s, _ = detect_swaras_from_values(y_data_seg, target_points, raga)
        return s

    def on_auto_click(_):
        n_pts = slider_num_points.value
        raga_val = dd_raga.value
        y_data_seg, _ = get_segment_data(current_idx)
        
        out_notation.clear_output()
        with out_notation:
            # Get Swaras AND Events
            swaras, events = detect_swaras_from_values(y_data_seg, target_points=n_pts, raga=raga_val)
        
        txt_swaras.value = swaras
        
        # Fit Spline (Trace 1 & 2)
        on_fit_click(None)
        
        # Update Trace 3 (Stable Points - Red Squares)
        plateau_x = []
        plateau_y = []
        for p in events:
             # p is (index, val, type)
             # Check if type is 'plateau_center'
             if len(p) > 2 and p[2] == 'plateau_center':
                 plateau_x.append(p[0])
                 plateau_y.append(p[1])
        
        with fig.batch_update():
            # Ensure Trace 3 exists (might need check if figure re-initialized)
            if len(fig.data) > 3:
                fig.data[3].x = plateau_x
                fig.data[3].y = plateau_y

    # --- New Function: Generate Full Notation ---
    def on_generate_notation_click(_):
        n_pts = slider_num_points.value
        raga_val = dd_raga.value
        out_notation.clear_output()
        with out_notation:
            print(f"Generating notation for {len(active_df)} segments (Points: {n_pts}, Raga: {raga_val})...")
            
            full_notation_lines = []
            
            for idx in range(len(active_df)):
                # This wrapper now handles the unpacking internally
                seg_swaras = detect_swaras_for_segment(idx, target_points=n_pts, raga=raga_val)
                if seg_swaras:
                    # Append formatted string: [Time] Swaras
                    row = active_df.iloc[idx]
                    start_f = row.get('StartFrame', 0)
                    full_notation_lines.append(f"[{start_f}]: {seg_swaras}")
            
            print("\n".join(full_notation_lines))
            print("\n--- End of Notation ---")

    def on_play_click(_):
        load_audio()
        if audio_data is None: 
            with out_audio: print("Audio file unavailable.")
            return
            
        step_ms = 20
        _, row = get_segment_data(current_idx)
        start_ms = row['StartFrame'] * step_ms
        end_ms = row['EndFrame'] * step_ms
        
        start_s = max(0, start_ms/1000 - 0.05)
        end_s = min(len(audio_data)/sr, end_ms/1000 + 0.05)
        
        segment_audio = audio_data[int(start_s*sr):int(end_s*sr)]
        
        out_audio.clear_output()
        with out_audio:
            display(Audio(segment_audio, rate=sr, autoplay=True))

    def on_overlap_change(change):
        nonlocal active_df, current_idx, total_segments
        if change['new']:
            active_df = df_all
        else:
            active_df = df_no_overlap
            
        total_segments = len(active_df)
        current_idx = 0 
        update_view()

    chk_overlap = widgets.Checkbox(value=False, description='Show Overlaps')
    chk_overlap.observe(on_overlap_change, names='value')

    # Slider for Number of Anchor Points
    slider_num_points = widgets.IntSlider(
        value=16, 
        min=4, 
        max=30, 
        step=1, 
        description='Num Points:',
        continuous_update=False,
        layout=widgets.Layout(width='300px')
    )
    
    # Auto-detect Raga from filename
    csv_filename = os.path.basename(csv_path).lower()
    default_raga = 'None'
    for r in COMMON_RAGAS.keys():
        if r.lower() in csv_filename:
            default_raga = r
            break
            
    if default_raga != 'None':
        notes = sorted(list(get_allowed_swaras(default_raga)))
        print(f"🎵 Auto-selected Raga: {default_raga}")
        print(f"   Allowed Notes: {notes}")

    # Raga Selection Dropdown
    raga_options = ['None'] + sorted(list(COMMON_RAGAS.keys()))
    dd_raga = widgets.Dropdown(
        options=raga_options,
        value=default_raga,
        description='Raga:',
        layout=widgets.Layout(width='200px')
    )

    btn_notation = widgets.Button(description="Generate Full Notation", button_style='success', icon='music')
    out_notation = widgets.Output(layout={'border': '1px solid #444', 'height': '200px', 'overflow_y': 'scroll'})

    # Event Wiring
    btn_fit.on_click(on_fit_click)
    btn_auto.on_click(on_auto_click)
    btn_play.on_click(on_play_click)
    btn_notation.on_click(on_generate_notation_click)
    
    # New NN Detect Logic
    btn_nn_detect = widgets.Button(description="NN Detect")

    def on_nn_detect_click(b):
        # 1. Force NN View ON
        chk_nn_clean.value = True
        update_view()
        
        # 2. Extract swaras from the NN Spline (Calculated locally to be safe)
        idx = current_idx
        y_raw, _ = get_segment_data(idx)
        
        if len(y_raw) > 0 and spline_cleaner is not None:
             x_knots, y_knots = spline_cleaner.predict_knots(y_raw)
             if x_knots is not None:
                 cs = PchipInterpolator(x_knots, y_knots)
                 x_new = np.linspace(0, len(y_raw)-1, MAX_FRAMES)
                 y_new = cs(x_new)
                 
                 # Detect on NN Curve
                 n_pts = slider_num_points.value
                 raga_val = dd_raga.value
                 swaras, _ = detect_swaras_from_values(y_new, target_points=n_pts, raga=raga_val)
                 
                 txt_swaras.value = f"[NN] {swaras}"
                 
             else:
                 txt_swaras.value = "NN Fail"
        else:
             txt_swaras.value = "No Data"

    btn_nn_detect.on_click(on_nn_detect_click)

    # --- Evaluation Utility (Auto-Label Mode) ---
    def run_evaluate_entire_song(_):
        out_notation.clear_output()
        with out_notation:
            print(f"🚀 Evaluating entire song (Auto-Label Mode, Sensitivity=0.14): {csv_path}")
            
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"❌ Error loading CSV: {e}")
                return

            rmse_list = []
            inlier_list = []
            results = []
            
            valid_ratios = list(CARNATIC_RATIOS.values())
            valid_cents = 1200 * np.log2(valid_ratios)

            print(f"Processing {len(df)} segments...")
            
            for idx, row in df.iterrows():
                try:
                    y_raw = np.array(json.loads(row['SegmentList']))
                    if len(y_raw) < 5: continue
                    
                    # 1. Detect Swaras (Target Points 16)
                    swara_str, _ = detect_swaras_from_values(y_raw, target_points=16)
                    if not swara_str: continue
                    
                    # 2. Convert to Ratios
                    user_ratios = []
                    for s in swara_str.split():
                        val = CARNATIC_RATIOS.get(s)
                        if not val: val = CARNATIC_RATIOS.get(s + "'")
                        if not val: val = CARNATIC_RATIOS.get(s + "_")
                        if not val: val = CARNATIC_RATIOS.get(s[:1]) 
                        if val: user_ratios.append(val)
                    if not user_ratios: continue
                    
                    # 3. Fit Spline
                    x_knots, y_knots = spline_fit_logic(y_raw, user_ratios)
                    
                    knots_combined = sorted(list(set(zip(x_knots, y_knots))), key=lambda x: x[0])
                    x_knots = [k[0] for k in knots_combined]
                    y_knots = [k[1] for k in knots_combined]
                    
                    if len(x_knots) < 2: continue
                    
                    # 4. Generate Spline
                    cs = PchipInterpolator(x_knots, y_knots)
                    target_len = MAX_FRAMES
                    x_new = np.linspace(0, len(y_raw)-1, target_len)
                    y_spline = cs(x_new)
                    
                    # 5. Metrics
                    original_x = np.linspace(0, 1, len(y_raw))
                    target_x = np.linspace(0, 1, target_len)
                    y_raw_resampled = np.interp(target_x, original_x, y_raw)
                    
                    cents_error = 1200 * np.log2(y_spline / y_raw_resampled)
                    rmse = np.sqrt(np.mean(cents_error**2))
                    
                    spline_cents = 1200 * np.log2(y_spline)
                    diffs = np.abs(spline_cents[:, None] - valid_cents[None, :])
                    min_diffs = np.min(diffs, axis=1)
                    inliers = np.sum(min_diffs < 25.0) 
                    inlier_pct = (inliers / len(y_spline)) * 100
                    
                    rmse_list.append(rmse)
                    inlier_list.append(inlier_pct)
                    results.append({'id': idx, 'rmse': rmse, 'adherence': inlier_pct})
                    
                except Exception as e:
                    continue
            
            if not rmse_list:
                print("No valid segments processed.")
                return

            avg_rmse = np.mean(rmse_list)
            avg_inlier = np.mean(inlier_list)
            
            results.sort(key=lambda x: x['adherence'])
            
            print("\n✅ === EVALUATION REPORT ===")
            print(f"Total Segments: {len(results)}")
            print(f"Average RMSE (Fidelity): {avg_rmse:.2f} cents")
            print(f"Average Raga Adherence: {avg_inlier:.2f}%")
            print("\n🏆 Best Segments (Highest Adherence):")
            for s in reversed(results[-3:]):
                print(f"   Seg {s['id']}: {s['adherence']:.1f}% (RMSE: {s['rmse']:.1f})")

    btn_eval_all = widgets.Button(description="Eval All", icon='table')
    btn_eval_all.on_click(run_evaluate_entire_song)

    # --- Metrics Logic ---
    btn_metrics = widgets.Button(description="Metrics", icon='chart-bar')
    
    def on_metrics_click(b):
       # ... (rest of function as is)
        out_notation.clear_output()
        with out_notation:
            # 1. Get Data
            y_raw, _ = get_segment_data(current_idx)
            
            # Get Spline Data (Trace 1)
            if len(fig.data[1].y) == 0:
                print("⚠️ No Spline generated yet. Click 'Fit Spline' or 'Auto Label' first.")
                return
                
            y_spline = np.array(fig.data[1].y)
            
            # Ensure lengths match (interpolate raw if needed, but spline usually matches MAX_FRAMES)
            # y_raw is usually variable length, y_spline is MAX_FRAMES (60).
            # We must interpolate y_raw to 60 to compare point-to-point.
            if len(y_raw) != len(y_spline):
                original_x = np.linspace(0, 1, len(y_raw))
                target_x = np.linspace(0, 1, len(y_spline))
                y_raw_resampled = np.interp(target_x, original_x, y_raw)
            else:
                y_raw_resampled = y_raw
                
            # 2. RMSE (Cents)
            # 1200 * log2(ratio_error)
            cents_error = 1200 * np.log2(y_spline / y_raw_resampled)
            rmse = np.sqrt(np.mean(cents_error**2))
            
            print(f"📊 metrics for Segment {current_idx}:")
            print(f"   • RMSE (Fidelity): {rmse:.2f} cents")
            
            # 3. Raga Inlier %
            # Count points close to valid notes
            # Tolerance: 20 cents (approx 0.012 ratio difference around 1.0)
            # Lets work in Cents space relative to 1.0
            
            spline_cents = 1200 * np.log2(y_spline)
            
            # Valid Notes in Cents
            valid_ratios = list(CARNATIC_RATIOS.values())
            valid_cents = 1200 * np.log2(valid_ratios)
            
            # Check for each point if it's within 20 cents of ANY valid note
            # Using broadcasting: (60, 1) - (1, 12)
            diffs = np.abs(spline_cents[:, None] - valid_cents[None, :])
            min_diffs = np.min(diffs, axis=1) # Closest note distance
            
            inliers = np.sum(min_diffs < 25.0) # 25 cents tolerance
            inlier_pct = (inliers / len(y_spline)) * 100
            
            print(f"   • Raga Adherence: {inlier_pct:.1f}% (points within 25 cents of scale)")
            
            # 4. Histogram
            plt.figure(figsize=(6, 3))
            plt.hist(spline_cents, bins=30, color='gold', edgecolor='black', alpha=0.7, label='Spline Pitch')
            # Plot valid lines
            for vc, name in zip(valid_cents, CARNATIC_RATIOS.keys()):
                plt.axvline(vc, color='red', linestyle='--', alpha=0.3)
                if 0 <= vc <= 1200: # Only plot relevant ones
                    plt.text(vc, 1, name, rotation=90, verticalalignment='bottom', fontsize=8, color='red')
            
            plt.title("Pitch Distribution (Cents)")
            plt.xlabel("Cents from Sa")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

    btn_metrics.on_click(on_metrics_click)

    # --- Evaluation Utility (Auto-Label Mode) ---
    def run_evaluate_entire_song(_):
        out_notation.clear_output()
        with out_notation:
            print(f"🚀 Evaluating entire song (Auto-Label Mode, Sensitivity=0.14): {csv_path}")
            
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"❌ Error loading CSV: {e}")
                return

            rmse_list = []
            inlier_list = []
            results = []
            
            valid_ratios = list(CARNATIC_RATIOS.values())
            valid_cents = 1200 * np.log2(valid_ratios)

            print(f"Processing {len(df)} segments...")
            
            for idx, row in df.iterrows():
                try:
                    y_raw = np.array(json.loads(row['SegmentList']))
                    if len(y_raw) < 5: continue
                    
                    # 1. Detect Swaras (Sensitivity 0.14)
                    swara_str = detect_swaras_from_values(y_raw, prominence_val=0.14)
                    if not swara_str: continue
                    
                    # 2. Convert to Ratios
                    user_ratios = []
                    for s in swara_str.split():
                        val = CARNATIC_RATIOS.get(s)
                        if not val: val = CARNATIC_RATIOS.get(s + "'")
                        if not val: val = CARNATIC_RATIOS.get(s + "_")
                        if not val: val = CARNATIC_RATIOS.get(s[:1]) 
                        if val: user_ratios.append(val)
                    if not user_ratios: continue
                    
                    # 3. Fit Spline
                    x_knots, y_knots = spline_fit_logic(y_raw, user_ratios)
                    
                    knots_combined = sorted(list(set(zip(x_knots, y_knots))), key=lambda x: x[0])
                    x_knots = [k[0] for k in knots_combined]
                    y_knots = [k[1] for k in knots_combined]
                    
                    if len(x_knots) < 2: continue
                    
                    # 4. Generate Spline
                    cs = PchipInterpolator(x_knots, y_knots)
                    target_len = MAX_FRAMES
                    x_new = np.linspace(0, len(y_raw)-1, target_len)
                    y_spline = cs(x_new)
                    
                    # 5. Metrics
                    original_x = np.linspace(0, 1, len(y_raw))
                    target_x = np.linspace(0, 1, target_len)
                    y_raw_resampled = np.interp(target_x, original_x, y_raw)
                    
                    cents_error = 1200 * np.log2(y_spline / y_raw_resampled)
                    rmse = np.sqrt(np.mean(cents_error**2))
                    
                    spline_cents = 1200 * np.log2(y_spline)
                    diffs = np.abs(spline_cents[:, None] - valid_cents[None, :])
                    min_diffs = np.min(diffs, axis=1)
                    inliers = np.sum(min_diffs < 25.0) 
                    inlier_pct = (inliers / len(y_spline)) * 100
                    
                    rmse_list.append(rmse)
                    inlier_list.append(inlier_pct)
                    results.append({'id': idx, 'rmse': rmse, 'adherence': inlier_pct})
                    
                except Exception as e:
                    continue
            
            if not rmse_list:
                print("No valid segments processed.")
                return

            avg_rmse = np.mean(rmse_list)
            avg_inlier = np.mean(inlier_list)
            
            results.sort(key=lambda x: x['adherence'])
            
            print("\n✅ === AUTO-LABEL EVALUATION REPORT ===")
            print(f"Total Segments: {len(results)}")
            print(f"Average RMSE (Fidelity): {avg_rmse:.2f} cents")
            print(f"Average Raga Adherence: {avg_inlier:.2f}%")
            print("\n🏆 Best Segments (Highest Adherence):")
            for s in reversed(results[-3:]):
                print(f"   Seg {s['id']}: {s['adherence']:.1f}% (RMSE: {s['rmse']:.1f})")
            
            print("\n⚠️ Worst Segments (Lowest Adherence):")
            for s in results[:3]:
                print(f"   Seg {s['id']}: {s['adherence']:.1f}% (RMSE: {s['rmse']:.1f})")

    btn_eval_all = widgets.Button(description="Eval All", icon='table')
    btn_eval_all.on_click(run_evaluate_entire_song)

# NN detect functionality removed

    # --- Layout ---
    chk_nn_clean.observe(lambda c: update_view(), names='value')

    nav_row = widgets.HBox([btn_prev, btn_next, lbl_info, chk_overlap])
    # Add slider to spline row or new row
    spline_controls = widgets.HBox([widgets.Label("Spline:"), txt_swaras, btn_fit, btn_auto])
    sensitivity_row = widgets.HBox([slider_num_points, dd_raga]) # Added raga dropdown
    audio_row = widgets.HBox([btn_play, out_audio])
    notation_row = widgets.VBox([btn_notation, out_notation])
    
    ui = widgets.VBox([fig, nav_row, spline_controls, sensitivity_row, audio_row, notation_row])
    
    update_view()
    display(ui)
