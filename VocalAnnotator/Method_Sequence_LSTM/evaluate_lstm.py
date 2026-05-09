import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = r"C:\Desktop\Python\CarnaticAnnotater"
ALL_RAGAS = ["Kalyani", "Keeravani", "Kharaharapriya", "Mayamalavagowlai", "Panthuvarali", "Shankarabharanam", "Shanmukhapriya", "Thodi"]

# Identify Valid Ragas
RAGAS = []
for r in ALL_RAGAS:
    p1 = os.path.join(BASE_DIR, "Raagas", r, f"{r}_Data", f"carva_{r}.csv")
    p2 = os.path.join(BASE_DIR, "Raagas", r, f"{r}_Data", "carva.csv")
    if os.path.exists(p1) or os.path.exists(p2):
        RAGAS.append(r)
        
raga_to_id = {r: i for i, r in enumerate(RAGAS)}
NUM_CLASSES = len(RAGAS)
MAX_SEQ_LEN = 8  # MUST MATCH TRAINING SCRIPT


GAMAKAS_PER_CHUNK = 250  


EVAL_CHUNKS_PER_RAGA = 50  


USE_HIERARCHICAL = True

# ==========================
# REUSABLE FUNCTIONS
# ==========================
def get_sequence_from_segment(y_raw):
    W = len(y_raw)
    if W < 5: return None, W
    target_frames = 60
    x_old = np.linspace(0, 1, W)
    x_new = np.linspace(0, 1, target_frames)
    y_interp = np.interp(x_new, x_old, y_raw)
    
    peaks, _ = find_peaks(y_interp, prominence=0.01)
    valleys, _ = find_peaks(-y_interp, prominence=0.01)
    
    knots = [(0, y_interp[0])]
    for p in peaks: knots.append((p, y_interp[p]))
    for v in valleys: knots.append((v, y_interp[v]))
    knots.append((target_frames - 1, y_interp[-1]))
    
    knots = list({k[0]: k for k in knots}.values())
    knots.sort(key=lambda x: x[0])
    
    target_pts = min(MAX_SEQ_LEN, max(4, int(W / 3)))
    
    while len(knots) > target_pts:
        min_area = float('inf')
        min_idx = -1
        for i in range(1, len(knots) - 1):
            x1, y1 = knots[i-1]
            x2, y2 = knots[i]
            x3, y3 = knots[i+1]
            area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
            if area < min_area:
                min_area = area
                min_idx = i
        if min_idx != -1: knots.pop(min_idx)
        else: break
            
    sequence = []
    prev_t = knots[0][0] / target_frames
    for t_frame, y_val in knots:
        if y_val <= 0: continue
        t = t_frame / target_frames
        delta_t = t - prev_t
        prev_t = t
        semitone = 12 * np.log2(y_val)
        note_id = int(round(semitone)) % 12
        sequence.append([float(note_id), float(delta_t)])
    return sequence, W

def load_data_by_song():
    songs_data = [] 
    for raga in RAGAS:
        raga_id = raga_to_id[raga]
        carva_path = os.path.join(BASE_DIR, raga, f"{raga}_Data", f"carva_{raga}.csv")
        if not os.path.exists(carva_path):
            carva_path = os.path.join(BASE_DIR, raga, f"{raga}_Data", "carva.csv")
            if not os.path.exists(carva_path): continue
                
        df = pd.read_csv(carva_path)
        if 'Primary_Label' in df.columns:
            df = df[df['Primary_Label'] != -1]
            
        for song_idx, song_df in df.groupby('Index'):
            seqs, ws = [], []
            for _, row in song_df.iterrows():
                try:
                    y_raw = json.loads(row.get('SegmentList', '[]'))
                    if isinstance(y_raw, list):
                        seq, W = get_sequence_from_segment(np.array(y_raw))
                        if seq and len(seq) > 0:
                            seqs.append(seq)
                            ws.append(W)
                except:
                    pass
            if len(seqs) > 10: 
                songs_data.append({
                    'raga': raga_id,
                    'song_id': f"{raga}_{song_idx}",
                    'seqs': seqs,
                    'Ws': ws
                })
    return songs_data

def build_tensors(seq_list, w_list):
    n = len(seq_list)
    X = np.zeros((n, MAX_SEQ_LEN, 2))
    X[:, :, 0] = 12 
    W_arr = np.zeros((n, 1))
    
    for i in range(n):
        seq = seq_list[i]
        act_len = min(len(seq), MAX_SEQ_LEN)
        for j in range(act_len):
            X[i, j, 0] = seq[j][0]
            X[i, j, 1] = seq[j][1]
        W_arr[i, 0] = w_list[i]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(W_arr, dtype=torch.float32)

def build_hierarchical_tensors(chunk_list, w_list):
    n = len(chunk_list)
    X = np.zeros((n, GAMAKAS_PER_CHUNK, MAX_SEQ_LEN, 2))
    X[:, :, :, 0] = 12
    W_arr = np.zeros((n, GAMAKAS_PER_CHUNK, 1))
    
    for i in range(n):
        gamakas = chunk_list[i]
        for g_idx in range(len(gamakas)):
            gamaka = gamakas[g_idx]
            act_len = min(len(gamaka), MAX_SEQ_LEN)
            for j in range(act_len):
                X[i, g_idx, j, 0] = gamaka[j][0]
                X[i, g_idx, j, 1] = gamaka[j][1]
            W_arr[i, g_idx, 0] = w_list[i][g_idx]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(W_arr, dtype=torch.float32)

class RagaLSTM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(RagaLSTM, self).__init__()
        self.note_embedding = nn.Embedding(num_embeddings=13, embedding_dim=16, padding_idx=12)
        self.lstm = nn.LSTM(input_size=17, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(65, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x_seq, w_dur):
        # Flatten num_gamakas into Sequence dimension if standard mode is fed 4D tensors unexpectedly
        if x_seq.dim() == 4:
            B, num_gamakas, num_nodes, feats = x_seq.size()
            x_seq = x_seq.view(B * num_gamakas, num_nodes, feats)
            w_dur = w_dur.view(B * num_gamakas, 1)
            
        note_ids = x_seq[:, :, 0].long()
        delta_ts = x_seq[:, :, 1].unsqueeze(-1)
        embedded_notes = self.note_embedding(note_ids)
        lstm_input = torch.cat((embedded_notes, delta_ts), dim=2)
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        final_h = h_n[-1]
        dense_input = torch.cat((final_h, w_dur), dim=1)
        out = self.dropout(self.relu(self.fc1(dense_input)))
        out = self.fc2(out)
        return out

class HierarchicalRagaLSTM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(HierarchicalRagaLSTM, self).__init__()
        self.note_embedding = nn.Embedding(num_embeddings=13, embedding_dim=16, padding_idx=12)
        self.micro_lstm = nn.LSTM(input_size=17, hidden_size=32, num_layers=1, batch_first=True)
        self.macro_lstm = nn.LSTM(input_size=33, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x_seq, w_dur):
        # x_seq: (B, NumGamakas, MaxNodes, 2)
        if x_seq.dim() == 3:
            x_seq = x_seq.unsqueeze(0)
            w_dur = w_dur.unsqueeze(0)
            
        B, num_gamakas, num_nodes, feats = x_seq.size()
        x_flat = x_seq.view(B * num_gamakas, num_nodes, feats)
        
        note_ids = x_flat[:, :, 0].long()
        delta_ts = x_flat[:, :, 1].unsqueeze(-1)
        embedded_notes = self.note_embedding(note_ids)
        micro_input = torch.cat((embedded_notes, delta_ts), dim=2)
        
        _, (h_n, _) = self.micro_lstm(micro_input)
        gamaka_vectors = h_n[-1].view(B, num_gamakas, 32)
        
        macro_input = torch.cat((gamaka_vectors, w_dur), dim=2)
        macro_out, (h_n_macro, _) = self.macro_lstm(macro_input)
        final_macro_h = h_n_macro[-1]
        
        out = self.dropout(self.relu(self.fc1(final_macro_h)))
        out = self.fc2(out)
        return out

# ==========================
# FAST EVALUATION
# ==========================
def main():
    if USE_HIERARCHICAL:
        print(f"🚀 Loading Pre-Trained HIERARCHICAL Model to evaluate {GAMAKAS_PER_CHUNK}-Gamaka Audio Syntax...")
        model_path = os.path.join(BASE_DIR, "VocalAnnotator", "raga_hierarchical_lstm_model.pth")
        model = HierarchicalRagaLSTM(num_classes=NUM_CLASSES)
    else:
        print(f"🚀 Loading Pre-Trained STANDARD Model to evaluate {GAMAKAS_PER_CHUNK}-Gamaka Audio Chunks...")
        model_path = os.path.join(BASE_DIR, "VocalAnnotator", "raga_lstm_model.pth")
        model = RagaLSTM(num_classes=NUM_CLASSES)
        
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Please train it first!")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Isolate exact validation songs to guarantee no leak
    songs_data = load_data_by_song()
    random.seed(42)  # MUST match training seed to grab exact same held-out validation songs!
    random.shuffle(songs_data)
    
    val_songs = []
    raga_groups = {i: [] for i in range(NUM_CLASSES)}
    for s in songs_data:
        raga_groups[s['raga']].append(s)
        
    for r_id, s_list in raga_groups.items():
        split_idx = int(0.8 * len(s_list))
        val_songs.extend(s_list[split_idx:])

    # Generate Exact Number of Random Chunks per Raga
    balanced_val_chunks = []
    
    print(f"\nEvaluating exactly {EVAL_CHUNKS_PER_RAGA} randomly sliced test clips per raga.")
    
    for raga_idx in range(NUM_CLASSES):
        # All validation songs for this raga
        s_list = [s for s in val_songs if s['raga'] == raga_idx]
        
        chunks_collected = 0
        valid_songs = [s for s in s_list if len(s['seqs']) >= GAMAKAS_PER_CHUNK]
        
        if len(valid_songs) == 0:
            # Fallback: Pool all sequences together if no individual song is long enough
            all_seqs, all_ws = [], []
            for s in s_list:
                all_seqs.extend(s['seqs'])
                all_ws.extend(s['Ws'])
            
            while chunks_collected < EVAL_CHUNKS_PER_RAGA:
                if len(all_seqs) < GAMAKAS_PER_CHUNK:
                    # Duplicate to pad if utterly starving for data
                    all_seqs = all_seqs * (GAMAKAS_PER_CHUNK // len(all_seqs) + 2)
                    all_ws = all_ws * (GAMAKAS_PER_CHUNK // len(all_ws) + 2)
                    
                start_idx = random.randint(0, len(all_seqs) - GAMAKAS_PER_CHUNK)
                c_seqs = all_seqs[start_idx : start_idx + GAMAKAS_PER_CHUNK]
                c_ws = all_ws[start_idx : start_idx + GAMAKAS_PER_CHUNK]
                balanced_val_chunks.append((raga_idx, c_seqs, c_ws))
                chunks_collected += 1
        else:
            # Randomly slice a N-gamaka crop from an arbitrary validation song
            while chunks_collected < EVAL_CHUNKS_PER_RAGA:
                song = random.choice(valid_songs)
                start_idx = random.randint(0, len(song['seqs']) - GAMAKAS_PER_CHUNK)
                c_seqs = song['seqs'][start_idx : start_idx + GAMAKAS_PER_CHUNK]
                c_ws = song['Ws'][start_idx : start_idx + GAMAKAS_PER_CHUNK]
                balanced_val_chunks.append((raga_idx, c_seqs, c_ws))
                chunks_collected += 1
                
    # Forward Pass Evaluation
    correct_chunks = 0
    epoch_chunk_preds = []
    epoch_chunk_trues = []
    
    with torch.no_grad():
        for true_raga, c_seqs, c_ws in balanced_val_chunks:
            if USE_HIERARCHICAL:
                # Build 4D Tensor: (1, 150, 16, 2)
                X_val, W_val = build_hierarchical_tensors([c_seqs], [c_ws]) 
                seg_logits = model(X_val, W_val)
                # Hierarchical model outputs a SINGLE logit vector summarizing the entire chunk sequence
                chunk_pred = torch.argmax(seg_logits).item()
            else:
                # Build 3D Tensor: (150, 16, 2)
                X_val, W_val = build_tensors(c_seqs, c_ws)
                seg_logits = model(X_val, W_val)
                # Standard model outputs logits for EACH independent gamaka, which we must sum to aggregate
                chunk_logit_sum = torch.sum(seg_logits, dim=0)
                chunk_pred = torch.argmax(chunk_logit_sum).item()
            
            if chunk_pred == true_raga:
                correct_chunks += 1
                
            epoch_chunk_preds.append(chunk_pred)
            epoch_chunk_trues.append(true_raga)
            
    acc = correct_chunks / len(balanced_val_chunks) if len(balanced_val_chunks) > 0 else 0
    print(f"\n🏆 FINAL CHUNK ACCURACY: {acc*100:.1f}%\n")
    
    # Generate Confusion Matrix (Transposed for Actual=Columns, Predicted=Rows)
    cm = confusion_matrix(epoch_chunk_trues, epoch_chunk_preds)
    cm = cm.T  # Transpose the matrix
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=RAGAS, yticklabels=RAGAS, cmap='Blues', annot_kws={"size": 12})
    plt.title(f'Validation Confusion Matrix ({GAMAKAS_PER_CHUNK} Gamakas per Test)', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, "VocalAnnotator", f"confusion_matrix_{GAMAKAS_PER_CHUNK}_gamakas.png")
    plt.savefig(cm_path)
    print(f"📉 Confusion matrix uniquely saved as {cm_path}")

if __name__ == '__main__':
    main()
