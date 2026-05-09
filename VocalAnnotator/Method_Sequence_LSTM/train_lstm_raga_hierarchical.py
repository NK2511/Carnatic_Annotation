import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from carnatic_functions import extract_melodic_sequence

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = r"C:\Desktop\Python\CarnaticAnnotater"
ALL_RAGAS = ["Kalyani", "Keeravani", "Kharaharapriya", "Mayamalavagowlai", "Panthuvarali", "Shankarabharanam", "Shanmukhapriya", "Thodi"]

RAGAS = []
for r in ALL_RAGAS:
    p1 = os.path.join(BASE_DIR, "Raagas", r, f"{r}_Data", f"carva_{r}.csv")
    p2 = os.path.join(BASE_DIR, "Raagas", r, f"{r}_Data", "carva.csv")
    if os.path.exists(p1) or os.path.exists(p2):
        RAGAS.append(r)
        
NUM_CLASSES = len(RAGAS)
MAX_SEQ_LEN = 8  

# HIERARCHICAL SETTINGS
GAMAKAS_PER_CHUNK = 150   # A macro-block is 50 sequential gamakas
STRIDE = 2              # Rolling window stride to extract blocks from song

BATCH_SIZE = 32
EPOCHS =500
LEARNING_RATE = 0.001

# ==========================
# SPLINE PARSING
# ==========================
def get_sequence_from_segment(y_raw):
    """Wrapper to maintain compatibility with existing training loops."""
    W = len(y_raw)
    seq = extract_melodic_sequence(y_raw, target_points=MAX_SEQ_LEN, target_frames=60)
    return seq, W

def load_data_by_song():
    songs_data = []
    for raga_id, raga in enumerate(RAGAS):
        carva_path = os.path.join(BASE_DIR, "Raagas", raga, f"{raga}_Data", f"carva_{raga}.csv")
        if not os.path.exists(carva_path):
            carva_path = os.path.join(BASE_DIR, "Raagas", raga, f"{raga}_Data", "carva.csv")
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
            # We must load songs regardless of size so that the pooling function can stitch short songs 
            # together to achieve massive (250+) gamaka horizons!
            if len(seqs) > 10: 
                songs_data.append({'raga': raga_id, 'song_id': f"{raga}_{song_idx}", 'seqs': seqs, 'Ws': ws})
    return songs_data

def build_hierarchical_tensors(chunk_list, w_list):
    n = len(chunk_list)
    X = np.zeros((n, GAMAKAS_PER_CHUNK, MAX_SEQ_LEN, 2))
    X[:, :, :, 0] = 12 # Pad index
    W_arr = np.zeros((n, GAMAKAS_PER_CHUNK, 1))
    
    for i in range(n):
        gamakas = chunk_list[i]
        for g_idx in range(len(gamakas)):
            gamaka = gamakas[g_idx]
            act_len = min(len(gamaka), MAX_SEQ_LEN)
            for j in range(act_len):
                X[i, g_idx, j, 0] = gamaka[j][0] # Note
                X[i, g_idx, j, 1] = gamaka[j][1] # Delta T
            W_arr[i, g_idx, 0] = w_list[i][g_idx]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(W_arr, dtype=torch.float32)

# ==========================
# HIERARCHICAL MODEL
# ==========================
class HierarchicalRagaLSTM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(HierarchicalRagaLSTM, self).__init__()
        self.note_embedding = nn.Embedding(num_embeddings=13, embedding_dim=16, padding_idx=12)
        
        # Level 1: Micro LSTM (Analyzes points inside a single Gamaka)
        self.micro_lstm = nn.LSTM(input_size=17, hidden_size=32, num_layers=1, batch_first=True)
        
        # Level 2: Macro LSTM (Analyzes transitions across 50 Gamakas)
        self.macro_lstm = nn.LSTM(input_size=33, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x_seq, w_dur):
        B, num_gamakas, num_nodes, feats = x_seq.size()
        
        # Flatten to process all gamakas independently first
        x_flat = x_seq.view(B * num_gamakas, num_nodes, feats)
        
        note_ids = x_flat[:, :, 0].long()
        delta_ts = x_flat[:, :, 1].unsqueeze(-1)
        
        embedded_notes = self.note_embedding(note_ids)
        micro_input = torch.cat((embedded_notes, delta_ts), dim=2)
        
        _, (h_n, _) = self.micro_lstm(micro_input)
        gamaka_vectors = h_n[-1] # (B * num_gamakas, 32)
        
        # Reshape back to sequence of gamakas
        gamaka_vectors = gamaka_vectors.view(B, num_gamakas, 32)
        
        # Contextualize each Gamaka vector with its original time duration
        macro_input = torch.cat((gamaka_vectors, w_dur), dim=2) # (B, num_gamakas, 33)
        
        # Find transition patterns across the melody phrase
        macro_out, (h_n_macro, _) = self.macro_lstm(macro_input)
        final_macro_h = h_n_macro[-1]
        
        out = self.dropout(self.relu(self.fc1(final_macro_h)))
        out = self.fc2(out)
        return out

# ==========================
# TRAINING LOOP
# ==========================
def main():
    print("🚀 Extracting Sequences for Hierarchical Network...")
    songs_data = load_data_by_song()
    random.seed(42)
    random.shuffle(songs_data)
    
    # 1. SPLIT BY SONG
    train_songs, val_songs = [], []
    raga_groups = {i: [] for i in range(NUM_CLASSES)}
    for s in songs_data: raga_groups[s['raga']].append(s)
        
    for r_id, s_list in raga_groups.items():
        split_idx = int(0.8 * len(s_list))
        train_songs.extend(s_list[:split_idx])
        val_songs.extend(s_list[split_idx:])
        
    # 2. CREATE HIERARCHICAL BLOCKS
    train_X_blocks, train_W_blocks, train_y_blocks = [], [], []
    raga_counts = {i: 0 for i in range(NUM_CLASSES)}
    
    for s in train_songs:
        r_id = s['raga']
        seqs, ws = s['seqs'], s['Ws']
        # Rolling window to extract overlapping chunks of gamakas
        for i in range(0, len(seqs) - GAMAKAS_PER_CHUNK + 1, STRIDE):
            train_X_blocks.append(seqs[i:i+GAMAKAS_PER_CHUNK])
            train_W_blocks.append(ws[i:i+GAMAKAS_PER_CHUNK])
            train_y_blocks.append(r_id)
            raga_counts[r_id] += 1
            
    # CRITICAL FIX: If 250 gamakas is requested, some Ragas' songs might frankly be too short!
    # They collapse to 0 counts. We must pool their songs together seamlessly to prevent crashing!
    for r_id in range(NUM_CLASSES):
        if raga_counts[r_id] < 100:  
            print(f"⚠️ Raga {r_id} is extremely starved for 250-length continuous songs. Pooling sequences...")
            pool_seq, pool_w = [], []
            for s in train_songs:
                if s['raga'] == r_id:
                    pool_seq.extend(s['seqs'])
                    pool_w.extend(s['Ws'])
            
            # If still violently too short (e.g. less than 250 total gamakas in the entire Raga), pad it by looping!
            if len(pool_seq) > 0:
                while len(pool_seq) <= GAMAKAS_PER_CHUNK + 10:
                    pool_seq = pool_seq * 2
                    pool_w = pool_w * 2
                    
                # Extract pooled chunks
                for i in range(0, len(pool_seq) - GAMAKAS_PER_CHUNK + 1, STRIDE):
                    train_X_blocks.append(pool_seq[i:i+GAMAKAS_PER_CHUNK])
                    train_W_blocks.append(pool_w[i:i+GAMAKAS_PER_CHUNK])
                    train_y_blocks.append(r_id)
                    raga_counts[r_id] += 1
                    
    print("Macro Blocks per Raga before balancing:", raga_counts)
    
    # Balance Blocks
    min_blocks = min(raga_counts.values()) if raga_counts else 0
    balanced_X, balanced_W, balanced_y = [], [], []
    added_counts = {i: 0 for i in range(NUM_CLASSES)}
    
    combined = list(zip(train_X_blocks, train_W_blocks, train_y_blocks))
    random.shuffle(combined)
    for x, w, y in combined:
        if added_counts[y] < min_blocks:
            balanced_X.append(x)
            balanced_W.append(w)
            balanced_y.append(y)
            added_counts[y] += 1
            
    # VALIDATION BLOCKS
    val_X_blocks, val_W_blocks, val_y_blocks = [], [], []
    for s in val_songs:
        r_id, seqs, ws = s['raga'], s['seqs'], s['Ws']
        # Jump by full chunk size for clean independent tests
        for i in range(0, len(seqs) - GAMAKAS_PER_CHUNK + 1, GAMAKAS_PER_CHUNK):
            val_X_blocks.append(seqs[i:i+GAMAKAS_PER_CHUNK])
            val_W_blocks.append(ws[i:i+GAMAKAS_PER_CHUNK])
            val_y_blocks.append(r_id)
            
    X_train_t, W_train_t = build_hierarchical_tensors(balanced_X, balanced_W)
    y_train_t = torch.tensor(balanced_y, dtype=torch.long)
    
    X_val_t, W_val_t = build_hierarchical_tensors(val_X_blocks, val_W_blocks)
    y_val_t = torch.tensor(val_y_blocks, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_t, W_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(X_val_t, W_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 3. TRAIN
    model = HierarchicalRagaLSTM(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n🚀 Commencing Hierarchical LSTM Training ({GAMAKAS_PER_CHUNK}-Gamaka Syntax)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for b_x, b_w, b_y in train_loader:
            optimizer.zero_grad()
            out = model(b_x, b_w)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for b_x, b_w, b_y in val_loader:
                out = model(b_x, b_w)
                preds = torch.argmax(out, dim=1)
                total += b_y.size(0)
                correct += (preds == b_y).sum().item()
                
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | Validation Chunk Acc: {(correct/total)*100:.1f}%")
            
    # Save Model
    model_path = os.path.join(BASE_DIR, "VocalAnnotator", "raga_hierarchical_lstm_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Training Complete. Saved to raga_hierarchical_lstm_model.pth")

if __name__ == '__main__':
    main()
