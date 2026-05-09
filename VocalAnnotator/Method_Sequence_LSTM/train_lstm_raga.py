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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from carnatic_functions import extract_melodic_sequence

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = r"C:\Desktop\Python\CarnaticAnnotater"
ALL_RAGAS = ["Kalyani", "Keeravani", "Kharaharapriya", "Mayamalavagowlai", "Panthuvarali", "Shankarabharanam", "Shanmukhapriya", "Thodi"]

# Only keep ragas that actually have a carva file to prevent 0-count balancing errors
RAGAS = []
for r in ALL_RAGAS:
    p1 = os.path.join(BASE_DIR, r, f"{r}_Data", f"carva_{r}.csv")
    p2 = os.path.join(BASE_DIR, r, f"{r}_Data", "carva.csv")
    if os.path.exists(p1) or os.path.exists(p2):
        RAGAS.append(r)
        
raga_to_id = {r: i for i, r in enumerate(RAGAS)}
NUM_CLASSES = len(RAGAS)

BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 8  

# ==========================
# SPLINE & SEQUENCE ENCODING
# ==========================
def get_sequence_from_segment(y_raw):
    """Wrapper to maintain compatibility with existing training loops."""
    W = len(y_raw)
    seq = extract_melodic_sequence(y_raw, target_points=MAX_SEQ_LEN, target_frames=60)
    return seq, W

# ==========================
# DATA LOADING: SONG-LEVEL
# ==========================
def load_data_by_song():
    print("🚀 Extracting Sequences (Grouped by Song to prevent data leak)...")
    songs_data = [] # List of DB objects: {raga, song_id, seqs, Ws}
    
    for raga in RAGAS:
        raga_id = raga_to_id[raga]
        carva_path = os.path.join(BASE_DIR, raga, f"{raga}_Data", f"carva_{raga}.csv")
        if not os.path.exists(carva_path):
            carva_path = os.path.join(BASE_DIR, raga, f"{raga}_Data", "carva.csv")
            if not os.path.exists(carva_path): continue
                
        df = pd.read_csv(carva_path)
        if 'Primary_Label' in df.columns:
            df = df[df['Primary_Label'] != -1]
            
        # Group by song index
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
            
            if len(seqs) > 10: # Only keep songs with at least 10 segments
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
    X[:, :, 0] = 12 # Padding Note
    W_arr = np.zeros((n, 1))
    
    for i in range(n):
        seq = seq_list[i]
        act_len = min(len(seq), MAX_SEQ_LEN)
        for j in range(act_len):
            X[i, j, 0] = seq[j][0]
            X[i, j, 1] = seq[j][1]
        W_arr[i, 0] = w_list[i]
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(W_arr, dtype=torch.float32)

# ==========================
# LSTM MODEL
# ==========================
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

# ==========================
# TRAINING & EVALUATION
# ==========================
def main():
    songs_data = load_data_by_song()
    random.seed(42)
    random.shuffle(songs_data)
    
    # 1. SPLIT BY SONG (80% Train, 20% Val per Raga)
    train_songs, val_songs = [], []
    raga_groups = {i: [] for i in range(NUM_CLASSES)}
    for s in songs_data:
        raga_groups[s['raga']].append(s)
        
    for r_id, s_list in raga_groups.items():
        # Force a strict 80/20 slice per raga
        split_idx = int(0.8 * len(s_list))
        train_songs.extend(s_list[:split_idx])
        val_songs.extend(s_list[split_idx:])
        
    print(f"\n📊 DATASET SPLIT EXPLANATION:")
    print(f"   • We group all sequences back into their original songs so the network doesn't memorize singers.")
    print(f"   • We then take {len(train_songs)} full songs (80%) and extract their sequences strictly for TRAINING.")
    print(f"   • We hold out {len(val_songs)} completely unseen songs (20%) strictly for VALIDATION.")

    # 1.5. PREPARE 1-MIN VALIDATION CHUNKS (AND BALANCE THEM)
    val_chunks_dict = {i: [] for i in range(NUM_CLASSES)}
    CHUNK_SIZE = 30 # roughly 30 gamakas = ~1 minute of active singing
    
    for s in val_songs:
        r_id = s['raga']
        seqs = s['seqs']
        ws = s['Ws']
        for i in range(0, len(seqs), CHUNK_SIZE):
            c_seqs = seqs[i:i+CHUNK_SIZE]
            c_ws = ws[i:i+CHUNK_SIZE]
            if len(c_seqs) >= 5:
                val_chunks_dict[r_id].append((c_seqs, c_ws))
                
    min_val_chunks = min(len(chunks) for chunks in val_chunks_dict.values()) if val_chunks_dict else 0
    print(f"\n   • Validation chunking resulted in an unbalanced set. Balancing to {min_val_chunks} '1-min clips' per raga.")
    print(f"   • The Confusion Matrix will rigorously evaluate exactly {min_val_chunks * NUM_CLASSES} balanced validation clips.")
    
    balanced_val_chunks = []
    for r_id, chunks in val_chunks_dict.items():
        if len(chunks) > 0:
            # Randomly undersample so every Raga gets the exact same number of test cases
            samp = random.sample(chunks, min_val_chunks)
            for c in samp:
                balanced_val_chunks.append((r_id, c[0], c[1]))
    
    random.shuffle(balanced_val_chunks)
    
    # 2. FLATTEN & BALANCE TRAIN SEGMENTS
    train_X_list, train_W_list, train_y_list = [], [], []
    train_raga_counts = {i: 0 for i in range(NUM_CLASSES)}
    
    # Collect all train segments
    for s in train_songs:
        r_id = s['raga']
        for i in range(len(s['seqs'])):
            train_X_list.append(s['seqs'][i])
            train_W_list.append(s['Ws'][i])
            train_y_list.append(r_id)
            train_raga_counts[r_id] += 1
            
    print("Train Segments per Raga before balancing:", train_raga_counts)
    
    # Balance Train Data
    min_train_seg = min(train_raga_counts.values()) if train_raga_counts else 0
    balanced_X, balanced_W, balanced_y = [], [], []
    added_counts = {i: 0 for i in range(NUM_CLASSES)}
    
    # Randomly sample to match min_train_seg
    combined_train = list(zip(train_X_list, train_W_list, train_y_list))
    random.shuffle(combined_train)
    
    for x, w, y in combined_train:
        if added_counts[y] < min_train_seg:
            balanced_X.append(x)
            balanced_W.append(w)
            balanced_y.append(y)
            added_counts[y] += 1
            
    X_train_t, W_train_t = build_tensors(balanced_X, balanced_W)
    y_train_t = torch.tensor(balanced_y, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_t, W_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. SETUP MODEL
    model = RagaLSTM(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses, seg_accs, song_accs = [], [], []
    
    print(f"\n🚀 Commencing LSTM Training (Target: {NUM_CLASSES}-Way Classification)...")
    print("Metrics will report BOTH Segment-Level votes and Final Song-Level Assembly.\n")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for b_seq, b_w, b_y in train_loader:
            optimizer.zero_grad()
            out = model(b_seq, b_w)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 4. EVALUATE ON BALANCED VAL CHUNKS (Simulating 1-min clips)
        model.eval()
        correct_segments = 0
        total_segments = 0
        correct_chunks = 0
        
        epoch_chunk_preds = []
        epoch_chunk_trues = []
        
        with torch.no_grad():
            for true_raga, c_seqs, c_ws in balanced_val_chunks:
                X_val, W_val = build_tensors(c_seqs, c_ws)
                
                # Forward Pass
                seg_logits = model(X_val, W_val)
                seg_preds = torch.argmax(seg_logits, dim=1)
                
                # Segment Tracking
                total_segments += len(seg_preds)
                correct_segments += (seg_preds == true_raga).sum().item()
                
                # Chunk-Level Majority Vote
                chunk_logit_sum = torch.sum(seg_logits, dim=0)
                chunk_pred = torch.argmax(chunk_logit_sum).item()
                
                if chunk_pred == true_raga:
                    correct_chunks += 1
                    
                epoch_chunk_preds.append(chunk_pred)
                epoch_chunk_trues.append(true_raga)
                    
        seg_acc = correct_segments / total_segments if total_segments > 0 else 0
        song_acc = correct_chunks / len(balanced_val_chunks) if len(balanced_val_chunks) > 0 else 0
        
        train_losses.append(total_loss / len(train_loader))
        seg_accs.append(seg_acc)
        song_accs.append(song_acc)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {train_losses[-1]:.4f} | Segment Voting Acc: {seg_acc*100:.1f}% | 🏆 1-MIN CHUNK ACC: {song_acc*100:.1f}%")
            
    # Save Model
    model_path = os.path.join(BASE_DIR, "VocalAnnotator", "raga_lstm_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Training Complete. Model saved.")
    
    # Plot Training Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(seg_accs, label="Segment-Level Accuracy", linestyle='--')
    plt.plot(song_accs, label="Song-Level Accuracy", color='green', linewidth=2)
    plt.title("LSTM Raga Classifier Performance")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "VocalAnnotator", "lstm_training_curve.png"))
    
    # Generate Confusion Matrix for final epoch
    cm = confusion_matrix(epoch_chunk_trues, epoch_chunk_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=RAGAS, yticklabels=RAGAS, cmap='Blues', annot_kws={"size": 12})
    plt.ylabel('True Raga (1-Min Chunk Level)', fontsize=12)
    plt.xlabel('Predicted Raga (1-Min Chunk Level)', fontsize=12)
    plt.title('1-Min Chunk Validation Confusion Matrix', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "VocalAnnotator", "lstm_confusion_matrix.png"))
    print("📉 Confusion matrix saved as lstm_confusion_matrix.png")

if __name__ == '__main__':
    main()
