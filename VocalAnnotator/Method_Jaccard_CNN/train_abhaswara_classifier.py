"""
Train Abhaswara Classifier.
Using 1D CNN to distinguish between Valid Swaras (1) and Invalid Abhaswaras (0).
DATASET IS BALANCED BEFORE TRAINING (1900 vs 1900).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import sys
import os
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================
DATASET_FILE = r"C:\Desktop\Python\CarnaticAnnotater\VocalAnnotator\Method_Jaccard_CNN\abhaswara_dataset.pth"
MODEL_FILE = r"C:\Desktop\Python\CarnaticAnnotater\VocalAnnotator\Method_Jaccard_CNN\abhaswara_model.pth"
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

class StabilityCNN(nn.Module):
    def __init__(self):
        super(StabilityCNN, self).__init__()
        # Input: (B, 1, 60)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2) # -> 30
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        # -> 15
        self.fc1 = nn.Linear(32 * 15, 64)
        self.fc2 = nn.Linear(64, 1) # Logits
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    print("🚀 Training Abhaswara Classifier (Balanced)...")
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Dataset not found: {DATASET_FILE}")
        return

    # Load Data (weights_only=False due to pandas)
    data = torch.load(DATASET_FILE, weights_only=False)
    X = data['segments']
    y = data['labels']
    
    # Check loaded data
    print(f"   Original Data: {len(X)} samples.")
    if X.ndim == 2:
        X = X.unsqueeze(1) # Add channel dim (N, 1, 60)

    # Calculate Class Balance (for Weighted Loss)
    pos_indices = (y == 1).nonzero(as_tuple=True)[0]
    neg_indices = (y == 0).nonzero(as_tuple=True)[0]
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    print(f"   Swaras (1): {n_pos}, Abhaswaras (0): {n_neg}")
    
    # Calculate pos_weight for BCEWithLogitsLoss
    # We want to down-weight the majority class (1) so that minority class contributes equally
    pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"   Using Weighted Loss with pos_weight: {pos_weight_val:.4f}")
    
    # Use Full Dataset (No undersampling)
    # Variable names kept matching existing code flow
    X_balanced = X
    y_balanced = y
    
    print(f"   Full Dataset Used: {len(X_balanced)} samples")
    
    EPOCHS = 200 # User requested 200 epochs
    PATIENCE = 10
    
    # ---------------------------
    # Normalize Inputs (Scale -50..50 -> -1..1)
    # ---------------------------
    X_balanced = X_balanced / 50.0
    print("   Applied Global Normalization (X /= 50.0)")

    # Create Dataset
    dataset = TensorDataset(X_balanced, y_balanced.float().unsqueeze(1))
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    model = StabilityCNN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val))
    
    history = {'train_loss': [], 'val_acc': []}
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"🚀 Training for max {EPOCHS} epochs (Early Stopping Patience: {PATIENCE})...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
            
        # Validation
        model.eval()
        val_loss_total = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                v_loss = criterion(outputs, y_val)
                val_loss_total += v_loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        acc = correct / total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(acc)
        
        
        # Save Model Every 10 Epochs
        if (epoch + 1) % 10 == 0:
             torch.save(model.state_dict(), MODEL_FILE)
             print(f"✅ Saved Checkpoint at Epoch {epoch+1}")
             
    # Save Final
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"✅ Final Model saved to {MODEL_FILE}")
    
    # Plot
    plt.plot(history['train_loss'], label='Loss')
    plt.plot(history['val_acc'], label='Accuracy')
    plt.legend()
    plt.savefig('abhaswara_training.png')

if __name__ == "__main__":
    main()
