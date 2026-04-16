import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import time
from sklearn.model_selection import train_test_split

# ==========================================
# 1. UI 
# ==========================================
class KerasProgressBar:
    def __init__(self, target, width=20):
        self.target = target
        self.width = width
        self.start_time = time.time()

    def update(self, current, values):
        percent = current / self.target
        bar_len = int(self.width * percent)
        bar = '━' * bar_len + '━' * (self.width - bar_len)
        
        elapsed = time.time() - self.start_time
        fps = current / elapsed if elapsed > 0 else 0
        step_time = (1/fps) if fps > 0 else 0
        
        
        metrics = " - ".join([f"{k}: {v:.4f}" for k, v in values])
        
        
        sys.stdout.write(f"\r{current}/{self.target} {bar} {int(elapsed)}s {int(step_time*1000)}ms/step - {metrics}")
        sys.stdout.flush()

    def finalize(self):
        sys.stdout.write("\n")
        sys.stdout.flush()

# ==========================================
# 2. CONFIGURATION
# ==========================================
DATA_PATH = r"C:\Users\hamak\Downloads\sign language\neww\working ones\tarnsformer\nmpy"
OUTPUT_DIR = r"C:\Users\hamak\Downloads\sign language\neww\working ones\tarnsformer\output"
ACTIONS = ["catch", "drown", "cool", "cry", "sandwich"] 

MAX_SEQ_LENGTH = 80  
INPUT_DIM = 126      
BATCH_SIZE = 16      
EPOCHS = 400         
LEARNING_RATE = 1e-4

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# ==========================================
# 3. MODEL & DATASET
# ==========================================
class SignTransformer(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(SignTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(MAX_SEQ_LENGTH)
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_SEQ_LENGTH, 256))
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.embedding(x); x = self.bn1(x); x = x + self.pos_encoder
        x = self.transformer(x); x = x.mean(dim=1); return self.fc(x)

class SignDataset(Dataset):
    def __init__(self, seqs, lbls): self.seqs = seqs; self.lbls = lbls
    def __len__(self): return len(self.lbls)
    def __getitem__(self, idx):
        s = self.seqs[idx]
        if len(s) > MAX_SEQ_LENGTH: s = s[:MAX_SEQ_LENGTH]
        else: s = np.concatenate([s, np.zeros((MAX_SEQ_LENGTH - len(s), s.shape[1]))], axis=0)
        return torch.tensor(s, dtype=torch.float32), torch.tensor(self.lbls[idx], dtype=torch.long)

def load_data():
    X, y = [], []
    label_map = {label: i for i, label in enumerate(ACTIONS)}
    counts = [len([f for f in os.listdir(os.path.join(DATA_PATH, a)) if f.endswith('.npy')]) for a in ACTIONS]
    min_samples = min(counts)
    print(f"--- Dataset Balanced to {min_samples} samples per class ---")
    for action in ACTIONS:
        path = os.path.join(DATA_PATH, action)
        files = [f for f in os.listdir(path) if f.endswith('.npy')][:min_samples]
        for f in files:
            res = np.load(os.path.join(path, f))
            if res.shape[0] > 0: X.append(res); y.append(label_map[action])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test = load_data()
    
    train_loader = DataLoader(SignDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SignDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = SignTransformer(len(ACTIONS), INPUT_DIM).to(device)
    # Using Label Smoothing to help generalized accuracy
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        pbar = KerasProgressBar(target=len(train_loader))
        
        model.train()
        t_correct, t_total, t_loss = 0, 0, 0
        for i, (seqs, lbls) in enumerate(train_loader):
            seqs, lbls = seqs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(seqs)
            loss = criterion(out, lbls)
            loss.backward(); optimizer.step()
            
            t_loss += loss.item()
            _, pred = torch.max(out.data, 1)
            t_total += lbls.size(0); t_correct += (pred == lbls).sum().item()
            
            pbar.update(i + 1, [("loss", t_loss/(i+1)), ("categorical_accuracy", t_correct/t_total)])

        # Validation
        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0
        with torch.no_grad():
            for seqs, lbls in test_loader:
                seqs, lbls = seqs.to(device), lbls.to(device)
                out = model(seqs)
                v_loss += criterion(out, lbls).item()
                _, pred = torch.max(out.data, 1)
                v_total += lbls.size(0); v_correct += (pred == lbls).sum().item()
        
        v_acc = v_correct / v_total
        pbar.finalize()
        print(f"val_categorical_accuracy: {v_acc:.4f} - val_loss: {v_loss/len(test_loader):.4f} - learning_rate: {LEARNING_RATE:.1e}\n")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_sign_transformer.pth"))

    print(f"Training Complete! Best Accuracy: {best_acc*100:.2f}%")