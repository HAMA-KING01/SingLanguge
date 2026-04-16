import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = r"C:\Users\hamak\Downloads\sign language\neww\working ones\tarnsformer\nmpy"
OUTPUT_DIR = r"C:\Users\hamak\Downloads\sign language\neww\working ones\tarnsformer\output"
ACTIONS = ["catch", "drown", "cool", "cry", "sandwich"] 

MAX_SEQ_LENGTH = 80  
INPUT_DIM = 126      
BATCH_SIZE = 16      
EPOCHS = 500         
LEARNING_RATE = 1e-4

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. UI & DIAGNOSTICS
# ==========================================
class KerasProgressBar:
    def __init__(self, target, width=25):
        self.target = target
        self.width = width
        self.start_time = time.time()

    def update(self, current, values):
        percent = current / self.target
        bar = '━' * int(self.width * percent) + '━' * (self.width - int(self.width * percent))
        elapsed = time.time() - self.start_time
        metrics = " - ".join([f"{k}: {v:.4f}" for k, v in values])
        sys.stdout.write(f"\r{current}/{self.target} {bar} {int(elapsed)}s - {metrics}")
        sys.stdout.flush()

    def finalize(self):
        sys.stdout.write("\n")
        sys.stdout.flush()

def save_confusion_matrix(model, test_loader, actions, output_path, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for seqs, lbls in test_loader:
            seqs = seqs.to(device)
            out = model(seqs)
            _, predicted = torch.max(out, 1)
            y_true.extend(lbls.numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=actions, yticklabels=actions, cmap='Greens')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Final Project: Confusion Matrix')
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    print(f"\n[DIAGNOSTIC] Confusion Matrix saved to {output_path}")

# ==========================================
# 3. NORMALIZED DATASET
# ==========================================
class SignDataset(Dataset):
    def __init__(self, seqs, lbls, augment=False): 
        self.seqs = seqs
        self.lbls = lbls
        self.augment = augment

    def __len__(self): return len(self.lbls)

    def __getitem__(self, idx):
        s = self.seqs[idx].copy()
        
        # --- WRIST NORMALIZATION ---
        # Centering every landmark relative to the wrist (0 and 63)
        for hand_offset in [0, 63]:
            wrist_x, wrist_y, wrist_z = s[:, hand_offset], s[:, hand_offset+1], s[:, hand_offset+2]
            for i in range(21):
                s[:, hand_offset + i*3] -= wrist_x
                s[:, hand_offset + i*3 + 1] -= wrist_y
                s[:, hand_offset + i*3 + 2] -= wrist_z

        if self.augment:
            # Spatial Jitter (Noise)
            s += np.random.normal(0, 0.002, s.shape)
            # Temporal Subsampling (Speed variation)
            if len(s) > 40 and np.random.random() > 0.5:
                idx_step = np.random.choice([1, 2])
                s = s[::idx_step]

        if len(s) > MAX_SEQ_LENGTH: s = s[:MAX_SEQ_LENGTH]
        else: s = np.concatenate([s, np.zeros((MAX_SEQ_LENGTH - len(s), s.shape[1]))], axis=0)
        
        return torch.tensor(s, dtype=torch.float32), torch.tensor(self.lbls[idx], dtype=torch.long)

# ==========================================
# 4. LEAN TRANSFORMER ARCHITECTURE
# ==========================================
class SignTransformer(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(SignTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(MAX_SEQ_LENGTH)
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_SEQ_LENGTH, 128))
        
        # 2 layers is the 'Goldilocks' zone for small datasets (136 samples)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.5, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x); x = self.bn1(x); x = x + self.pos_encoder
        x = self.transformer(x); x = x.mean(dim=1); return self.fc(x)

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
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test = load_data()
    
    train_loader = DataLoader(SignDataset(X_train, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SignDataset(X_test, y_test, augment=False), batch_size=BATCH_SIZE, shuffle=False)

    model = SignTransformer(len(ACTIONS), INPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    
    best_acc, patience = 0.0, 0

    for epoch in range(EPOCHS):
        show = (epoch + 1) % 5 == 0 or epoch == 0
        if show:
            print(f"Epoch {epoch+1}/{EPOCHS}")
            pbar = KerasProgressBar(target=len(train_loader))
        
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for i, (seqs, lbls) in enumerate(train_loader):
            seqs, lbls = seqs.to(device), lbls.to(device)
            optimizer.zero_grad(); out = model(seqs); loss = criterion(out, lbls)
            loss.backward(); optimizer.step()
            
            t_loss += loss.item(); _, pred = torch.max(out.data, 1)
            t_total += lbls.size(0); t_correct += (pred == lbls).sum().item()
            if show: pbar.update(i + 1, [("loss", t_loss/(i+1)), ("acc", t_correct/t_total)])

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for seqs, lbls in test_loader:
                seqs, lbls = seqs.to(device), lbls.to(device); out = model(seqs)
                _, pred = torch.max(out.data, 1); v_total += lbls.size(0); v_correct += (pred == lbls).sum().item()
        
        v_acc = v_correct / v_total; scheduler.step(v_acc)
        if v_acc > best_acc:
            best_acc, patience = v_acc, 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_sign_transformer.pth"))
        else: patience += 1

        if show:
            pbar.finalize(); print(f"val_acc: {v_acc:.4f} - lr: {optimizer.param_groups[0]['lr']:.1e}\n")
        if patience > 100: break

    # Finish with diagnostics
    best_model = SignTransformer(len(ACTIONS), INPUT_DIM).to(device)
    best_model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_sign_transformer.pth"), weights_only=True))
    save_confusion_matrix(best_model, test_loader, ACTIONS, OUTPUT_DIR, device)
    print(f"Final Best Validation Accuracy: {best_acc*100:.2f}%")