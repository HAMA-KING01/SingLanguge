import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = r"C:\Users\hamak\Downloads\sign language\neww\working ones\tarnsformer\nmpy"
# YOUR NEW OUTPUT PATH
OUTPUT_DIR = r"C:\Users\hamak\Downloads\sign language\neww\working ones\tarnsformer\output"

ACTIONS = ["HELLO", "THANK YOU", "PLEASE"] 
MAX_SEQ_LENGTH = 30  
INPUT_DIM = 126      
BATCH_SIZE = 32
EPOCHS = 100         
LEARNING_RATE = 0.001

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# [The SignDataset and SignTransformer classes remain the same]

class SignDataset(Dataset):
    def __init__(self, sequences, labels, max_len=30):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            diff = self.max_len - len(seq)
            padding = np.zeros((diff, seq.shape[1]))
            seq = np.concatenate([seq, padding], axis=0)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class SignTransformer(nn.Module):
    def __init__(self, num_classes, input_dim, num_heads=8, num_layers=3):
        super(SignTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_SEQ_LENGTH, 128)) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1) 
        return self.fc(x)

def load_data():
    X, y = [], []
    label_map = {label: num for num, label in enumerate(ACTIONS)}
    for action in ACTIONS:
        path = os.path.join(DATA_PATH, action)
        if not os.path.exists(path): continue
        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        for npy_file in files:
            res = np.load(os.path.join(path, npy_file))
            X.append(res)
            y.append(label_map[action])
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 5. MAIN TRAINING EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    X_train, X_test, y_train, y_test = load_data()
    train_loader = DataLoader(SignDataset(X_train, y_train, MAX_SEQ_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SignDataset(X_test, y_test, MAX_SEQ_LENGTH), batch_size=BATCH_SIZE, shuffle=False)

    model = SignTransformer(num_classes=len(ACTIONS), input_dim=INPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    print("\n--- Starting Training (100 Epochs) ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {accuracy:.2f}%")

        # UPDATED SAVE PATHS USING os.path.join
        if accuracy > best_acc:
            best_acc = accuracy
            save_path = os.path.join(OUTPUT_DIR, "best_sign_transformer.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> [SAVE] New Best Accuracy reached: {best_acc:.2f}%")

    # Save final model to output directory as well
    final_path = os.path.join(OUTPUT_DIR, "final_sign_transformer.pth")
    torch.save(model.state_dict(), final_path)
    
    print(f"\nTraining Complete! Highest Val Acc: {best_acc:.2f}%")
    print(f"Files saved in: {OUTPUT_DIR}")