import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# CONFIG
MODEL_PATH = r"C:\Users\hamak\Downloads\sign language\neww\working ones\tarnsformer\output\best_sign_transformer.pth"
ACTIONS = ["catch", "drown", "cool", "cry", "sandwich"] 
MAX_SEQ_LENGTH = 80  
INPUT_DIM = 126

class SignTransformer(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(SignTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(MAX_SEQ_LENGTH)
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_SEQ_LENGTH, 128))
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.embedding(x); x = self.bn1(x); x = x + self.pos_encoder
        x = self.transformer(x); x = x.mean(dim=1); return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignTransformer(len(ACTIONS), INPUT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_landmarks(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    res = np.concatenate([lh, rh])
    
    # WRIST NORMALIZATION (Must match Training script!)
    res[0:63:3] -= res[0]; res[1:63:3] -= res[1]; res[2:63:3] -= res[2]
    res[63:126:3] -= res[63]; res[64:126:3] -= res[64]; res[65:126:3] -= res[65]
    return res

sequence, sentence = [], []
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic.process(image); image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if res.left_hand_landmarks or res.right_hand_landmarks:
        sequence.append(extract_landmarks(res))
        sequence = sequence[-MAX_SEQ_LENGTH:]
        if len(sequence) == MAX_SEQ_LENGTH:
            with torch.no_grad():
                out = model(torch.tensor([sequence], dtype=torch.float32).to(device))
                conf, idx = torch.max(torch.softmax(out, dim=1)[0], 0)
                if conf.item() > 0.85:
                    action = ACTIONS[idx.item()]
                    if not sentence or action != sentence[-1]:
                        sentence.append(action)
    
    cv2.rectangle(image, (0,0), (640, 45), (245, 117, 16), -1)
    cv2.putText(image, ' | '.join(sentence[-5:]), (10,33), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Real-Time Demo', image)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()