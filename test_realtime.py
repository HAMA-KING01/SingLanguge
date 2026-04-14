import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os

# ==========================================
# 1. CONFIGURATION (Must match your Training Script)
# ==========================================
MODEL_PATH = r"C:\Users\hamak\Downloads\sign language\neww\working ones\tarnsformer\output\best_sign_transformer.pth"
# CRITICAL: These MUST be in the same order as your training folders!
ACTIONS = ["HELLO", "THANK YOU", "PLEASE"] 
MAX_SEQ_LENGTH = 30
INPUT_DIM = 126

# ==========================================
# 2. TRANSFORMER MODEL DEFINITION
# ==========================================
class SignTransformer(nn.Module):
    def __init__(self, num_classes, input_dim, num_heads=8, num_layers=3):
        super(SignTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_SEQ_LENGTH, 128)) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1) 
        return self.fc(x)

# ==========================================
# 3. INITIALIZATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignTransformer(num_classes=len(ACTIONS), input_dim=INPUT_DIM).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print(f"--- Model Loaded Successfully on {device} ---")
else:
    print(f"ERROR: No model file found at {MODEL_PATH}")
    exit()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_landmarks(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# ==========================================
# 4. REAL-TIME INFERENCE LOOP
# ==========================================
sequence = []
sentence = []
threshold = 0.6  # Adjust this: Lower = more sensitive, Higher = more strict
current_action = "Nothing"
probs = [0.0, 0.0, 0.0]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Only process if hands are in the frame
    if results.left_hand_landmarks or results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-MAX_SEQ_LENGTH:]

        if len(sequence) == MAX_SEQ_LENGTH:
            input_data = torch.tensor([sequence], dtype=torch.float32).to(device)
            with torch.no_grad():
                res = model(input_data)
                # Convert logic to probabilities
                probabilities = torch.softmax(res, dim=1)[0] 
                conf, action_idx = torch.max(probabilities, 0)
                
                probs = probabilities.cpu().numpy()
                current_action = ACTIONS[action_idx.item()]

                # TERMINAL DEBUG: Watch these numbers!
                print(f"H: {probs[0]:.2f} | T: {probs[1]:.2f} | P: {probs[2]:.2f} -> {current_action}")

                if conf.item() > threshold:
                    if len(sentence) > 0:
                        if current_action != sentence[-1]:
                            sentence.append(current_action)
                    else:
                        sentence.append(current_action)
    else:
        # Reset if you drop your hands
        sequence = []
        current_action = "No Hands"
        probs = [0.0, 0.0, 0.0]

    if len(sentence) > 5: sentence = sentence[-5:]

    # --- ENHANCED UI ---
    # Top Bar
    cv2.rectangle(image, (0,0), (640, 45), (245, 117, 16), -1)
    cv2.putText(image, ' | '.join(sentence), (10,33), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Probability Dashboard (Bottom Left)
    cv2.putText(image, f'H: {probs[0]:.2f}', (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, f'T: {probs[1]:.2f}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, f'P: {probs[2]:.2f}', (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Prediction Status
    cv2.putText(image, f'PRED: {current_action}', (150, 430), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Transformer Sign Language Debugger', image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'): break
    if key == ord('c'): sentence = []

cap.release()
cv2.destroyAllWindows()