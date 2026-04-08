import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
from scipy.signal import savgol_filter
import collections

# --- CONFIGURATION ---
MODEL_PATH = 'sign_language_model_fold_1.keras'
ACTIONS_PATH = 'actions.npy'
THRESHOLD = 0.40  # Lowered to make the AI more confident in the real world
SMOOTHING_FRAMES = 15 # Averages the last 15 frames to stop text flickering

try:
    model = load_model(MODEL_PATH)
    actions = np.load(ACTIONS_PATH)
    print(f"✅ Model Loaded. Recognizing {len(actions)} classes.")
except Exception as e:
    print(f"❌ Error: {e}"); exit()

mp_holistic, mp_drawing = mp.solutions.holistic, mp.solutions.drawing_utils

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1)) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# --- CAMERA SETUP ---
cap = cv2.VideoCapture(0) # Built-in laptop webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

sequence, sentence = [], []
predictions_queue = collections.deque(maxlen=SMOOTHING_FRAMES) 

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        draw_styled_landmarks(image, results)
        
        sequence.append(extract_keypoints(results))
        sequence = sequence[-30:] 

        if len(sequence) == 30:
            try:
                # Signal Processing matching training data
                data = savgol_filter(np.array(sequence), window_length=5, polyorder=3, axis=0)
                mean, std = np.mean(data, axis=0), np.std(data, axis=0)
                std[std == 0] = 1.0 
                input_data = np.expand_dims((data - mean) / std, axis=0)
                
                # Prediction
                res = model.predict(input_data, verbose=0)[0]
                predictions_queue.append(res)
                
                # Averaging
                avg_res = np.mean(predictions_queue, axis=0)
                best_idx, confidence = np.argmax(avg_res), np.max(avg_res)

                if confidence > THRESHOLD:
                    if len(sentence) > 0:
                        if actions[best_idx] != sentence[-1]:
                            sentence.append(actions[best_idx])
                    else:
                        sentence.append(actions[best_idx])

                if len(sentence) > 5: sentence = sentence[-5:]

                # UI Graphics
                bar_color = (0, 0, 255) 
                if confidence > 0.5: bar_color = (0, 255, 255) 
                if confidence > 0.8: bar_color = (0, 255, 0)   
                
                cv2.rectangle(image, (0, 60), (300, 110), (0,0,0), -1)
                cv2.putText(image, f"{actions[best_idx]} ({confidence*100:.0f}%)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, bar_color, 2, cv2.LINE_AA)
                cv2.rectangle(image, (0, 110), (int(confidence * 300), 115), bar_color, -1)
            except Exception:
                pass

        cv2.rectangle(image, (0,0), (1280, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Sign Language AI Prototype', image)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()