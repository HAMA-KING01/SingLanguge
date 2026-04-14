import cv2
import mediapipe as mp
import numpy as np
import os

# 1. Setup MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

# --- CONFIGURATION ---
DATA_PATH = r"C:\Users\hamak\Downloads\sign language\neww\working ones\tarnsformer\nmpy" # Output folder
VIDEO_PATH = r"U:\Projects\sing Languge data\Organized_Classes"        # Where your .mp4 files are
ACTIONS = ["HELLO", "THANK YOU", "PLEASE"] # Your sign categories
# ---------------------

def extract_landmarks(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh]) # Total 126 features (63 per hand)

# Create folders if they don't exist
for action in ACTIONS:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

# Loop through every video in your video folder
for action in ACTIONS:
    video_folder = os.path.join(VIDEO_PATH, action)
    for video_file in os.listdir(video_folder):
        cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
        landmarks_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Convert to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # Extract and append
            landmarks_sequence.append(extract_landmarks(results))
            
        cap.release()
        
        # Save the sequence as a .npy file
        file_name = video_file.split('.')[0] # Remove .mp4 extension
        save_path = os.path.join(DATA_PATH, action, file_name)
        np.save(save_path, np.array(landmarks_sequence))
        print(f"Processed: {action} - {video_file}")

print("All videos converted to landmark data!")