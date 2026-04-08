import cv2
import numpy as np
import os
import mediapipe as mp

# --- CONFIGURATION ---
# PATH TO YOUR ORGANIZED VIDEOS
RAW_DATA_PATH = r"C:\Users\hamak\Desktop\datset\American-Sign-Language-Dataset\Organized_Data"
# PATH TO SAVE NUMBERS
EXPORT_PATH = os.path.join(r"C:\Users\hamak\Downloads\sign language\neww\npy") 

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468*3)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def process_videos():
    if not os.path.exists(EXPORT_PATH): os.makedirs(EXPORT_PATH)
    
  
    # world class
    # --- 20 High-Quality Classes ---
    actions_to_process = [
        "A", "about", "ABOVE", "ACCENT", "ACCEPT", 
        "ACCIDENT", "ACQUIRE", "ACTION", "ACTOR", "ADD", 
        "ADDRESS", "adjust", "ADULT", "AFTER", "AFTERNOON", 
        "AGE", "AIRPLANE", "ALARM", "ALL", "AGAIN"
    ]
    
    print(f"Processing ONLY these {len(actions_to_process)} elite actions.")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for action in actions_to_process:
            action_path = os.path.join(RAW_DATA_PATH, action)
            target_dir = os.path.join(EXPORT_PATH, action)
            os.makedirs(target_dir, exist_ok=True)
            
            videos = os.listdir(action_path)
            print(f"--> {action}: Found {len(videos)} raw videos.")
            
            processed_count = 0
            for video_name in videos:
                if not video_name.lower().endswith('.mp4'): continue
                
                video_path = os.path.join(action_path, video_name)
                save_path = os.path.join(target_dir, os.path.splitext(video_name)[0] + ".npy")
                
                cap = cv2.VideoCapture(video_path)
                
                # ---  EVENLY SPACED FRAMES ---
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames >= 30:
                    # Pick 30 frames evenly spread across the whole video
                    target_indices = np.linspace(0, total_frames - 1, 30).astype(int)
                else:
                    # Video is short, grab all frames available
                    target_indices = np.arange(total_frames)

                frames_sequence = []
                current_frame = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Only process the frame if it's one of our target evenly-spaced frames
                    if current_frame in target_indices:
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = holistic.process(image)
                        
                        keypoints = extract_keypoints(results)
                        frames_sequence.append(keypoints)
                    
                    current_frame += 1
                    
                    # Stop if we hit 30 frames
                    if len(frames_sequence) == 30: break
                
                cap.release()
                
                # ---  PADDING WITH FREEZE FRAME ---
                if len(frames_sequence) > 0:
                    while len(frames_sequence) < 30:
                        
                        frames_sequence.append(frames_sequence[-1])
                    
                    # Save the strict (30, 1662) array
                    np.save(save_path, np.array(frames_sequence))
                    processed_count += 1
                else:
                    print(f"    [!] Skipping {video_name}: No readable frames.")
            
            print(f"    Saved {processed_count} .npy files.")

if __name__ == "__main__":
    process_videos()