import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization, Activation, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from scipy.signal import savgol_filter

# --- CONFIGURATION ---
DATA_PATH = os.path.join(r"C:\Users\hamak\Downloads\sign language\neww\npy")   
MODEL_NAME = 'sign_language_model_20_classes.keras'

def load_and_process_data():
    sequences, labels = [], []
    actions = []
    
    if not os.path.exists(DATA_PATH):
        return None, None, None

    actions = np.array(os.listdir(DATA_PATH))
    label_map = {label:num for num, label in enumerate(actions)}
    
    print(f"Loading {len(actions)} Classes...") 

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.isdir(action_path): continue
        
        files = os.listdir(action_path)
        for file_name in files:
            if file_name.endswith('.npy'):
                file_path = os.path.join(action_path, file_name)
                try:
                    data = np.load(file_path)
                    # Ensure exactly 30 frames and 1662 raw keypoints
                    if data.shape == (30, 1662):
                        pose = data[:, :132]
                        hands = data[:, 1536:]
                        clean_data = np.concatenate([pose, hands], axis=1) 
                        
                        # Apply Savitzky-Golay filter for smoothing
                        clean_data = savgol_filter(clean_data, window_length=5, polyorder=3, axis=0)
                        
                        # Standardize (Z-score normalization)
                        mean = np.mean(clean_data, axis=0)
                        std = np.std(clean_data, axis=0)
                        std[std == 0] = 1.0 
                        clean_data = (clean_data - mean) / std

                        sequences.append(clean_data)
                        labels.append(label_map[action])
                except Exception:
                    pass

    return np.array(sequences), np.array(labels), actions

if __name__ == "__main__":
    X, y, actions = load_and_process_data()
    if X is None or len(X) == 0: 
        print("No data found. Please check DATA_PATH.")
        exit()
    
    # Save the class names for the real-time script
    np.save('actions.npy', actions)

    # Force a strict 15% test set to get a more accurate validation score
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # Handle class imbalances
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=len(actions))
    y_test = to_categorical(y_test, num_classes=len(actions))

    # --- MODEL ARCHITECTURE ---
    model = Sequential()
    
    # Spatial Feature Extraction
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', input_shape=(30, 258)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Temporal Sequence Learning with heavy Dropout (0.5) and L2 Regularization
    model.add(Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(0.005))))
    model.add(Dropout(0.5)) 
    model.add(BatchNormalization())

    model.add(Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2(0.005))))
    model.add(Dropout(0.5)) 
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling1D())

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.5))

    model.add(Dense(len(actions), activation='softmax'))

    # Compile with a slow learning rate for careful convergence
    opt = tf.keras.optimizers.Adam(learning_rate=0.00005) 
    
    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy', TopKCategoricalAccuracy(k=5, name='top_5_acc')])

    callbacks = [
        EarlyStopping(monitor='val_categorical_accuracy', patience=40, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=8, min_lr=0.000001),
        ModelCheckpoint(MODEL_NAME, monitor='val_categorical_accuracy', save_best_only=True)
    ]

    print(f"\nStarting Training on {len(X_train)} pure baseline samples...")
    model.fit(X_train, y_train, epochs=600, batch_size=32, validation_data=(X_test, y_test),
              callbacks=callbacks, class_weight=class_weights_dict)