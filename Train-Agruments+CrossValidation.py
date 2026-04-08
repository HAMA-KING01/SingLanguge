import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization, Activation, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from scipy.signal import savgol_filter, resample

# --- CONFIGURATION ---
DATA_PATH = os.path.join(r"C:\Users\hamak\Downloads\sign language\neww\npy")   
USE_JITTER = False
USE_SPEED = False    
USE_SHIFT = False

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
                    if data.shape == (30, 1662):
                        pose = data[:, :132]
                        hands = data[:, 1536:]
                        clean_data = np.concatenate([pose, hands], axis=1) 
                        
                        clean_data = savgol_filter(clean_data, window_length=5, polyorder=3, axis=0)
                        
                        mean = np.mean(clean_data, axis=0)
                        std = np.std(clean_data, axis=0)
                        std[std == 0] = 1.0 
                        clean_data = (clean_data - mean) / std

                        sequences.append(clean_data)
                        labels.append(label_map[action])
                except Exception:
                    pass

    return np.array(sequences), np.array(labels), actions

def augment_sequences(sequences, labels):
    """Applies augmentation ONLY to the training fold to prevent Data Leakage."""
    aug_seq, aug_lbl = [], []
    
    for seq, label in zip(sequences, labels):
        # 1. ALWAYS KEEP THE ORIGINAL VIDEO
        aug_seq.append(seq); aug_lbl.append(label)
        
        # 2. TEST 1: Heavy Jitter (Noise)
        if USE_JITTER:
            for _ in range(2):
                aug_seq.append(seq + np.random.normal(0, 0.05, seq.shape))
                aug_lbl.append(label)
                
        # 3. TEST 2: Speed Variations
        if USE_SPEED:
            for speed in [0.80, 1.20]:
                new_seq = resample(resample(seq, int(len(seq) * speed)), 30)
                aug_seq.append(new_seq); aug_lbl.append(label)
                
        # 4. TEST 3: Spatial Shift
        if USE_SHIFT:
            aug_seq.append(seq + np.random.uniform(-0.1, 0.1, seq.shape))
            aug_lbl.append(label)

    return np.array(aug_seq), np.array(aug_lbl)

def create_model(num_classes):
    """Builds a fresh model for each fold so weights don't carry over."""
    model = Sequential()
    
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', input_shape=(30, 258)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(0.005))))
    model.add(Dropout(0.5)) 
    model.add(BatchNormalization())

    model.add(Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2(0.005))))
    model.add(Dropout(0.5)) 
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling1D())

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.00005) 
    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy', TopKCategoricalAccuracy(k=5, name='top_5_acc')])
    return model

if __name__ == "__main__":
    X, y, actions = load_and_process_data()
    if X is None or len(X) == 0: exit()
    
    np.save('actions.npy', actions)

    # --- CROSS-VALIDATION SETUP ---
    # Stratified ensures every fold gets an equal balance of all 20 classes
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_no = 1
    top1_scores = []
    top5_scores = []

    for train_index, val_index in skf.split(X, y):
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold_no} / {n_splits}...")
        print(f"{'='*50}")

        # 1. Split the raw data first
        X_train_raw, X_val = X[train_index], X[val_index]
        y_train_raw, y_val = y[train_index], y[val_index]

        # 2. Augment ONLY the training data
        X_train, y_train = augment_sequences(X_train_raw, y_train_raw)
        
        # 3. Calculate balanced class weights based on the augmented training set
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))

        # 4. Convert labels to one-hot encoding
        y_train = to_categorical(y_train, num_classes=len(actions))
        y_val = to_categorical(y_val, num_classes=len(actions))

        # 5. Build a fresh model
        model = create_model(len(actions))

        # Save the best model for this specific fold
        model_name = f'sign_language_model_fold_{fold_no}.keras'
        callbacks = [
            EarlyStopping(monitor='val_categorical_accuracy', patience=35, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=8, min_lr=0.000001),
            ModelCheckpoint(model_name, monitor='val_categorical_accuracy', save_best_only=True)
        ]

        print(f"Training on {len(X_train)} augmented samples. Validating on {len(X_val)} pure samples.")
        
        # 6. Train the fold
        model.fit(X_train, y_train, epochs=400, batch_size=32, validation_data=(X_val, y_val),
                  callbacks=callbacks, class_weight=class_weights_dict, verbose=1)

        # 7. Evaluate the best weights for this fold
        print(f"\nEvaluating Fold {fold_no}...")
        scores = model.evaluate(X_val, y_val, verbose=0)
        
        # scores[1] is Top-1, scores[2] is Top-5
        print(f"Fold {fold_no} - Top-1 Acc: {scores[1]*100:.2f}% | Top-5 Acc: {scores[2]*100:.2f}%")
        
        top1_scores.append(scores[1])
        top5_scores.append(scores[2])
        fold_no += 1

    # --- FINAL VERDICT ---
    print(f"\n{'='*50}")
    print("FINAL CROSS-VALIDATION RESULTS (5 FOLDS):")
    print(f"{'='*50}")
    print(f"Average Top-1 Accuracy: {np.mean(top1_scores)*100:.2f}% (+/- {np.std(top1_scores)*100:.2f}%)")
    print(f"Average Top-5 Accuracy: {np.mean(top5_scores)*100:.2f}% (+/- {np.std(top5_scores)*100:.2f}%)")
    print("Training Complete. The most robust models have been saved.")