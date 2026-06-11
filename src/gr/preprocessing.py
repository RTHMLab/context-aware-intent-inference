import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings before import

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root to Python path

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import joblib

from scripts.data_loader import USE_MIRRORED  # Import ablation toggle

# Paths for saving stats and encoder
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'trained_models')
NORMALIZATION_STATS_PATH = os.path.join(MODEL_DIR, f"normalization_stats{'_ablation' if not USE_MIRRORED else ''}.npz")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, f"label_encoder{'_ablation' if not USE_MIRRORED else ''}.pkl")

def normalize_segments(segments):
    all_data = np.concatenate(segments, axis=0)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0) + 1e-8
    normalized_segments = [(seg - mean) / std for seg in segments]
    return normalized_segments, mean, std

def preprocess_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw, save_dir):
    print("Normalizing features...")
    X_train_norm, mean, std = normalize_segments(X_train_raw)
    X_test_norm = [(seg - mean) / std for seg in X_test_raw]

    np.savez(NORMALIZATION_STATS_PATH, mean=mean, std=std)
    print(f"✅ Saved normalization stats to {NORMALIZATION_STATS_PATH}")

    max_len = max(max(seg.shape[0] for seg in X_train_norm), max(seg.shape[0] for seg in X_test_norm))
    num_features = X_train_norm[0].shape[1]

    print(f"Padding sequences to length: {max_len}")
    X_train_padded = pad_sequences(X_train_norm, maxlen=max_len, dtype='float32', padding='post', value=0.0)
    X_test_padded = pad_sequences(X_test_norm, maxlen=max_len, dtype='float32', padding='post', value=0.0)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_raw)
    y_test_encoded = label_encoder.transform(y_test_raw)

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"✅ Saved label encoder to {LABEL_ENCODER_PATH}")

    return X_train_padded, y_train_encoded, X_test_padded, y_test_encoded, label_encoder, max_len, num_features

if __name__ == "__main__":
    # main function to execute just for testing the functionality
    from scripts.data_loader import load_all_data

    print("Loading raw data...")
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_all_data()

    print("Preprocessing...")
    X_train_padded, y_train_encoded, X_test_padded, y_test_encoded, label_encoder, max_len, num_features = preprocess_data(
        X_train_raw, y_train_raw, X_test_raw, y_test_raw, save_dir='trained_models')

    print(f"✅ Done. Train shape: {X_train_padded.shape}, Test shape: {X_test_padded.shape}")
