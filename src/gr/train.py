import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings before import

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root to Python path

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load data and check if we're using mirrored data
from scripts.data_loader import load_all_data, USE_MIRRORED
from scripts.preprocessing import preprocess_data
from scripts.model import build_transformer_model

if __name__ == "__main__":
    print("Checking for GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {gpus[0].name}")
    else:
        print("⚠️ No GPU detected. Training will use CPU.")

    # --- ABLATION STUDY CONTROL ---
    # Set USE_MIRRORED = False in data_loader.py to exclude mirrored data (for ablation)
    print(f"🧪 Ablation mode (USE_MIRRORED=False): {'ON' if not USE_MIRRORED else 'OFF'}")

    print("Loading raw data...")
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_all_data()

    print("Preprocessing data...")
    X_train, y_train, X_test, y_test, label_encoder, max_len, num_features = preprocess_data(
        X_train_raw, y_train_raw, X_test_raw, y_test_raw, save_dir='trained_models'
    )

    print("Building model...")
    model = build_transformer_model(input_shape=(max_len, num_features), num_classes=len(label_encoder.classes_))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Training model...")
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    print("Evaluating on test set...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n✅ Test Loss: {loss:.4f}")
    print(f"✅ Test Accuracy: {accuracy:.4f}")

    # --- Save model with appropriate name ---
    model_name = "transformer_model_ablation.keras" if not USE_MIRRORED else "transformer_model.keras"
    model.save(os.path.join('trained_models', model_name))
    print(f"✅ Model saved to trained_models/{model_name}")
