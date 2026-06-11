import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.data_loader import extract_features, parse_anvil, load_offsets
from scripts.preprocessing import normalize_segments
from scripts.model import build_transformer_model

# --- Configuration ---
SESSION_ID = "sub3"
ANNOTATIONS_DIR = Path("annotations") / "Averaged_Annotations"
DATA_DIR = Path("Data")
MODEL_PATH = Path("trained_models") / "transformer_model.keras"
ENCODER_PATH = Path("trained_models") / "label_encoder.pkl"
NORM_STATS_PATH = Path("trained_models") / "normalization_stats.npz"
GROUNDTRUTH_PATH = Path("GroundTruth-Annotations-Offset.csv")
OUTPUT_MODEL_PATH = Path("trained_models") / f"transformer_model_{SESSION_ID}_finetuned.keras"
WINDOW_SIZE = 500
MAX_TIMESTEPS = 121  # from previous training

print("🔧 Starting fine-tuning for subject:", SESSION_ID)

# Load model, encoder, and normalization stats
model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
norm_stats = np.load(NORM_STATS_PATH)
mean, std = norm_stats['mean'], norm_stats['std']

# Load offset and annotations
offsets = load_offsets(GROUNDTRUTH_PATH)
offset = offsets[SESSION_ID]
anvil_path = ANNOTATIONS_DIR / f"{SESSION_ID}_averaged.anvil"

print("📄 Parsing annotations...")
motion_data_original = pd.read_csv(DATA_DIR / "extracted_JointAngles" / f"{SESSION_ID}_ja.csv")
motion_data_mirrored = pd.read_csv(DATA_DIR / "mirrored_JointAngles" / f"{SESSION_ID}_ja_m.csv")
min_time = motion_data_original['Time (in ms)'].min()
annotations = parse_anvil(anvil_path, min_time, offset)

print("📊 Preparing fine-tuning data (first 15 annotations only)...")
train_segments = []
train_labels = []

for source, motion_data in zip(["original", "mirrored"], [motion_data_original, motion_data_mirrored]):
    for (start, _, label) in annotations[:15]:
        for i in range(3):
            end = start - (250 * i)
            start_win = end - WINDOW_SIZE
            seg = motion_data[(motion_data['Time (in ms)'] >= start_win) & (motion_data['Time (in ms)'] < end)]
            if not seg.empty:
                features = extract_features(seg)
                train_segments.append(features)
                train_labels.append(label)

        for i in range(7):
            end = start - (250 * (i + 4))
            start_win = end - WINDOW_SIZE
            seg = motion_data[(motion_data['Time (in ms)'] >= start_win) & (motion_data['Time (in ms)'] < end)]
            if not seg.empty:
                features = extract_features(seg)
                train_segments.append(features)
                train_labels.append("Nothing")

if not train_segments:
    print("❌ No training segments found for fine-tuning.")
    sys.exit(1)

print(f"✅ Collected {len(train_segments)} segments for fine-tuning")

# Normalize and pad
normalized_segments = [(seg - mean) / (std + 1e-8) for seg in train_segments]
X = pad_sequences(normalized_segments, maxlen=MAX_TIMESTEPS, padding='post', dtype='float32')
y = label_encoder.transform(train_labels)

# Train
print("🚀 Fine-tuning the model...")
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X, y, epochs=25, batch_size=32, verbose=1, callbacks=[early_stop])

# Save
model.save(OUTPUT_MODEL_PATH)
print(f"✅ Fine-tuned model saved to: {OUTPUT_MODEL_PATH}")
