import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings before import

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root to Python path

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from scripts.data_loader import extract_features, parse_anvil, load_offsets

'''
# --- Configuration ---
SESSION_ID = "sub3"
MODES = ["original", "mirrored"]
RESULTS_DIR = Path("results") / SESSION_ID
DATA_DIR = Path("Data")
ANNOTATIONS_DIR = Path("annotations") / "Averaged_Annotations"  # ✅ Updated subdirectory
MODEL_PATH = Path("trained_models") / "transformer_model.keras"  # ✅ Updated to .keras
ENCODER_PATH = Path("trained_models") / "label_encoder.pkl"
NORMALIZATION_PATH = Path("trained_models") / "normalization_stats.npz"
GROUNDTRUTH_PATH = Path("GroundTruth-Annotations-Offset.csv")
WINDOW_SIZE = 500'''

# --- Configuration ---
SESSION_ID = "sub3"
MODES = ["original"]
RESULTS_DIR = Path("results") / SESSION_ID
DATA_DIR = Path("Data")
ANNOTATIONS_DIR = Path("annotations") / "Averaged_Annotations"  # ✅ Updated subdirectory
MODEL_PATH = Path("trained_models") / "transformer_model_ablation.keras"  # ✅ Updated to .keras
ENCODER_PATH = Path("trained_models") / "label_encoder_ablation.pkl"
NORMALIZATION_PATH = Path("trained_models") / "normalization_stats_ablation.npz"
GROUNDTRUTH_PATH = Path("GroundTruth-Annotations-Offset.csv")
WINDOW_SIZE = 500

# --- Load Trained Model & Tools ---
print("Loading model and utilities...")
model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
norm_stats = np.load(NORMALIZATION_PATH)
mean, std = norm_stats['mean'], norm_stats['std']

# --- Load Annotations & Offset ---
print("Loading annotations...")
csv_path = DATA_DIR / ("mirrored_JointAngles" if MODES[0] == "mirrored" else "extracted_JointAngles") / f"{SESSION_ID}_ja{'_m' if MODES[0] == 'mirrored' else ''}.csv"
motion_data = pd.read_csv(csv_path)
time_col = 'Time (in ms)'
min_time = motion_data[time_col].min()

anvil_path = ANNOTATIONS_DIR / f"{SESSION_ID}_averaged.anvil"
offsets = load_offsets(GROUNDTRUTH_PATH)
offset = offsets[SESSION_ID]
annotations = parse_anvil(anvil_path, min_time, offset)

# --- Helper for normalization ---
def normalize_sequence(seq, mean, std):
    return (seq - mean) / (std + 1e-8)

# --- Process Each Mode (original & mirrored) ---
for mode in MODES:
    print(f"\nProcessing {mode} data...")
    results_path = RESULTS_DIR / mode
    results_path.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / ("mirrored_JointAngles" if mode == "mirrored" else "extracted_JointAngles") / f"{SESSION_ID}_ja{'_m' if mode == 'mirrored' else ''}.csv"
    motion_data = pd.read_csv(csv_path)
    min_time = motion_data[time_col].min()

    # Sliding window inference per annotation
    for i, (start, _, gesture) in enumerate(annotations):
        pred_list = []
        start_time = start - 3000 - 499
        end_time = start

        time_zero = start_time + WINDOW_SIZE  # This corresponds to the first inference window ending at (start - 3000)

        for t in motion_data[time_col]:
            if t < start_time:
                continue
            if t > end_time:
                break

            window_start = t
            window_end = t + WINDOW_SIZE
            if window_start < min_time:
                continue
            segment = motion_data[(motion_data[time_col] >= window_start) & (motion_data[time_col] < window_end)]
            if segment.empty:
                continue

            features = extract_features(segment)
            normalized = normalize_sequence(features, mean, std)
            padded = pad_sequences([normalized], maxlen=121, padding='post', dtype='float32')

            preds = model.predict(padded, verbose=0)[0]
            pred_row = {f"{cls}_confidence": preds[j] for j, cls in enumerate(label_encoder.classes_)}
            pred_row['time'] = window_end - time_zero  # ✅ 0-based time axis starting at (start - 3000)
            pred_list.append(pred_row)

        df = pd.DataFrame(pred_list)
        output_file = results_path / f"{str(i+1).zfill(3)}_{gesture}.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file}")

        # Plot
        plt.figure(figsize=(12, 6))
        for cls in label_encoder.classes_:
            plt.plot(df['time'], df[f"{cls}_confidence"], label=cls)
        plt.xlabel("Time (ms, relative to inference window start)")
        plt.ylabel("Confidence")
        plt.title(f"Gesture Confidence - {gesture} ({mode})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = output_file.with_suffix(".png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Plot saved: {plot_path}")

print("\n🎉 Inference complete for all annotations.")
