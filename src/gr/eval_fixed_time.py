import os
import pandas as pd
from pathlib import Path

# --- Configuration ---
SESSION_ID = "sub3"
INPUT_DIR = Path("results") / SESSION_ID / "mirrored" #/"original_finetuned_results"
OUTPUT_DIR = Path("results") / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "mirrored2800-fixedTimeGR-evaluation.csv"

# Get all CSV files in the original results directory
csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

# Initialize list to store results
results = []

for file in csv_files:
    # Extract annotation number and ground truth from filename
    filename_parts = file.split("_")
    annotation_number = filename_parts[0]  # First part of filename
    ground_truth = filename_parts[1].replace(".csv", "")  # Second part (without .csv)

    # Load inference data
    df = pd.read_csv(INPUT_DIR / file)

    if "time" not in df.columns:
        print(f"⚠️ Warning: 'time' column missing in {file}. Skipping...")
        continue

    # Find the row where 'time' is closest to but not exceeding 3000 ms
    target_time = 2800
    df_filtered = df[df['time'] <= target_time]

    if df_filtered.empty:
        print(f"⚠️ Warning: No row found at or before 3000ms in {file}. Skipping...")
        continue

    closest_row = df_filtered.iloc[-1]  # Last row <= 3000ms

    # Extract confidence scores (excluding time)
    confidence_scores = closest_row.drop("time")

    # Find gesture with highest confidence
    predicted_gesture = confidence_scores.idxmax()

    # Store result
    results.append([
        annotation_number,
        ground_truth,
        predicted_gesture,
        *confidence_scores.values
    ])

# Create DataFrame for results
columns = ["Annotation Number", "Ground Truth", "Predicted Gesture"] + list(confidence_scores.index)
results_df = pd.DataFrame(results, columns=columns)

# Save results
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Gesture recognition analysis saved as: {OUTPUT_CSV}")
