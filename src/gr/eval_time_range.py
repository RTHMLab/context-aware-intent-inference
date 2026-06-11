import os
import pandas as pd
from pathlib import Path

# --- Configuration ---
SESSION_ID = "sub3"
INPUT_DIR = Path("results") / SESSION_ID / "original_finetuned_ablation_results"
OUTPUT_DIR = Path("results") / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "ablation-with finetuning-evaluation.csv"

# Define evaluation ranges (start_ms, end_ms)
RANGES = {
    "Range 1 (0-3000ms)": (0, 3000),
    "Range 2 (500-3000ms)": (500, 3000),
    "Range 3 (0-500ms)": (0, 500),
    "Range 4 (2500-3000ms)": (2500, 3000)
}

# Collect results per trial and totals for each range
results = []
range_totals = {k: 0 for k in RANGES}
range_correct = {k: 0 for k in RANGES}

csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

for file in csv_files:
    filename_parts = file.split("_")
    annotation_number = filename_parts[0]
    gesture_label = filename_parts[1].replace(".csv", "")

    df = pd.read_csv(INPUT_DIR / file)
    if "time" not in df.columns:
        print(f"⚠️ Warning: 'time' column missing in {file}. Skipping...")
        continue

    correct_counts = {k: 0 for k in RANGES}
    total_counts = {k: 0 for k in RANGES}

    for _, row in df.iterrows():
        predicted_gesture = row.drop("time").idxmax()

        # --- 🔧 Sanitize labels ---
        predicted_clean = predicted_gesture.replace("_confidence", "").strip().lower()

        # Determine ground truth based on time
        if row['time'] < 2500:
            ground_truth_clean = "nothing"
        else:
            ground_truth_clean = gesture_label.strip().lower()

        for range_label, (start_ms, end_ms) in RANGES.items():
            if start_ms <= row['time'] <= end_ms:
                total_counts[range_label] += 1
                if predicted_clean == ground_truth_clean:
                    correct_counts[range_label] += 1

    # Store trial-level diagnostics
    row_data = [file]
    for range_label in RANGES:
        total = total_counts[range_label]
        correct = correct_counts[range_label]
        acc = correct / total if total > 0 else 0
        row_data.append(round(acc * 100, 2))

        range_totals[range_label] += total
        range_correct[range_label] += correct

    results.append(row_data)

# Build and save results
columns = ["File"] + [f"{r} Accuracy (%)" for r in RANGES.keys()]
df_results = pd.DataFrame(results, columns=columns)
df_results.to_csv(OUTPUT_CSV, index=False)

print("\n📊 Final Overall Accuracies of Fine-tuned model:")
for range_label in RANGES:
    total = range_totals[range_label]
    correct = range_correct[range_label]
    acc = correct / total if total > 0 else 0
    print(f"{range_label}: {acc:.4f} accuracy ({correct}/{total})")
