import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
SESSION_ID = "sub3"
INPUT_DIR = Path("results") / SESSION_ID / "original"
OUTPUT_DIR = Path("results") / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILES = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

# --- Threshold for confidence ---
CONF_THRESH = 0.5
MAX_TIME = 3000

# --- Initialize storage ---
time_to_conf_top = defaultdict(list)     # time when correct class becomes top-1 prediction
time_to_conf_thresh = defaultdict(list)  # time when correct class exceeds confidence threshold

# --- Process each CSV file ---
for file in CSV_FILES:
    filepath = INPUT_DIR / file
    df = pd.read_csv(filepath)
    if "time" not in df.columns:
        continue

    parts = file.split("_")
    true_label = parts[1].replace(".csv", "").strip().lower()

    detected_top = False
    detected_thresh = False

    for _, row in df.iterrows():
        t = row['time']
        if t > MAX_TIME:
            break

        confidences = row.drop("time")
        top_pred = confidences.idxmax().replace("_confidence", "").strip().lower()
        true_conf = confidences.get(f"{true_label}_confidence", 0)

        if not detected_top and top_pred == true_label:
            time_to_conf_top[true_label].append(MAX_TIME - t)
            detected_top = True

        if not detected_thresh and true_conf >= CONF_THRESH:
            time_to_conf_thresh[true_label].append(MAX_TIME - t)
            detected_thresh = True

# --- Convert to DataFrame for plotting ---
def plot_boxplot(data_dict, title, filename):
    data = []
    for label, values in data_dict.items():
        for val in values:
            data.append({"Gesture": label, "Time to Confidence (ms)": val})
    df_plot = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_plot, x="Gesture", y="Time to Confidence (ms)", palette="Set3")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()

# --- Plotting ---
plot_boxplot(time_to_conf_top, "Time to Confidence (Top-1 Prediction)", "boxplot_top1.png")
plot_boxplot(time_to_conf_thresh, f"Time to Confidence (>={CONF_THRESH*100:.0f}% Confidence)", "boxplot_thresh.png")

# --- Save raw stats ---
def save_stats(data_dict, name):
    rows = []
    for cls, times in data_dict.items():
        if times:
            rows.append([
                cls,
                len(times),
                round(np.mean(times), 2),
                round(np.std(times), 2),
                round(np.min(times), 2),
                round(np.max(times), 2)
            ])
    df = pd.DataFrame(rows, columns=["Gesture", "Count", "Mean", "Std", "Min", "Max"])
    df.to_csv(OUTPUT_DIR / f"{name}_stats.csv", index=False)

save_stats(time_to_conf_top, "top1")
save_stats(time_to_conf_thresh, "conf_thresh")

print("✅ Time-to-confidence plots and statistics saved.")
