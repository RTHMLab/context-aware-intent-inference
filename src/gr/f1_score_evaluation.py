import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
SESSION_ID = "sub3"
INPUT_DIR = Path("results") / SESSION_ID / "original_finetuned_results"
OUTPUT_DIR = Path("results") / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Confidence threshold for one of the two time-to-confidence types ---
CONF_THRESH = 0.5

# --- Data structures to hold time-to-confidence info ---
time_to_conf_top1 = defaultdict(list)
time_to_conf_thresh = defaultdict(list)
outlier_labels_top1 = defaultdict(list)
outlier_labels_thresh = defaultdict(list)

# --- Read prediction CSVs ---
csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

for file in csv_files:
    gesture_label = file.split("_")[1].replace(".csv", "").strip().lower()
    trial_id = file.split("_")[0]
    df = pd.read_csv(INPUT_DIR / file)

    # Ground truth gesture is only valid after 2000 ms
    valid_df = df[df['time'] >= 2000]
    valid_df = valid_df[valid_df['time'] <= 3000]

    found_top1, found_thresh = False, False

    for _, row in valid_df.iterrows():
        time_ms = row['time']
        pred = row.drop("time")

        # Get gesture with highest confidence
        top1_label = pred.idxmax().replace("_confidence", "").strip().lower()
        top1_conf = pred.max()
        true_label = gesture_label

        if not found_top1 and top1_label == true_label:
            ttc = 3000 - time_ms
            time_to_conf_top1[true_label].append(ttc)
            outlier_labels_top1[true_label].append(trial_id)
            found_top1 = True

        # Check if true gesture's confidence crosses threshold
        true_conf_key = true_label + "_confidence"
        if true_conf_key in pred and not found_thresh and pred[true_conf_key] >= CONF_THRESH:
            ttc = 3000 - time_ms
            time_to_conf_thresh[true_label].append(ttc)
            outlier_labels_thresh[true_label].append(trial_id)
            found_thresh = True

# --- Plotting function ---
def plot_boxplot(data_dict, title, filename, outlier_labels_dict=None):
    gesture_labels = []
    times = []
    trial_ids = []

    for gesture, values in data_dict.items():
        for i, val in enumerate(values):
            gesture_labels.append(gesture)
            times.append(val)
            if outlier_labels_dict:
                trial_ids.append(outlier_labels_dict[gesture][i])

    df_plot = pd.DataFrame({
        "Gesture": gesture_labels,
        "Time to Confidence (ms)": times
    })

    if outlier_labels_dict:
        df_plot["TrialID"] = trial_ids

    if df_plot.empty:
        print(f"⚠️ Skipping plot {filename}: No data available.")
        return

    plt.figure(figsize=(14, 6))
    ax = sns.boxplot(data=df_plot, x="Gesture", y="Time to Confidence (ms)", palette="Set3")
    #plt.ylim(200, 510)
    #plt.xticks(rotation=45)
    plt.title(title)

    # Annotate outliers
    if outlier_labels_dict:
        grouped = df_plot.groupby("Gesture")
        for gesture, group in grouped:
            q1 = group["Time to Confidence (ms)"].quantile(0.25)
            q3 = group["Time to Confidence (ms)"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = group[
                (group["Time to Confidence (ms)"] < lower_bound) |
                (group["Time to Confidence (ms)"] > upper_bound)
            ]

            for _, row in outliers.iterrows():
                ax.text(row.name, row["Time to Confidence (ms)"] + 3, str(row["TrialID"]),
                        horizontalalignment='center', fontsize=8, color='red', rotation=30)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()
    plt.show()
    print(f"✅ Saved box plot: {filename}")

# --- Create plots ---
plot_boxplot(time_to_conf_top1, "Time to Confidence (Top-1 Prediction Match)", "boxplot_top1_labeled.png", outlier_labels_top1)
plot_boxplot(time_to_conf_thresh, f"Time to Confidence (>={CONF_THRESH*100:.0f}% Confidence)", "boxplot_thresh_labeled.png", outlier_labels_thresh)