import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.gr.data_loader import (
    TIME_COL,
    TRIAL_NUMBER_COL,
    resolve_case_insensitive_file,
    extract_features,
)


WINDOW_MS = 500


def normalize_sequence(seq, mean, std):
    return (seq - mean) / (std + 1e-8)


def run_trial_inference(
    model,
    label_encoder,
    mean,
    std,
    motion_df,
    trial_row,
    max_len,
    window_ms=WINDOW_MS,
):
    trial_start = float(trial_row["Start Time (Xsens)"])
    action_start = float(trial_row["Action Start Time (Xsens)"])

    # First prediction at t=0 uses window [-500, 0] relative to trial start.
    first_window_start = trial_start - window_ms

    # Last prediction at t=2500 uses window [2000, 2500] relative to trial start.
    last_window_start = action_start - window_ms

    candidate_starts = motion_df[
        (motion_df[TIME_COL] >= first_window_start)
        & (motion_df[TIME_COL] <= last_window_start)
    ][TIME_COL].to_numpy()

    rows = []

    for window_start in candidate_starts:
        window_end = window_start + window_ms

        segment = motion_df[
            (motion_df[TIME_COL] >= window_start)
            & (motion_df[TIME_COL] < window_end)
        ]

        if segment.empty:
            continue

        features = extract_features(segment)
        normalized = normalize_sequence(features, mean, std)
        padded = pad_sequences(
            [normalized],
            maxlen=max_len,
            dtype="float32",
            padding="post",
            value=0.0,
        )

        preds = model.predict(padded, verbose=0)[0]

        row = {
            "time_ms": window_end - trial_start,
        }

        for cls, conf in zip(label_encoder.classes_, preds):
            row[f"{cls}_confidence"] = float(conf)

        confidence_cols = [f"{cls}_confidence" for cls in label_encoder.classes_]
        predicted_col = max(confidence_cols, key=lambda c: row[c])
        row["predicted_gesture"] = predicted_col.replace("_confidence", "")
        row["predicted_confidence"] = row[predicted_col]

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run continuous GR inference on target-subject test trials.")
    parser.add_argument("--target_subject", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--base_model_dir", default=None)
    parser.add_argument("--split_csv", default=None)
    parser.add_argument("--joint_angles_root", default="data/extracted_JointAngles")
    parser.add_argument("--trials_root", default="data/trials")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    target_subject = args.target_subject.lower()

    model_dir = Path(args.model_dir or f"results/gr/finetuned_{target_subject}")
    base_model_dir = Path(args.base_model_dir or f"results/gr/base_{target_subject}")
    split_csv = Path(args.split_csv or f"results/gr/splits/{target_subject}_trial_split.csv")
    output_dir = Path(args.output_dir or f"results/gr/inference_{target_subject}")
    joint_angles_root = Path(args.joint_angles_root)
    trials_root = Path(args.trials_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_trial_dir = output_dir / "per_trial"
    per_trial_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"transformer_gr_finetuned_{target_subject}.keras"
    norm_path = base_model_dir / "normalization_stats.npz"
    encoder_path = base_model_dir / "label_encoder.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing fine-tuned model: {model_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalization stats: {norm_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing label encoder: {encoder_path}")
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_csv}")

    print(f"Loading model from {model_path}")
    model = load_model(model_path)

    label_encoder = joblib.load(encoder_path)
    norm_stats = np.load(norm_path)
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    max_len = model.input_shape[1]

    split_df = pd.read_csv(split_csv)
    test_df = split_df[split_df["Split"] == "test"].copy()

    all_rows = []

    for session_id, session_split_df in test_df.groupby("Session", sort=False):
        ja_csv = resolve_case_insensitive_file(joint_angles_root, f"{session_id}_ja.csv")
        trials_csv = resolve_case_insensitive_file(trials_root, f"{session_id}_trials.csv")

        motion_df = pd.read_csv(ja_csv).sort_values(TIME_COL).reset_index(drop=True)
        trials_df = pd.read_csv(trials_csv)

        for _, split_row in session_split_df.iterrows():
            session_trial = int(split_row["Session Trial"])
            global_trial = int(split_row["Global Trial"])
            gesture = str(split_row["Gesture"])

            trial_match = trials_df[trials_df[TRIAL_NUMBER_COL] == session_trial]
            if trial_match.empty:
                raise RuntimeError(
                    f"Could not find trial {session_trial} in {trials_csv}"
                )

            trial_row = trial_match.iloc[0]

            pred_df = run_trial_inference(
                model=model,
                label_encoder=label_encoder,
                mean=mean,
                std=std,
                motion_df=motion_df,
                trial_row=trial_row,
                max_len=max_len,
            )

            if pred_df.empty:
                print(f"No predictions for {session_id} trial {session_trial}")
                continue

            pred_df.insert(0, "Subject", target_subject)
            pred_df.insert(1, "Session", session_id)
            pred_df.insert(2, "Global Trial", global_trial)
            pred_df.insert(3, "Session Trial", session_trial)
            pred_df.insert(4, "Gesture", gesture)
            pred_df.insert(5, "Split", "test")

            safe_gesture = gesture.replace("/", "-")
            trial_csv = per_trial_dir / f"{global_trial:04d}_{session_id}_trial{session_trial}_{safe_gesture}.csv"
            pred_df.to_csv(trial_csv, index=False)

            all_rows.append(pred_df)

            print(
                f"Saved trial predictions: {trial_csv} "
                f"rows={len(pred_df)} time={pred_df['time_ms'].min():.1f}-{pred_df['time_ms'].max():.1f} ms"
            )

    if not all_rows:
        raise RuntimeError("No inference predictions were generated.")

    all_df = pd.concat(all_rows, ignore_index=True)

    combined_csv = output_dir / "gr_predictions_all_test_trials.csv"
    all_df.to_csv(combined_csv, index=False)

    print(f"Saved combined GR predictions to {combined_csv}")
    print(f"Total prediction rows: {len(all_df)}")
    print(f"Total test trials: {all_df[['Session', 'Session Trial']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
