import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


DEFAULT_FIXED_TIMES = [0, 500, 1000, 1500, 2000, 2500]


def get_confidence_columns(df: pd.DataFrame):
    return [
        c for c in df.columns
        if c.endswith("_confidence") and c != "predicted_confidence"
    ]


def infer_prediction_from_confidence(row, confidence_cols):
    best_col = max(confidence_cols, key=lambda c: row[c])
    return best_col.replace("_confidence", "")


def add_prediction_column(df: pd.DataFrame, prediction_col: str | None):
    df = df.copy()

    if prediction_col and prediction_col in df.columns:
        df["predicted_label_eval"] = df[prediction_col].astype(str)
        return df

    for candidate in ["predicted_gesture", "predicted_intent", "prediction", "predicted_label"]:
        if candidate in df.columns:
            df["predicted_label_eval"] = df[candidate].astype(str)
            return df

    confidence_cols = get_confidence_columns(df)
    if not confidence_cols:
        raise RuntimeError("No prediction column or confidence columns found.")

    df["predicted_label_eval"] = df.apply(
        lambda row: infer_prediction_from_confidence(row, confidence_cols),
        axis=1,
    )
    return df


def add_true_label_column(
    df: pd.DataFrame,
    label_col: str,
    label_mode: str,
    onset_time_ms: float,
):
    df = df.copy()

    if label_col not in df.columns:
        raise RuntimeError(f"Missing label column: {label_col}")

    action_label = df[label_col].astype(str)

    if label_mode == "action":
        # Intent-style evaluation: the trial's intended action is the label
        # for every time point from 0 to onset.
        df["true_label_eval"] = action_label

    elif label_mode == "dynamic":
        # Gesture-detection-style evaluation:
        # before onset = Nothing, at/after onset = action.
        df["true_label_eval"] = np.where(
            df["time_ms"] < onset_time_ms,
            "Nothing",
            action_label,
        )

    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    return df


def nearest_time_rows(df: pd.DataFrame, target_time: float):
    rows = []

    group_cols = ["Subject", "Session", "Global Trial", "Session Trial"]
    for _, group in df.groupby(group_cols, sort=False):
        idx = (group["time_ms"] - target_time).abs().idxmin()
        rows.append(df.loc[idx])

    return pd.DataFrame(rows)


def metrics_for_rows(df: pd.DataFrame):
    y_true = df["true_label_eval"]
    y_pred = df["predicted_label_eval"]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "num_rows": len(df),
        "num_trials": df[["Subject", "Session", "Global Trial", "Session Trial"]]
        .drop_duplicates()
        .shape[0],
    }


def build_per_class_metrics(df: pd.DataFrame):
    report = classification_report(
        df["true_label_eval"],
        df["predicted_label_eval"],
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label, vals in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue

        rows.append({
            "class": label,
            "precision": vals["precision"],
            "recall": vals["recall"],
            "f1_score": vals["f1-score"],
            "support": vals["support"],
        })

    return pd.DataFrame(rows)


def build_fixed_time_metrics(df: pd.DataFrame, fixed_times):
    rows = []

    for t in fixed_times:
        tdf = nearest_time_rows(df, t)
        m = metrics_for_rows(tdf)
        m["time_ms"] = t
        rows.append(m)

    return pd.DataFrame(rows)[
        ["time_ms", "accuracy", "macro_f1", "weighted_f1", "num_rows", "num_trials"]
    ]


def build_subject_metrics(df: pd.DataFrame):
    rows = []

    for subject, sdf in df.groupby("Subject", sort=False):
        m = metrics_for_rows(sdf)
        m["Subject"] = subject
        rows.append(m)

    return pd.DataFrame(rows)[
        ["Subject", "accuracy", "macro_f1", "weighted_f1", "num_rows", "num_trials"]
    ]


def build_lead_time_metrics(df: pd.DataFrame, onset_time_ms: float):
    rows = []

    group_cols = ["Subject", "Session", "Global Trial", "Session Trial", "Gesture"]

    for keys, group in df.groupby(group_cols, sort=False):
        subject, session, global_trial, session_trial, gesture = keys

        correct = group[group["predicted_label_eval"] == gesture]

        if correct.empty:
            detected = False
            first_correct_time_ms = np.nan
            lead_time_ms = np.nan
        else:
            detected = True
            first_correct_time_ms = float(correct["time_ms"].min())
            lead_time_ms = onset_time_ms - first_correct_time_ms

        rows.append({
            "Subject": subject,
            "Session": session,
            "Global Trial": global_trial,
            "Session Trial": session_trial,
            "Gesture": gesture,
            "detected": detected,
            "first_correct_time_ms": first_correct_time_ms,
            "lead_time_ms": lead_time_ms,
        })

    lead_df = pd.DataFrame(rows)

    summary = (
        lead_df.groupby("Gesture", dropna=False)
        .agg(
            trials=("Gesture", "count"),
            detected_trials=("detected", "sum"),
            detection_rate=("detected", "mean"),
            mean_lead_time_ms=("lead_time_ms", "mean"),
            median_lead_time_ms=("lead_time_ms", "median"),
            std_lead_time_ms=("lead_time_ms", "std"),
        )
        .reset_index()
    )

    return lead_df, summary


def build_confidence_gap_metrics(df: pd.DataFrame, gap_threshold: float):
    confidence_cols = get_confidence_columns(df)

    if not confidence_cols:
        return pd.DataFrame()

    df = df.copy()

    def top2_gap(row):
        vals = sorted([float(row[c]) for c in confidence_cols], reverse=True)
        if len(vals) < 2:
            return np.nan
        return vals[0] - vals[1]

    df["top2_gap"] = df.apply(top2_gap, axis=1)
    amb = df[df["top2_gap"] <= gap_threshold].copy()

    if amb.empty:
        return pd.DataFrame([{
            "confidence_gap_threshold": gap_threshold,
            "num_rows": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
        }])

    m = metrics_for_rows(amb)
    m["confidence_gap_threshold"] = gap_threshold
    return pd.DataFrame([m])[
        ["confidence_gap_threshold", "accuracy", "macro_f1", "weighted_f1", "num_rows", "num_trials"]
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction CSVs for GR/BF/E2E-style outputs.")
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--output_dir", default="results/evaluation")
    parser.add_argument("--label_col", default="Gesture")
    parser.add_argument("--prediction_col", default=None)
    parser.add_argument("--label_mode", choices=["action", "dynamic"], default="action")
    parser.add_argument("--onset_time_ms", type=float, default=2500.0)
    parser.add_argument("--max_time_ms", type=float, default=2500.0)
    parser.add_argument("--confidence_gap_threshold", type=float, default=0.25)
    args = parser.parse_args()

    paths = sorted(Path(".").glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for p in paths:
        print(f"Reading: {p}")
        dfs.append(pd.read_csv(p))

    df = pd.concat(dfs, ignore_index=True)

    required = ["Subject", "Session", "Global Trial", "Session Trial", "time_ms", args.label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = df[(df["time_ms"] >= 0) & (df["time_ms"] <= args.max_time_ms)].copy()

    df = add_prediction_column(df, args.prediction_col)
    df = add_true_label_column(
        df,
        label_col=args.label_col,
        label_mode=args.label_mode,
        onset_time_ms=args.onset_time_ms,
    )

    overall_df = pd.DataFrame([metrics_for_rows(df)])
    subject_df = build_subject_metrics(df)
    per_class_df = build_per_class_metrics(df)
    fixed_time_df = build_fixed_time_metrics(df, DEFAULT_FIXED_TIMES)
    lead_trial_df, lead_summary_df = build_lead_time_metrics(df, onset_time_ms=args.onset_time_ms)
    ambiguous_df = build_confidence_gap_metrics(df, gap_threshold=args.confidence_gap_threshold)

    prefix = args.method_name

    df.to_csv(output_dir / f"{prefix}_predictions_with_eval_labels.csv", index=False)
    overall_df.to_csv(output_dir / f"{prefix}_overall_metrics.csv", index=False)
    subject_df.to_csv(output_dir / f"{prefix}_subject_metrics.csv", index=False)
    per_class_df.to_csv(output_dir / f"{prefix}_per_class_metrics.csv", index=False)
    fixed_time_df.to_csv(output_dir / f"{prefix}_fixed_time_metrics.csv", index=False)
    lead_trial_df.to_csv(output_dir / f"{prefix}_lead_time_by_trial.csv", index=False)
    lead_summary_df.to_csv(output_dir / f"{prefix}_lead_time_summary.csv", index=False)
    ambiguous_df.to_csv(output_dir / f"{prefix}_confidence_gap_metrics.csv", index=False)

    print()
    print("Overall metrics")
    print(overall_df.to_string(index=False))

    print()
    print("Subject metrics")
    print(subject_df.to_string(index=False))

    print()
    print("Saved evaluation outputs to:", output_dir)


if __name__ == "__main__":
    main()
