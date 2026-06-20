#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

REPORT_DIR = Path("docs/experiment_reports/final_method_comparison")
OUT_DIR = REPORT_DIR / "per_class_f1_plots"
PNG_DIR = OUT_DIR / "png"
PDF_DIR = OUT_DIR / "pdf"

PNG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

per_class_path = REPORT_DIR / "final_per_class_f1.csv"
subject_metrics_path = REPORT_DIR / "final_subject_metrics.csv"

# This file has full per-class F1 by Method, Class
full_df = pd.read_csv(per_class_path)

# We need subject-level per-class F1. Compute from prediction outputs directly.
CLASSES = [
    "Clean-table",
    "Nothing",
    "Pick-up-backpack",
    "Push-chair",
    "Sit-on-chair",
    "Sit-on-couch",
    "Sit-on-table",
    "Stand-on-couch",
    "Wear-backpack",
]

SUBJECTS = ["sub2b", "sub3", "sub4", "sub5", "sub6"]

METHOD_FILES = {
    "GR": "results/gr/inference_{subject}/gr_predictions_all_test_trials.csv",
    "BF": "results/bayesian_fusion/inference_{subject}/bf_predictions_all_test_trials.csv",
    "E2E": "results/e2e_transformer_matched_gr/inference_{subject}/e2e_predictions_all_test_trials.csv",
}

DISPLAY_LABELS = {
    "Clean-table": "Clean",
    "Nothing": "Nothing",
    "Pick-up-backpack": "Pick-up",
    "Push-chair": "Push-back",
    "Sit-on-chair": "Sit-on-chair",
    "Sit-on-couch": "Sit-on-couch",
    "Sit-on-table": "Sit-on-table",
    "Stand-on-couch": "Stand",
    "Wear-backpack": "Wear",
}

PLOT_ORDER = [
    "Nothing",
    "Stand-on-couch",
    "Sit-on-table",
    "Pick-up-backpack",
    "Sit-on-couch",
    "Sit-on-chair",
    "Clean-table",
    "Push-chair",
    "Wear-backpack",
]

def norm(s):
    return str(s).lower().replace("-", "_").replace(" ", "_")

def find_col(df, candidates):
    norm_map = {norm(c): c for c in df.columns}
    for cand in candidates:
        if norm(cand) in norm_map:
            return norm_map[norm(cand)]
    for c in df.columns:
        nc = norm(c)
        if any(norm(cand) in nc for cand in candidates):
            return c
    raise ValueError(f"Missing column from candidates {candidates}. Columns: {df.columns.tolist()}")

def find_conf_cols(df):
    out = {}
    for cls in CLASSES:
        key = norm(cls)
        hits = [
            c for c in df.columns
            if key in norm(c)
            and ("conf" in norm(c) or "confidence" in norm(c) or "intent" in norm(c))
        ]
        if not hits:
            raise ValueError(f"Could not find confidence column for {cls}. Columns: {df.columns.tolist()}")
        out[cls] = hits[0]
    return out

def compute_subject_per_class(subject):
    from sklearn.metrics import precision_recall_fscore_support

    rows = []

    for method, template in METHOD_FILES.items():
        path = Path(template.format(subject=subject))
        df = pd.read_csv(path)

        time_col = find_col(df, ["Time (ms)", "time_ms", "timestamp_ms", "Normalized_Time"])
        label_col = find_col(df, ["True_Label", "True Label", "Gesture", "label"])
        conf_cols = find_conf_cols(df)

        scores = df[list(conf_cols.values())].copy()
        scores.columns = list(conf_cols.keys())

        y_pred = scores.idxmax(axis=1)
        y_action = df[label_col].astype(str)
        y_true = y_action.where(df[time_col] >= 2000, "Nothing")

        _, _, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASSES,
            zero_division=0,
        )

        for cls, f, s in zip(CLASSES, f1, support):
            rows.append({
                "Subject": subject,
                "Method": method,
                "Class": cls,
                "f1_score": f,
                "support": s,
            })

    return pd.DataFrame(rows)

def plot_per_class(df, title, out_stem):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    methods = ["GR", "BF", "E2E"]
    colors = {"GR": "#4C97D8", "BF": "#E67E22", "E2E": "#43B581"}
    hatches = {"GR": "///", "BF": "\\\\\\", "E2E": "xxx"}

    x = np.arange(len(PLOT_ORDER))
    width = 0.24

    for i, method in enumerate(methods):
        vals = []
        for cls in PLOT_ORDER:
            row = df[(df["Method"] == method) & (df["Class"] == cls)]
            vals.append(float(row["f1_score"].iloc[0]) if not row.empty else 0.0)

        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=method,
            color="white",
            edgecolor=colors[method],
            linewidth=1.1,
            hatch=hatches[method],
        )

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=10)
    ax.set_xlabel("Ground Truth Intent Label", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS[c] for c in PLOT_ORDER], rotation=35, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="Method", fontsize=8, title_fontsize=8, loc="upper right")

    fig.tight_layout()

    png = PNG_DIR / f"{out_stem}.png"
    pdf = PDF_DIR / f"{out_stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png)
    print("Saved:", pdf)

# Full pooled plot
plot_per_class(
    full_df,
    "Per-Class F1-Score Across Three Methods",
    "per_class_f1_full_loso",
)

# Subject plots
all_subject_rows = []
for subject in SUBJECTS:
    sdf = compute_subject_per_class(subject)
    all_subject_rows.append(sdf)

    plot_per_class(
        sdf,
        f"Per-Class F1-Score Across Three Methods ({subject})",
        f"per_class_f1_{subject}",
    )

subject_df = pd.concat(all_subject_rows, ignore_index=True)
subject_df.to_csv(REPORT_DIR / "final_subject_per_class_f1.csv", index=False)
print("Saved:", REPORT_DIR / "final_subject_per_class_f1.csv")
