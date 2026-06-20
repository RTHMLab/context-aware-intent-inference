#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

INPUT_CSV = Path("docs/experiment_reports/final_method_comparison/final_paired_all_methods_lead_time.csv")
OUT_DIR = Path("docs/experiment_reports/final_method_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "figure7_early_prediction_boxplot.png"
OUT_PDF = OUT_DIR / "figure7_early_prediction_boxplot.pdf"

df = pd.read_csv(INPUT_CSV).dropna(subset=["GR_lead", "BF_lead", "E2E_lead"])

gr = df["GR_lead"]
bf = df["BF_lead"]
e2e = df["E2E_lead"]

means = {
    "GR": gr.mean(),
    "BF": bf.mean(),
    "E2E": e2e.mean(),
}

# Paired tests on the common successful-trial subset
t_bf_gr = ttest_rel(bf, gr)
t_bf_e2e = ttest_rel(bf, e2e)
w_bf_gr = wilcoxon(bf, gr)
w_bf_e2e = wilcoxon(bf, e2e)

print("n_common_successful_trials:", len(df))
print("GR mean lead:", means["GR"])
print("BF mean lead:", means["BF"])
print("E2E mean lead:", means["E2E"])
print()
print("BF vs GR paired t-test:", t_bf_gr)
print("BF vs GR Wilcoxon:", w_bf_gr)
print("BF vs E2E paired t-test:", t_bf_e2e)
print("BF vs E2E Wilcoxon:", w_bf_e2e)

fig, ax = plt.subplots(figsize=(5.6, 3.6))

bp = ax.boxplot(
    [gr, bf, e2e],
    labels=["GR", "BF", "E2E"],
    patch_artist=True,
    widths=0.5,
    showfliers=False,
)

facecolors = ["#EAF3FF", "#FFF2E6", "#EAF9F1"]
edgecolors = ["#4C97D8", "#E67E22", "#43B581"]
hatches = ["//", "\\\\", "xx"]

for patch, fc, ec, hatch in zip(bp["boxes"], facecolors, edgecolors, hatches):
    patch.set_facecolor(fc)
    patch.set_edgecolor(ec)
    patch.set_hatch(hatch)
    patch.set_linewidth(1.2)

for whisker in bp["whiskers"]:
    whisker.set_color("#555555")
    whisker.set_linewidth(1.0)

for cap in bp["caps"]:
    cap.set_color("#555555")
    cap.set_linewidth(1.0)

for median in bp["medians"]:
    median.set_color("#333333")
    median.set_linewidth(1.2)

ax.set_title("Overall Prediction Lead Time by Method (Box Plot)", fontsize=11)
ax.set_xlabel("Method", fontsize=10)
ax.set_ylabel("Prediction Lead Time (ms)", fontsize=10)
ax.tick_params(axis="both", labelsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.55)

# Mean labels with white background
for i, label in enumerate(["GR", "BF", "E2E"], start=1):
    val = means[label]
    ax.text(
        i,
        val + 8,
        f"Avg: {val:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2),
        color="#333333",
    )

def add_sig(ax, x1, x2, y, text, h=18):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.9, c="#333333")
    ax.text((x1 + x2) / 2, y + h + 4, text, ha="center", va="bottom", fontsize=9, color="#333333")

# Fixed paper-style y-axis so hidden fliers do not stretch the plot
ax.set_ylim(0, 950)

add_sig(ax, 1, 2, 805, "***")
add_sig(ax, 2, 3, 875, "***")

ax.text(0.62, 915, "*** p < 0.001", fontsize=8)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

print()
print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)
