"""U43 — generate competition-report figures (reproducible).

Outputs PNGs to reports/figures/. Labels are in English to avoid CJK-font
dependency; the report text references them in Chinese.

    python -m scripts.u43_report_figures
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)


def fig_gap():
    tasks = ["T1\npromise", "T2\ntimeline", "T3\nevidence", "T4\nquality"]
    oof = [0.94, 0.605, 0.86, 0.46]
    lb = [0.79, 0.606, 0.68, 0.44]
    x = range(len(tasks)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - w/2 for i in x], oof, w, label="OOF (cross-val proxy)", color="#9ecae1")
    ax.bar([i + w/2 for i in x], lb, w, label="LB (real test)", color="#3182bd")
    for i, (o, l) in enumerate(zip(oof, lb)):
        ax.text(i - w/2, o + .01, f"{o:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, l + .01, f"{l:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(tasks)
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.0)
    ax.set_title("OOF vs Leaderboard per task — gap concentrated in binary T1/T3")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig1_oof_vs_lb_gap.png", dpi=150); plt.close(fig)


def fig_trajectory():
    labels = ["equal8\n(c3)", "TAPT\nbinary", "best-T1\n+poolT3", "3-seed\nT1", "5-seed\nT1", "+pool7\nT3", "7-seed\nT1"]
    scores = [0.6037, 0.6113, 0.6161, 0.6176, 0.6195, 0.6210, 0.6218]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(range(len(scores)), scores, "-o", color="#e6550d", lw=2)
    for i, s in enumerate(scores):
        ax.text(i, s + .0004, f"{s:.4f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("leaderboard weighted score")
    ax.set_ylim(0.602, 0.624)
    ax.set_title("Real leaderboard trajectory via per-task recombination")
    ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(OUT / "fig2_score_trajectory.png", dpi=150); plt.close(fig)


def fig_t4():
    cls = ["Clear\n(566)", "Not Clear\n(101)", "Misleading\n(1)", "N/A\n(332)"]
    f1 = [0.82, 0.35, 0.00, 0.67]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(cls, f1, color=["#31a354", "#fd8d3c", "#de2d26", "#756bb1"])
    for b, v in zip(bars, f1):
        ax.text(b.get_x() + b.get_width()/2, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("F1 (val-slice)"); ax.set_ylim(0, 1.0)
    ax.set_title("T4 per-class F1 (val-slice) — macro gated by Not Clear & Misleading")
    fig.tight_layout(); fig.savefig(OUT / "fig3_t4_perclass.png", dpi=150); plt.close(fig)


def fig_oof_sota():
    ph = ["P32", "P33", "P34", "P35", "P36", "P37", "P38"]
    s = [0.6694, 0.6741, 0.6878, 0.7034, 0.71018, 0.71364, 0.71608]
    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.plot(range(len(s)), s, "-o", color="#756bb1", lw=2)
    for i, v in enumerate(s):
        ax.text(i, v + .0015, f"{v:.4f}".rstrip("0"), ha="center", fontsize=8)
    ax.set_xticks(range(len(ph))); ax.set_xticklabels(ph)
    ax.set_ylabel("OOF weighted score"); ax.set_ylim(0.66, 0.72)
    ax.set_title("OOF SOTA trajectory (Phase 32-38, cross-validation proxy)")
    ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(OUT / "fig_oof_sota.png", dpi=150); plt.close(fig)


def fig_pertask_sources():
    tasks = ["T1", "T2", "T3", "T4"]
    c3 = [0.7864, 0.6060, 0.6750, 0.4370]
    tapt = [0.7967, 0.5117, 0.6804, 0.4288]
    fin = [0.8056, 0.6147, 0.6974, 0.4549]
    x = range(len(tasks)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([i - w for i in x], c3, w, label="c3 (equal8)", color="#bdbdbd")
    ax.bar(list(x), tapt, w, label="TAPT-only", color="#fdae6b")
    ax.bar([i + w for i in x], fin, w, label="final 0.6218", color="#3182bd")
    ax.set_xticks(list(x)); ax.set_xticklabels(tasks)
    ax.set_ylabel("leaderboard F1"); ax.set_ylim(0, 1.0)
    ax.set_title("Per-task LB by source — best-of-each recombination")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "fig_pertask_sources.png", dpi=150); plt.close(fig)


def fig_backbone_t4():
    names = ["macbert-base\n(maxlen384)", "macbert-base\n(maxlen512)", "roberta-large\n+focal", "electra-large"]
    t4 = [0.4502, 0.4443, 0.4312, 0.4266]
    fig, ax = plt.subplots(figsize=(7, 4.0))
    bars = ax.bar(names, t4, color=["#3182bd", "#9ecae1", "#fc9272", "#fb6a4a"])
    for b, v in zip(bars, t4):
        ax.text(b.get_x() + b.get_width()/2, v + .002, f"{v:.4f}", ha="center", fontsize=8)
    ax.axhline(0.4502, ls="--", color="#888", lw=1)
    ax.set_ylabel("T4 single-stem OOF"); ax.set_ylim(0.40, 0.46)
    ax.set_title("Phase 48 — larger backbones overfit T4 (lose to macbert-base)")
    fig.tight_layout(); fig.savefig(OUT / "fig_backbone_t4.png", dpi=150); plt.close(fig)


def fig_groupkfold():
    tasks = ["T1", "T2", "T3", "T4"]
    plain = [0.934, 0.453, 0.857, 0.421]
    tapt = [0.933, 0.521, 0.858, 0.437]
    x = range(len(tasks)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - w/2 for i in x], plain, w, label="plain macbert", color="#bcbddc")
    ax.bar([i + w/2 for i in x], tapt, w, label="macbert-TAPT", color="#756bb1")
    for i, (p, t) in enumerate(zip(plain, tapt)):
        ax.text(i - w/2, p + .01, f"{p:.3f}", ha="center", fontsize=7)
        ax.text(i + w/2, t + .01, f"{t:.3f}", ha="center", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels(tasks)
    ax.set_ylabel("F1 (held-out company GroupKFold)"); ax.set_ylim(0, 1.0)
    ax.set_title("Honest GroupKFold — TAPT edge is T2/T4, not binary T1/T3")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "fig_groupkfold.png", dpi=150); plt.close(fig)


def main():
    fig_gap(); fig_trajectory(); fig_t4()
    fig_oof_sota(); fig_pertask_sources(); fig_backbone_t4(); fig_groupkfold()
    print("figures written to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print("  ", p, f"{p.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
