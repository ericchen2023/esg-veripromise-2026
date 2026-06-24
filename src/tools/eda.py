"""Automated EDA for the VeriPromise ESG 2026 dataset.

Outputs:
    reports/eda/eda_report.html
    reports/eda/metrics.json
    reports/eda/label_distribution.png
    reports/eda/text_length.png

Usage:
    python -m src.tools.eda --csv "data/raw/vpesg4k_train_1000 V1.csv"
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Allow `python -m src.tools.eda` and direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import LABEL_DOMAINS, load_dataset  # noqa: E402

REPORT_DIR = Path("reports/eda")


def _label_distribution(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for field, domain in LABEL_DOMAINS.items():
        counts = Counter(df[field])
        out[field] = {label: int(counts.get(label, 0)) for label in domain}
    return out


def _conditional_consistency(df: pd.DataFrame) -> dict[str, int]:
    """Count constraint violations in the raw labels (sanity check)."""
    violations = {
        "promise_no_but_t2_not_na": int(
            ((df["promise_status"] == "No") & (df["verification_timeline"] != "N/A")).sum()
        ),
        "promise_no_but_t3_not_na": int(
            ((df["promise_status"] == "No") & (df["evidence_status"] != "N/A")).sum()
        ),
        "promise_no_but_t4_not_na": int(
            ((df["promise_status"] == "No") & (df["evidence_quality"] != "N/A")).sum()
        ),
        "evidence_no_but_t4_not_na": int(
            ((df["evidence_status"] == "No") & (df["evidence_quality"] != "N/A")).sum()
        ),
    }
    return violations


def _text_length_stats(df: pd.DataFrame) -> dict[str, float]:
    lens = df["data"].astype(str).str.len()
    return {
        "min": int(lens.min()),
        "p25": float(lens.quantile(0.25)),
        "median": float(lens.median()),
        "mean": float(lens.mean()),
        "p75": float(lens.quantile(0.75)),
        "p95": float(lens.quantile(0.95)),
        "max": int(lens.max()),
        "over_256": int((lens > 256).sum()),
        "over_512": int((lens > 512).sum()),
    }


def _company_stats(df: pd.DataFrame) -> dict[str, int | float]:
    if "company" not in df.columns:
        return {}
    counts = df["company"].value_counts()
    return {
        "n_unique_companies": int(counts.shape[0]),
        "max_samples_per_company": int(counts.max()),
        "median_samples_per_company": float(counts.median()),
        "companies_with_ge_5_samples": int((counts >= 5).sum()),
    }


def _plot_labels(label_dist: dict[str, dict[str, int]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (field, counts) in zip(axes.flat, label_dist.items(), strict=False):
        ax.bar(list(counts.keys()), list(counts.values()), color="steelblue", alpha=0.85)
        ax.set_title(field)
        ax.tick_params(axis="x", rotation=25)
        total = sum(counts.values()) or 1
        for i, (k, v) in enumerate(counts.items()):
            ax.text(i, v, f"{v}\n{v / total * 100:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Label Distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_text_length(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    lens = df["data"].astype(str).str.len()
    ax.hist(lens, bins=50, color="coral", alpha=0.8, edgecolor="black")
    ax.axvline(256, color="red", linestyle="--", label="MAX_LEN=256")
    ax.axvline(512, color="darkred", linestyle="--", label="MAX_LEN=512")
    ax.set_xlabel("character count")
    ax.set_ylabel("samples")
    ax.set_title("Text Length Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _render_html(metrics: dict, paths: dict[str, str], out: Path) -> None:
    html = [
        "<html><head><meta charset='utf-8'><title>VP ESG 2026 EDA</title>",
        "<style>body{font-family:Segoe UI,Arial;margin:24px;}",
        "table{border-collapse:collapse;}td,th{border:1px solid #ccc;padding:6px 10px;}",
        "h1{color:#1a4480;}h2{color:#2a5599;margin-top:28px;}</style></head><body>",
        "<h1>VeriPromise ESG 2026 — EDA Report</h1>",
        f"<p><b>Source:</b> {metrics['source']}</p>",
        f"<p><b>Total samples:</b> {metrics['n_samples']}</p>",
        "<h2>1. Label Distribution</h2>",
        f"<img src='{paths['label_png']}' width='900'/>",
        "<h2>2. Conditional Constraint Violations (raw labels)</h2>",
        "<table><tr><th>Rule</th><th>Count</th></tr>",
    ]
    for k, v in metrics["constraint_violations"].items():
        html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    html.append("</table>")
    html.append("<h2>3. Text Length Statistics</h2>")
    html.append(f"<img src='{paths['len_png']}' width='800'/>")
    html.append("<table><tr><th>Stat</th><th>Value</th></tr>")
    for k, v in metrics["text_length"].items():
        html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    html.append("</table>")
    if metrics.get("company"):
        html.append("<h2>4. Company-level Stats</h2>")
        html.append("<table><tr><th>Stat</th><th>Value</th></tr>")
        for k, v in metrics["company"].items():
            html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        html.append("</table>")
    html.append("</body></html>")
    out.write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/raw/vpesg4k_train_1000 V1.csv",
        help="Path to dataset CSV or JSON",
    )
    parser.add_argument("--out", default=str(REPORT_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, df = load_dataset(args.csv)

    metrics = {
        "source": str(args.csv),
        "n_samples": int(len(df)),
        "label_distribution": _label_distribution(df),
        "constraint_violations": _conditional_consistency(df),
        "text_length": _text_length_stats(df),
        "company": _company_stats(df),
    }

    label_png = out_dir / "label_distribution.png"
    len_png = out_dir / "text_length.png"
    _plot_labels(metrics["label_distribution"], label_png)
    _plot_text_length(df, len_png)

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _render_html(
        metrics,
        {"label_png": label_png.name, "len_png": len_png.name},
        out_dir / "eda_report.html",
    )

    print(f"[EDA] wrote {out_dir / 'metrics.json'}")
    print(f"[EDA] wrote {out_dir / 'eda_report.html'}")
    print(f"[EDA] wrote {label_png} and {len_png}")
    print(f"[EDA] n_samples = {metrics['n_samples']}")
    for field, dist in metrics["label_distribution"].items():
        print(f"  - {field}: {dist}")
    print(f"  - text_length: {metrics['text_length']}")
    print(f"  - constraint_violations: {metrics['constraint_violations']}")


if __name__ == "__main__":
    main()
