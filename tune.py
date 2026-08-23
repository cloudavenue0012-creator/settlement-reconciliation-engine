"""Tolerance tuning — turn the precision/recall trade-off into a chosen threshold.

Sweeps the reconciliation match tolerance, scores anomaly detection against the
planted ground truth at each level, and plots precision / recall / F1 vs tolerance.
The peak-F1 point is the recommended operating tolerance. Structural anomalies
(missing / duplicate rows) are tolerance-independent; the curve moves because of the
amount-based checks, which is exactly the knob worth tuning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reconcile import reconcile


def score(df: pd.DataFrame, truth: pd.DataFrame):
    m = df.merge(truth, on="order_id")
    ta, pa = m["anomaly"] != "ok", m["is_anomaly"]
    tp = int((pa & ta).sum())
    fp = int((pa & ~ta).sum())
    fn = int((~pa & ta).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("docs/tuning_curve.png"))
    a = ap.parse_args()
    here = Path(__file__).parent

    truth = pd.read_csv(a.data / "ground_truth.csv")
    rows = []
    for t in range(10, 501, 10):
        df = reconcile(a.data / "orders.csv", a.data / "official_settlement.csv",
                       here / "rules.yaml", tol_override=t)
        rows.append((t, *score(df, truth)))
    res = pd.DataFrame(rows, columns=["tol", "precision", "recall", "f1"])
    best = res.loc[res.f1.idxmax()]

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(res.tol, res.recall, color="#d9663e", lw=1.8, label="Recall")
    ax.plot(res.tol, res.precision, color="#2b6cb0", lw=1.8, label="Precision")
    ax.plot(res.tol, res.f1, color="#1c2733", lw=2.6, label="F1")
    ax.axvline(best.tol, color="#9aa5b1", ls="--", lw=1)
    ax.scatter([best.tol], [best.f1], color="#1c2733", zorder=5)
    ax.annotate(f"best F1 = {best.f1:.2f}\n@ tol = {int(best.tol)} KRW",
                (best.tol, best.f1), textcoords="offset points", xytext=(12, -4), fontsize=9)
    ax.set_xlabel("Match tolerance (KRW)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", color="#eef1f5")
    ax.set_title("Anomaly detection vs. match tolerance", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower center", ncol=3)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)

    print(res.iloc[::3].to_string(index=False))
    print(f"\n[tune] best F1 = {best.f1:.3f} at tolerance = {int(best.tol)} KRW  ->  {a.out}")


if __name__ == "__main__":
    main()
