"""Tolerance tuning — turn the precision/recall trade-off into a *chosen* threshold.

The match tolerance is the one free parameter in the whole engine: below it a gap is
called rounding noise, above it an anomaly. Guessing it is the difference between a
reconciliation that cries wolf and one that quietly loses money.

This sweeps the tolerance, scores every value against ground truth, and reports the
two operating points that matter:

  max-F1        best balance of precision and recall
  max-recall    the widest tolerance that still catches *every* planted anomaly

They are not the same point, and for settlement work the second one wins: a missed
discrepancy is cash permanently lost, a false positive costs a minute of someone's
attention. The engine therefore ships at max-recall even though max-F1 scores higher.

Run:  python tune.py                 # after synth_data.py
      python tune.py --max 800 --step 10 --csv curve.csv
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
    """Precision / recall / F1 of anomaly detection against ground truth.

    Kept as a 3-tuple because app.py imports it for the live dashboard.
    """
    m = df.merge(truth, on="order_id")
    ta, pa = m["anomaly"] != "ok", m["is_anomaly"]
    tp = int((pa & ta).sum())
    fp = int((pa & ~ta).sum())
    fn = int((~pa & ta).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_at(data: Path, rules: Path, tol: int, truth: pd.DataFrame) -> dict:
    """Full scorecard at one tolerance, including the money actually recovered."""
    df = reconcile(data / "orders.csv", data / "official_settlement.csv", rules, tol_override=tol)
    p, r, f = score(df, truth)

    m = df.merge(truth, on="order_id")
    ta, pa = m["anomaly"] != "ok", m["is_anomaly"]
    caught = int(m.loc[pa & m["present"], "diff"].abs().sum())
    total = int(m.loc[m["present"], "diff"].abs().sum())

    return {"tolerance": tol, "precision": p, "recall": r, "f1": f,
            "fp": int((pa & ~ta).sum()), "fn": int((~pa & ta).sum()),
            "gap_caught": caught, "gap_total": total}


def bar(value: float, width: int = 20) -> str:
    """Inline bar so the curve is readable in a terminal, not just in the PNG."""
    return "#" * int(round(value * width)) + "." * (width - int(round(value * width)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--min", type=int, default=10)
    ap.add_argument("--max", type=int, default=800)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("docs/tuning_curve.png"))
    ap.add_argument("--csv", type=Path, default=None, help="also write the curve to CSV")
    a = ap.parse_args()
    here = Path(__file__).parent

    truth = pd.read_csv(a.data / "ground_truth.csv")
    rows = [score_at(a.data, here / "rules.yaml", t, truth)
            for t in range(a.min, a.max + 1, a.step)]
    res = pd.DataFrame(rows)

    best_f1 = max(rows, key=lambda r: r["f1"])
    full_recall = [r for r in rows if r["recall"] >= 1.0]
    chosen = max(full_recall, key=lambda r: r["tolerance"]) if full_recall else None

    # ---------------------------------------------------------------- terminal
    print("=" * 76)
    print(" Tolerance tuning curve")
    print("=" * 76)
    print(f"{'tol(KRW)':>9} {'prec':>7} {'recall':>7} {'F1':>7} {'FP':>4} {'FN':>4}  precision")
    print("-" * 76)
    for r in rows[::5]:
        print(f"{r['tolerance']:>9} {r['precision']:>7.1%} {r['recall']:>7.1%} "
              f"{r['f1']:>7.3f} {r['fp']:>4} {r['fn']:>4}  {bar(r['precision'])}")
    print("-" * 76)
    print(f" max-F1        tol={best_f1['tolerance']:>4}  F1={best_f1['f1']:.3f}  "
          f"precision={best_f1['precision']:.1%}  recall={best_f1['recall']:.1%}")
    if chosen:
        pct = chosen["gap_caught"] / chosen["gap_total"] if chosen["gap_total"] else 0
        print(f" max-recall    tol={chosen['tolerance']:>4}  F1={chosen['f1']:.3f}  "
              f"precision={chosen['precision']:.1%}  recall=100.0%   <- shipped")
        print(f"               -> recovers {chosen['gap_caught']:,} of {chosen['gap_total']:,} KRW "
              f"({pct:.1%}), at the cost of {chosen['fp']} false positives to review")
    else:
        print(" max-recall    no tolerance in this range reaches 100% recall")
    print("=" * 76)
    print(" Shipped operating point = max-recall. In settlement reconciliation a false")
    print(" negative is cash permanently lost; a false positive costs one review.")
    print(" Optimising F1 trades money for tidiness - the wrong direction here.")
    print("=" * 76)

    # -------------------------------------------------------------------- plot
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(res.tolerance, res.recall, color="#d9663e", lw=1.8, label="Recall")
    ax.plot(res.tolerance, res.precision, color="#2b6cb0", lw=1.8, label="Precision")
    ax.plot(res.tolerance, res.f1, color="#1c2733", lw=2.6, label="F1")

    ax.axvline(best_f1["tolerance"], color="#c9d2dc", ls=":", lw=1)
    ax.annotate(f"max F1 = {best_f1['f1']:.3f}\n@ {best_f1['tolerance']} KRW",
                (best_f1["tolerance"], best_f1["f1"]), textcoords="offset points",
                xytext=(8, -28), fontsize=8, color="#6b7684")
    if chosen:
        ax.axvline(chosen["tolerance"], color="#1c2733", ls="--", lw=1.2)
        ax.scatter([chosen["tolerance"]], [chosen["f1"]], color="#1c2733", zorder=5)
        ax.annotate(f"shipped: {chosen['tolerance']} KRW\nrecall 100%, F1 {chosen['f1']:.3f}",
                    (chosen["tolerance"], chosen["f1"]), textcoords="offset points",
                    xytext=(-116, 14), fontsize=9, fontweight="bold")

    ax.set_xlabel("Match tolerance (KRW)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", color="#eef1f5")
    ax.set_title("Anomaly detection vs. match tolerance", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower center", ncol=3)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"[tune] curve -> {a.out}")

    if a.csv:
        res.to_csv(a.csv, index=False, encoding="utf-8-sig")
        print(f"[tune] data  -> {a.csv}")


if __name__ == "__main__":
    main()
