"""Evaluation harness — the differentiator.

Scores the reconciler against the planted ground truth:
  reconciliation_accuracy  share of orders settled within tolerance of expected
  precision / recall / F1   anomaly detection quality (any-anomaly, one-vs-rest)
  false_positive_rate       clean orders wrongly flagged
  coverage                  orders present in the settlement file
  per-type recall           how well each discrepancy type is caught
Also reports the KRW payout gap recovered — the money reconciliation puts back.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reconcile import reconcile


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--min-f1", type=float, default=None, help="fail (exit 1) if F1 below this")
    ap.add_argument("--min-recall", type=float, default=None, help="fail (exit 1) if recall below this")
    a = ap.parse_args()
    here = Path(__file__).parent

    df = reconcile(a.data / "orders.csv", a.data / "official_settlement.csv", here / "rules.yaml")
    truth = pd.read_csv(a.data / "ground_truth.csv")
    m = df.merge(truth, on="order_id")
    m["true_anomaly"] = m["anomaly"] != "ok"

    tp = int((m.is_anomaly & m.true_anomaly).sum())
    fp = int((m.is_anomaly & ~m.true_anomaly).sum())
    fn = int((~m.is_anomaly & m.true_anomaly).sum())
    tn = int((~m.is_anomaly & ~m.true_anomaly).sum())
    p, r, f = prf(tp, fp, fn)

    clean = m[~m.true_anomaly]
    fpr = (clean.is_anomaly.sum() / len(clean)) if len(clean) else 0.0
    within_tol = m[m.present & ~m.true_anomaly]
    recon_acc = (within_tol.flag == "ok").mean() if len(within_tol) else 0.0
    coverage = m.present.mean()
    gap_total = int(m.loc[m.present, "diff"].abs().sum())
    gap_caught = int(m.loc[m.is_anomaly & m.present, "diff"].abs().sum())

    print("=" * 52)
    print(" Settlement Reconciliation - Eval")
    print("=" * 52)
    print(f" reconciliation_accuracy : {recon_acc:6.1%}   (clean orders within tolerance)")
    print(f" anomaly precision       : {p:6.1%}")
    print(f" anomaly recall          : {r:6.1%}")
    print(f" anomaly F1              : {f:6.3f}")
    print(f" false_positive_rate     : {fpr:6.1%}")
    print(f" coverage                : {coverage:6.1%}")
    print(f" confusion               : TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f" payout gap (KRW)        : {gap_caught:,} caught / {gap_total:,} total")
    print("-" * 52)
    print(" per-type recall:")
    for kind in ["fee_error", "burden_error", "amount_mismatch", "missing", "duplicate"]:
        sub = m[m.anomaly == kind]
        rec = sub.is_anomaly.mean() if len(sub) else float("nan")
        print(f"   {kind:16s} {rec:6.1%}  (n={len(sub)})")
    print("=" * 52)

    # CI gate: fail the build if quality regresses below thresholds
    failures = []
    if a.min_f1 is not None and f < a.min_f1:
        failures.append(f"F1 {f:.3f} < {a.min_f1}")
    if a.min_recall is not None and r < a.min_recall:
        failures.append(f"recall {r:.3f} < {a.min_recall}")
    if failures:
        print("FAIL: " + "; ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
