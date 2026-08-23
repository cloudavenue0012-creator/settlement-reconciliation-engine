"""Explanation layer + faithfulness eval.

For each flagged anomaly, produce a grounded, natural-language explanation of *why*
it was flagged. Then score the explanation's **faithfulness**: every monetary figure
it cites must be present in (or derivable from) the reconciliation record. This is the
RAGAS-style guard that catches a generator hallucinating a number.

Explanations here come from a deterministic, rule-grounded template (runs with no API
key, so faithfulness is 1.0 by construction). The point of shipping the *scorer* is that
it stays valid when you swap in an LLM to rephrase — and the `--demo-hallucination` flag
shows the scorer actually catches an invented figure rather than rubber-stamping 1.0.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _won(x) -> str:
    return f"{int(round(x)):,} KRW"


def explain(r: pd.Series) -> str:
    """Grounded explanation string built only from fields in the record r."""
    ch, flag = r["channel"], r["flag"]
    if flag == "missing":
        return (f"Order {r['order_id']} ({ch}) has no row in the settlement file, "
                f"but the merchant was owed {_won(r['expected_payout'])}. Likely a dropped payout.")
    if flag == "duplicate":
        return (f"Order {r['order_id']} ({ch}) appears {int(r['official_rows'])} times in the "
                f"settlement file (expected once), double-counting the payout.")
    exp, off, diff = r["expected_payout"], r["official_payout"], r["diff"]
    direction = "under-paid" if diff < 0 else "over-paid"
    head = (f"Order {r['order_id']} ({ch}) was {direction} by {_won(abs(diff))}: "
            f"expected {_won(exp)}, settled {_won(off)}.")
    if flag == "fee_error":
        return head + " The gap matches a commission rate higher than the contracted one."
    if flag == "burden_error":
        return head + " The merchant was charged for a platform-funded promo they should not fund."
    return head + " No rule change explains the gap - flagged for manual review."


NUM = re.compile(r"(\d[\d,]*)")


def cited_numbers(text: str) -> set[int]:
    return {int(m.replace(",", "")) for m in NUM.findall(text)}


def grounded_numbers(r: pd.Series) -> set[int]:
    """Facts the explanation is allowed to cite (record fields + trivially derived)."""
    vals = set()
    for c in ("expected_payout", "official_payout", "gross_amount", "discount_amount",
              "diff", "official_rows", "order_id"):
        v = r.get(c)
        if pd.notna(v):
            try:
                vals.add(int(round(float(v))))
                vals.add(abs(int(round(float(v)))))
            except (ValueError, TypeError):
                digits = re.sub(r"\D", "", str(v))   # order_id like ORD000123 -> 123
                if digits:
                    vals.add(int(digits))
    return vals


def faithfulness(text: str, r: pd.Series):
    cited = cited_numbers(text)
    grounded = grounded_numbers(r)
    ungrounded = {n for n in cited if n not in grounded}
    score = 1.0 if not cited else (len(cited) - len(ungrounded)) / len(cited)
    return score, ungrounded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--show", type=int, default=5, help="example explanations to print")
    ap.add_argument("--demo-hallucination", action="store_true",
                    help="inject a fake figure into one explanation to prove the scorer catches it")
    a = ap.parse_args()

    df = pd.read_csv(a.data / "reconciliation.csv")
    anom = df[df["is_anomaly"]].copy()

    scores = []
    print(f"=== Explanations for {len(anom)} flagged anomalies (showing {a.show}) ===")
    for i, (_, r) in enumerate(anom.iterrows()):
        text = explain(r)
        s, bad = faithfulness(text, r)
        scores.append(s)
        if i < a.show:
            print(f"\n[{r['flag']}] {text}\n  faithfulness={s:.2f}")

    mean = sum(scores) / len(scores) if scores else 1.0
    print(f"\nmean faithfulness over template explanations: {mean:.3f}  (1.0 = no ungrounded figures)")

    if a.demo_hallucination and len(anom):
        r = anom.iloc[0]
        good = explain(r)
        bad_text = good + " Total leakage this month was 9,999,999 KRW."   # invented figure
        s, bad = faithfulness(bad_text, r)
        print("\n--- demo: hallucinated figure injected ---")
        print(bad_text)
        print(f"  faithfulness={s:.2f}  ungrounded figures caught: {sorted(bad)}")


if __name__ == "__main__":
    main()
