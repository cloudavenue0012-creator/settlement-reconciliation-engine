"""Reconciliation engine.

Independently reconstructs the expected payout for every order from the orders +
channel rules, then matches it against the channel's official settlement file and
flags anomalies (fee errors, wrong discount burden, missing / duplicated rows,
unexplained amount gaps). This is the core "verification gate": we never trust the
channel's number — we recompute and diff.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from settlement_rules import ChannelRule, expected_payout
from synth_data import SCHEMES, load_rules


def reconstruct_expected(orders: pd.DataFrame, rules) -> pd.DataFrame:
    def _expect(o):
        rule = rules[o["channel"]]
        eff_discount = 0 if o["platform_funded"] else o["discount_amount"]
        return expected_payout(o["gross_amount"], eff_discount, rule)

    out = orders.copy()
    out["expected_payout"] = out.apply(_expect, axis=1)
    return out


def reconcile(orders_path: Path, official_path: Path, rules_path: Path,
              tol_override: int | None = None) -> pd.DataFrame:
    rules, tol = load_rules(rules_path)
    if tol_override is not None:
        tol = tol_override
    orders = pd.read_csv(orders_path)
    official = pd.read_csv(official_path)

    expected = reconstruct_expected(orders, rules)

    # collapse duplicates but remember the count so we can flag double-counting
    counts = official.groupby("order_id").size().rename("official_rows")
    off = official.groupby("order_id", as_index=False)["official_payout"].sum().merge(counts, on="order_id")

    df = expected.merge(off, on="order_id", how="left")
    df["official_rows"] = df["official_rows"].fillna(0).astype(int)
    df["present"] = df["official_rows"] > 0
    df["diff"] = df["official_payout"] - df["expected_payout"]

    def classify(r):
        if not r["present"]:
            return "missing"
        if r["official_rows"] > 1:
            return "duplicate"
        if abs(r["diff"]) <= tol:
            return "ok"
        # heuristic split between a clean fee error and everything else
        fee_only = expected_payout(
            r["gross_amount"],
            0 if r["platform_funded"] else r["discount_amount"],
            ChannelRule(rules[r["channel"]].commission_rate + 0.03,
                        rules[r["channel"]].merchant_burden, 0),
        )
        if abs(r["official_payout"] - fee_only) <= tol:
            return "fee_error"
        if r["platform_funded"] and r["diff"] < 0:
            return "burden_error"
        return "amount_mismatch"

    df["flag"] = df.apply(classify, axis=1)
    df["is_anomaly"] = df["flag"] != "ok"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    a = ap.parse_args()
    here = Path(__file__).parent
    df = reconcile(a.data / "orders.csv", a.data / "official_settlement.csv", here / "rules.yaml")

    df.to_csv(a.data / "reconciliation.csv", index=False, encoding="utf-8-sig")
    n_anom = int(df["is_anomaly"].sum())
    gap = int(df.loc[df["present"], "diff"].abs().sum())
    print(f"[reconcile] {len(df)} orders | flagged {n_anom} anomalies | abs payout gap {gap:,} KRW")
    print(df["flag"].value_counts().to_string())


if __name__ == "__main__":
    main()
