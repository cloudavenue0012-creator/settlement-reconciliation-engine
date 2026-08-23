"""Generate a fully synthetic multi-channel settlement dataset.

Outputs (to --out dir):
  orders.csv              customer orders (gross, discount, net) across channels
  official_settlement.csv the channel-reported payout, with PLANTED discrepancies
  ground_truth.csv        anomaly labels for the planted discrepancies (eval only)

No real data: order values, stores, and channel rules are all random/illustrative.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import yaml
from faker import Faker

from settlement_rules import ChannelRule, expected_payout

SCHEMES = {          # discount scheme -> (rate on gross, is_platform_funded)
    "none":         (0.00, False),
    "promo_10":     (0.10, False),
    "promo_20":     (0.20, False),
    "platform_25":  (0.25, True),   # platform-funded promo: merchant should NOT be charged for it
    "coupon_fixed": (0.00, False),  # fixed-won coupon, amount set below
}

# discrepancy types planted into the official settlement (these are the anomalies to catch)
ANOM_FEE = "fee_error"          # channel billed a wrong commission rate
ANOM_BURDEN = "burden_error"    # channel charged merchant for a platform-funded promo
ANOM_AMOUNT = "amount_mismatch" # unexplained payout error
ANOM_MISSING = "missing"        # order absent from the settlement file
ANOM_DUP = "duplicate"          # order double-counted


def load_rules(path: Path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules = {k: ChannelRule(**v) for k, v in raw["channels"].items()}
    return rules, int(raw["tolerance_krw"])


def gen_orders(n: int, rules, fake: Faker) -> pd.DataFrame:
    stores = [f"BR-{i:03d}" for i in range(1, 41)]
    rows = []
    for i in range(n):
        ch = random.choice(list(rules))
        scheme = random.choice(list(SCHEMES))
        gross = random.randrange(12_000, 90_000, 500)
        rate, platform_funded = SCHEMES[scheme]
        discount = 3_000 if scheme == "coupon_fixed" else round(gross * rate)
        rows.append(
            dict(
                order_id=f"ORD{i:06d}",
                store_id=random.choice(stores),
                channel=ch,
                order_date=fake.date_between("-30d", "today").isoformat(),
                gross_amount=gross,
                discount_scheme=scheme,
                discount_amount=discount,
                platform_funded=platform_funded,
                net_amount=gross - discount,
            )
        )
    return pd.DataFrame(rows)


def build_official(orders: pd.DataFrame, rules, anomaly_rate: float, tol: int):
    """Return (official_settlement_df, ground_truth_df).

    'official' starts as the correct expected payout, then we inject realistic
    noise and planted anomalies. Noise is *mostly* sub-tolerance, but ~2% of rows
    drift past tolerance (heavy tail) and some amount errors are tiny -- so the
    detector faces a genuine precision/recall trade-off, not a rigged 100%.
    """
    official, truth = [], []
    for o in orders.to_dict("records"):
        rule = rules[o["channel"]]
        # merchant is not supposed to fund platform-funded promos
        eff_discount = 0 if o["platform_funded"] else o["discount_amount"]
        correct = expected_payout(o["gross_amount"], eff_discount, rule)

        if random.random() < 0.02:                    # heavy-tailed rounding -> some false positives
            payout = correct + random.choice([-1, 1]) * random.randint(tol + 10, tol * 3)
        else:
            payout = correct + random.randint(-20, 20)  # ordinary sub-tolerance noise
        label, present, dup = "ok", True, False

        if random.random() < anomaly_rate:
            kind = random.choice([ANOM_FEE, ANOM_BURDEN, ANOM_AMOUNT, ANOM_MISSING, ANOM_DUP])
            label = kind
            if kind == ANOM_FEE:
                bad = ChannelRule(rule.commission_rate + 0.03, rule.merchant_burden, rule.settlement_lag_days)
                payout = expected_payout(o["gross_amount"], eff_discount, bad)
            elif kind == ANOM_BURDEN:
                # charge merchant for the full discount even though it was platform-funded
                payout = expected_payout(o["gross_amount"], o["discount_amount"], rule)
                if not o["platform_funded"]:      # only a real error when it *was* platform-funded
                    payout = correct - random.randint(1_500, 6_000)
            elif kind == ANOM_AMOUNT:
                # both directions; ~30% land in the gray zone near tolerance, so no single
                # threshold catches everything -> the tuning curve peaks below F1=1.0
                mag = random.randint(30, 400) if random.random() < 0.30 else random.randint(400, 15_000)
                payout = correct + random.choice([-1, 1]) * mag
            elif kind == ANOM_MISSING:
                present = False
            elif kind == ANOM_DUP:
                dup = True

        if present:
            official.append(dict(order_id=o["order_id"], channel=o["channel"], official_payout=payout))
            if dup:
                official.append(dict(order_id=o["order_id"], channel=o["channel"], official_payout=payout))
        truth.append(dict(order_id=o["order_id"], anomaly=label))

    return pd.DataFrame(official), pd.DataFrame(truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2000, help="number of orders")
    ap.add_argument("--anomaly-rate", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()

    random.seed(a.seed)
    fake = Faker("ko_KR")
    fake.seed_instance(a.seed)

    rules, tol = load_rules(Path(__file__).with_name("rules.yaml"))
    orders = gen_orders(a.n, rules, fake)
    official, truth = build_official(orders, rules, a.anomaly_rate, tol)

    a.out.mkdir(parents=True, exist_ok=True)
    orders.to_csv(a.out / "orders.csv", index=False, encoding="utf-8-sig")
    official.to_csv(a.out / "official_settlement.csv", index=False, encoding="utf-8-sig")
    truth.to_csv(a.out / "ground_truth.csv", index=False, encoding="utf-8-sig")

    planted = (truth.anomaly != "ok").sum()
    print(f"[synth] {len(orders)} orders, {len(official)} settlement rows, "
          f"{planted} planted anomalies ({planted/len(orders):.1%}) -> {a.out}/")


if __name__ == "__main__":
    main()
