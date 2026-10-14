"""Rule-change regression gate — did *only* the intended rows change?

When a settlement rule changes (a channel raises its commission from the 16th, a
promo's funding split is renegotiated), the engine gets edited and the month is re-run.
The question afterwards is never "do the new numbers look plausible?" — it is

    did exactly the rows I meant to change, change, by exactly the amount I meant,
    and did nothing else move?

This module answers that by running three versions of the engine on the same orders:

    before     rule version A (what production ran)
    after      rule version B (the edited engine, as actually implemented)
    predicted  A + the *declared* change only (a ChangeSpec, written by a human,
               separately from the code: "channel_B commission 0.14 -> 0.15 from 01-16")

and classifying every row:

    unchanged        after == before, and nothing was meant to change
    explained        after == predicted != before      (the intended change, landed)
    side_effect      after != before, but the declaration says this row is untouched
    wrong_magnitude  the row was meant to change, and changed by the wrong amount
    missed           the row was meant to change, and did not

Any side_effect / wrong_magnitude / missed row fails the gate.

Two design choices carry the weight:

* **Tolerance is 0.** reconcile.py needs a 400 KRW band because it compares two
  *independent* systems that disagree by rounding noise. Before/after compares the
  *same* deterministic engine to itself — there is no noise, so any band only hides
  real changes. eval_rule_change.py measures what borrowing 1 KRW or 400 KRW costs.
* **The month's data is not enough.** The gate can only see branches the month's
  orders actually exercise. A bug on a rare branch (a high-value platform-funded promo
  on one particular channel) passes silently if no such order happened this month.
  So the gate also runs a **probe grid**: one synthetic order per
  channel x scheme x order-value level x date edge (month start/end and either side of
  each declared effective date). eval_rule_change.py measures that difference too.

Run the three illustrative scenarios:

    python rule_change_gate.py      # exits 1 if any verdict is not the expected one
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ruleset import Override, RuleSet, base_ruleset, compute_payouts, synth_month
from synth_data import SCHEMES

FAILING = ("side_effect", "wrong_magnitude", "missed")
# min / mid / max of the order-value range, plus one off-grid value. Real-looking order
# data in round 500-KRW steps makes every payout a whole number, so a change to the
# rounding rule moves nothing in the month — 33,333 forces a fractional payout on every
# channel with a non-zero rate, so the rounding branch actually runs.
PROBE_GROSS = (12_000, 33_333, 50_000, 89_500)


@dataclass(frozen=True)
class ChangeSpec:
    """The declared, intended rule change — what the reviewer signs off on."""
    description: str
    overrides: tuple

    def apply(self, before: RuleSet) -> RuleSet:
        return before.with_overrides(*self.overrides)

    def scope(self, orders: pd.DataFrame) -> pd.Series:
        m = pd.Series(False, index=orders.index)
        for ov in self.overrides:
            m |= ov.mask(orders)
        return m


@dataclass
class GateResult:
    passed: bool
    rows: pd.DataFrame          # one row per order (month + probe), with a `status`
    eps: int

    def counts(self, source: str | None = None) -> dict:
        r = self.rows if source is None else self.rows[self.rows["source"] == source]
        vc = r["status"].value_counts()
        return {s: int(vc.get(s, 0)) for s in
                ("unchanged", "explained", "side_effect", "wrong_magnitude", "missed")}

    def unexplained(self) -> pd.DataFrame:
        return self.rows[self.rows["status"].isin(FAILING)]

    def unexplained_cells(self, top: int = 5) -> pd.DataFrame:
        """Where the unexplained rows cluster — usually points straight at the bug."""
        u = self.unexplained()
        if u.empty:
            return pd.DataFrame(columns=["channel", "discount_scheme", "status", "rows", "krw"])
        g = (u.assign(krw=(u["after"] - u["predicted"]).abs())
               .groupby(["channel", "discount_scheme", "status"], as_index=False)
               .agg(rows=("order_id", "size"), krw=("krw", "sum"))
               .sort_values(["rows", "krw"], ascending=False))
        return g.head(top)


def classify_rows(before: pd.Series, after: pd.Series, predicted: pd.Series, eps: int = 0) -> np.ndarray:
    moved = (after - before).abs() > eps
    meant = (predicted - before).abs() > eps
    hit = (after - predicted).abs() <= eps
    return np.where(
        hit,
        np.where(meant, "explained", "unchanged"),
        np.where(~meant, "side_effect", np.where(~moved, "missed", "wrong_magnitude")),
    )


def probe_grid(orders: pd.DataFrame, before: RuleSet, spec: ChangeSpec) -> pd.DataFrame:
    """One synthetic order per channel x scheme x value level x date edge.

    Date edges: first and last day of the month, plus the day before and the day of
    every effective date the change declares. Covers every branch the rules can take
    at the boundary values — not every possible value in between.
    """
    d0 = date.fromisoformat(str(orders["order_date"].min()))
    first = d0.replace(day=1)
    last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    dates = {first, last}
    for ov in spec.overrides:
        if ov.from_date:
            eff = date.fromisoformat(ov.from_date)
            dates |= {eff, eff - timedelta(days=1)}
    rows = []
    for ch in before.channels:
        for scheme, (rate, platform_funded) in SCHEMES.items():
            for gross in PROBE_GROSS:
                discount = 3_000 if scheme == "coupon_fixed" else round(gross * rate)
                for d in sorted(dates):
                    rows.append(dict(
                        order_id=f"PROBE-{ch}-{scheme}-{gross}-{d.isoformat()}",
                        store_id="PROBE", channel=ch, order_date=d.isoformat(),
                        gross_amount=gross, discount_scheme=scheme, discount_amount=discount,
                        platform_funded=platform_funded, net_amount=gross - discount,
                    ))
    return pd.DataFrame(rows)


def run_gate(orders: pd.DataFrame, before: RuleSet, after: RuleSet, spec: ChangeSpec,
             eps: int = 0, probe: bool = True) -> GateResult:
    frames = [orders.assign(source="month")]
    if probe:
        frames.append(probe_grid(orders, before, spec).assign(source="probe"))
    data = pd.concat(frames, ignore_index=True)

    a = compute_payouts(data, before)
    b = compute_payouts(data, after)
    p = compute_payouts(data, spec.apply(before))
    rows = data[["order_id", "source", "channel", "discount_scheme", "order_date", "gross_amount"]].copy()
    rows["before"], rows["after"], rows["predicted"] = a, b, p
    rows["status"] = classify_rows(a, b, p, eps)
    passed = not rows["status"].isin(FAILING).any()
    return GateResult(passed=passed, rows=rows, eps=eps)


# --------------------------------------------------------------------- scenarios

def demo_scenarios(base: RuleSet):
    """The three cases the gate exists for. Returns (name, spec, after, expect_pass)."""
    ch, eff = "channel_B", "2026-01-16"
    old = base.channels[ch].commission_rate
    new = round(old + 0.01, 4)
    intended = Override("commission_rate", new, channel=ch, from_date=eff)
    spec = ChangeSpec(f"{ch} commission {old:g} -> {new:g} from {eff}", (intended,))

    return [
        ("1. intended change only", spec,
         base.with_overrides(intended), True),
        # copy-paste while editing: channel_B's burden lands on channel_A as well
        ("2. intended change + side effect (copy-paste into channel_A)", spec,
         base.with_overrides(intended, Override(
             "merchant_burden", base.channels[ch].merchant_burden, channel="channel_A")), False),
        ("3. scope too wide (effective date ignored)", spec,
         base.with_overrides(Override("commission_rate", new, channel=ch)), False),
    ]


def format_result(name: str, spec: ChangeSpec, res: GateResult, expect_pass: bool) -> str:
    verdict = "PASS" if res.passed else "FAIL"
    ok = "as expected" if res.passed == expect_pass else "UNEXPECTED"
    cm, cp = res.counts("month"), res.counts("probe")
    lines = [f"\n{name}",
             f"  declared : {spec.description}",
             f"  verdict  : {verdict}  ({ok})",
             "  month    : " + "  ".join(f"{k}={v}" for k, v in cm.items()),
             "  probe    : " + "  ".join(f"{k}={v}" for k, v in cp.items())]
    if not res.passed:
        u = res.unexplained()
        krw = int((u["after"] - u["predicted"]).abs().sum())
        lines.append(f"  unexplained: {len(u)} rows, {krw:,} KRW not covered by the declared change")
        for _, c in res.unexplained_cells(3).iterrows():
            lines.append(f"    {c['channel']:16s} {c['discount_scheme']:13s} {c['status']:16s} "
                         f"rows={c['rows']:<4d} krw={int(c['krw']):,}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=2000, help="orders in the synthetic month")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    base = base_ruleset()
    orders = synth_month(a.seed, a.n)
    print("=" * 72)
    print(f" Rule-change regression gate - {len(orders)} synthetic orders, 2026-01, tolerance 0 KRW")
    print("=" * 72)
    wrong = 0
    for name, spec, after, expect_pass in demo_scenarios(base):
        res = run_gate(orders, base, after, spec)
        print(format_result(name, spec, res, expect_pass))
        wrong += res.passed != expect_pass
    print("=" * 72)
    if wrong:
        print(f"FAIL: {wrong} scenario verdict(s) did not match expectation")
        raise SystemExit(1)
    print(" all three verdicts as expected")


if __name__ == "__main__":
    main()
