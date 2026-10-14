"""Eval for the rule-change regression gate: how many broken rule changes does it catch?

Each scenario draws a synthetic month and a random *intended* change (one channel's
commission or merchant burden, effective from a random day mid-month), then builds the
edited engine either correctly or with exactly one planted defect:

    none                 the change, implemented as declared          (must PASS)
    side_effect_channel  + another channel's burden edited by mistake
    rounding_refactor    + rounding switched half-even -> floor
    scope_all_channels   the change applied to every channel
    scope_whole_month    the change applied from the 1st, effective date ignored
    wrong_magnitude      the change applied at 1.5x the declared delta
    too_narrow           the change applied to one discount scheme only
    latent_rare_branch   + high-value platform-funded promos on one channel
                           charged to the merchant (a branch most months never reach)
    latent_off_grid      the same bug, but only for orders of 70,000-70,999 KRW — a band
                           none of the probe grid's values land in. Kept on purpose: the
                           probe grid covers the values it was designed to cover, no more.

Ground truth is the scenario label: a defect was planted or it was not. Scoring is
scenario-level (did the check fail?), never row-level — a row-level score would compare
the gate's after-vs-predicted diff with the same diff computed again, which is 100% by
construction and measures nothing.

Every detector sees the same before / after / declared change:

    gate         row-level, tolerance 0, the month's orders only
    gate+probe   the same, plus the probe grid (rule_change_gate.probe_grid)
    +probe tol=1    gate+probe with a 1 KRW "float safety" band
    +probe tol=400  gate+probe borrowing the reconciliation tolerance from rules.yaml
                    (the two tol columns isolate the tolerance effect from data coverage)
    totals       month total moved by the declared amount, within 0.1% of the total
    scope-only   every row inside the declared scope matches the prediction

CI gates on gate+probe: --min-detect over all defective scenarios, --max-false-alarm
over the clean ones.

    python eval_rule_change.py                       # 50 scenarios per kind
    python eval_rule_change.py --min-detect 1.0 --max-false-alarm 0.0
"""
from __future__ import annotations

import argparse
import random
from datetime import date

import pandas as pd

from rule_change_gate import ChangeSpec, run_gate
from ruleset import Override, base_ruleset, compute_payouts, synth_month

KINDS = ["none", "side_effect_channel", "rounding_refactor", "scope_all_channels",
         "scope_whole_month", "wrong_magnitude", "too_narrow", "latent_rare_branch",
         "latent_off_grid"]
DETECTORS = ["gate", "gate+probe", "+probe tol=1", "+probe tol=400", "totals", "scope-only"]
RARE_CHANNEL, RARE_SCHEME, RARE_MIN_GROSS = "channel_direct", "platform_25", 88_000
OFF_GRID = (70_000, 71_000)      # an order-value band no probe value lands in
TOTALS_REL_TOL = 0.001


def build_scenario(seed: int, kind: str, n: int):
    rng = random.Random(seed * 7_919 + KINDS.index(kind))
    base = base_ruleset()
    orders = synth_month(seed, n)
    channels = list(base.channels)

    ch = rng.choice(channels)
    fld = rng.choice(["commission_rate", "merchant_burden"])
    old = getattr(base.channels[ch], fld)
    if fld == "commission_rate":
        delta = rng.choice([-0.02, -0.01, 0.01, 0.02])
        if old + delta < 0:
            delta = -delta
    else:
        delta = rng.choice([d for d in (-0.2, -0.1, 0.1, 0.2) if 0 <= old + d <= 1])
    new = round(old + delta, 4)
    eff = date(2026, 1, rng.randint(5, 25)).isoformat()
    intended = Override(fld, new, channel=ch, from_date=eff)
    spec = ChangeSpec(f"{ch} {fld} {old:g} -> {new:g} from {eff}", (intended,))

    after = base.with_overrides(intended)
    if kind == "side_effect_channel":
        other = rng.choice([c for c in channels if c != ch])
        ob = base.channels[other].merchant_burden
        ob2 = round(ob - 0.1, 4) if ob >= 0.1 else round(ob + 0.1, 4)
        after = after.with_overrides(Override("merchant_burden", ob2, channel=other))
    elif kind == "rounding_refactor":
        after = after.with_rounding("floor")
    elif kind == "scope_all_channels":
        after = base.with_overrides(Override(fld, new, from_date=eff))
    elif kind == "scope_whole_month":
        after = base.with_overrides(Override(fld, new, channel=ch))
    elif kind == "wrong_magnitude":
        after = base.with_overrides(Override(fld, round(old + 1.5 * delta, 4), channel=ch, from_date=eff))
    elif kind == "too_narrow":
        after = base.with_overrides(Override(fld, new, channel=ch, from_date=eff, scheme="promo_10"))
    elif kind == "latent_rare_branch":
        after = after.with_overrides(Override("charge_platform_promo", 1, channel=RARE_CHANNEL,
                                              scheme=RARE_SCHEME, min_gross=RARE_MIN_GROSS))
    elif kind == "latent_off_grid":
        after = after.with_overrides(Override("charge_platform_promo", 1, channel=RARE_CHANNEL,
                                              scheme=RARE_SCHEME, min_gross=OFF_GRID[0],
                                              max_gross=OFF_GRID[1]))
    return orders, base, after, spec


def detect_all(orders, before, after, spec) -> dict:
    out = {}
    out["gate"] = not run_gate(orders, before, after, spec, eps=0, probe=False).passed
    out["gate+probe"] = not run_gate(orders, before, after, spec, eps=0, probe=True).passed
    out["+probe tol=1"] = not run_gate(orders, before, after, spec, eps=1, probe=True).passed
    out["+probe tol=400"] = not run_gate(orders, before, after, spec, eps=400, probe=True).passed

    a = compute_payouts(orders, before)
    b = compute_payouts(orders, after)
    p = compute_payouts(orders, spec.apply(before))
    out["totals"] = abs(int(b.sum()) - int(p.sum())) > TOTALS_REL_TOL * int(a.sum())
    scope = spec.scope(orders)
    out["scope-only"] = bool(((b - p)[scope] != 0).any())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenarios", type=int, default=50, help="scenarios per defect kind")
    ap.add_argument("-n", type=int, default=400, help="orders per synthetic month")
    ap.add_argument("--min-detect", type=float, default=None,
                    help="fail (exit 1) if gate+probe catches fewer defective scenarios than this")
    ap.add_argument("--max-false-alarm", type=float, default=None,
                    help="fail (exit 1) if gate+probe fails more clean scenarios than this")
    a = ap.parse_args()

    rows = []
    for kind in KINDS:
        for s in range(a.scenarios):
            orders, before, after, spec = build_scenario(s, kind, a.n)
            d = detect_all(orders, before, after, spec)
            rows.append({"kind": kind, **d})
    res = pd.DataFrame(rows)
    rate = res.groupby("kind", sort=False)[DETECTORS].mean().reindex(KINDS)

    w = 21
    print("=" * 113)
    print(f" Rule-change gate - eval ({a.scenarios} scenarios per kind, {a.n} orders/month, synthetic)")
    print("=" * 113)
    print(f" {'planted defect':{w}s}" + "".join(f"{d:>15s}" for d in DETECTORS))
    print("-" * 113)
    for kind in KINDS:
        label = "none (false alarm)" if kind == "none" else kind
        print(f" {label:{w}s}" + "".join(f"{rate.loc[kind, d]:>15.0%}" for d in DETECTORS))
    defective = res[res.kind != "none"]
    print("-" * 113)
    print(f" {'all defects (detect)':{w}s}" + "".join(f"{defective[d].mean():>15.1%}" for d in DETECTORS))
    print("=" * 113)

    detect = float(defective["gate+probe"].mean())
    false_alarm = float(res.loc[res.kind == "none", "gate+probe"].mean())
    print(f" gate+probe: side-effect detection {detect:.1%} "
          f"({int(defective['gate+probe'].sum())}/{len(defective)}), "
          f"false alarms {false_alarm:.1%}")

    failures = []
    if a.min_detect is not None and detect < a.min_detect:
        failures.append(f"detection {detect:.3f} < {a.min_detect}")
    if a.max_false_alarm is not None and false_alarm > a.max_false_alarm:
        failures.append(f"false alarm {false_alarm:.3f} > {a.max_false_alarm}")
    if failures:
        print("FAIL: " + "; ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
