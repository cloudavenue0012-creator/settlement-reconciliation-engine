"""Tests for the versioned engine (ruleset.py) and the rule-change regression gate.

1. Version A of the vectorised engine must equal the shipped reconciler row for row —
   otherwise the gate would be checking a different engine from the one measured.
2. Every row status the gate can emit, pinned on hand-built frames.
3. The three named scenarios produce the verdicts the README claims.
4. The coverage and tolerance findings from eval_rule_change.py, pinned small.
"""
from pathlib import Path

import pandas as pd

import synth_data
from reconcile import reconstruct_expected
from rule_change_gate import ChangeSpec, classify_rows, demo_scenarios, probe_grid, run_gate
from ruleset import Override, base_ruleset, compute_payouts, synth_month
from synth_data import generate, load_rules

BASE = base_ruleset()


# ---- 1. the vectorised engine is the shipped engine ---------------------------

def test_base_ruleset_matches_reconciler_row_for_row():
    orders, _official, _truth = generate(n=1000, anomaly_rate=0.08, seed=42)
    rules, _ = load_rules(Path(synth_data.__file__).with_name("rules.yaml"))
    shipped = reconstruct_expected(orders, rules)["expected_payout"].astype("int64")
    assert (compute_payouts(orders, BASE) == shipped).all()


def test_override_filters_and_precedence():
    orders = pd.DataFrame([
        dict(order_id="a", channel="channel_A", discount_scheme="none", order_date="2026-01-10",
             gross_amount=10_000, discount_amount=0, platform_funded=False),
        dict(order_id="b", channel="channel_A", discount_scheme="none", order_date="2026-01-20",
             gross_amount=10_000, discount_amount=0, platform_funded=False),
        dict(order_id="c", channel="channel_B", discount_scheme="none", order_date="2026-01-20",
             gross_amount=10_000, discount_amount=0, platform_funded=False),
    ])
    ov = Override("commission_rate", 0.20, channel="channel_A", from_date="2026-01-15")
    assert list(ov.mask(orders)) == [False, True, False]
    later_wins = BASE.with_overrides(ov, Override("commission_rate", 0.30, channel="channel_A"))
    assert list(compute_payouts(orders, later_wins))[:2] == [7_000, 7_000]


def test_synth_month_is_deterministic_and_pinned_to_its_month():
    a, b = synth_month(7, 300), synth_month(7, 300)
    pd.testing.assert_frame_equal(a, b)
    assert a["order_date"].min() >= "2026-01-01" and a["order_date"].max() <= "2026-01-31"


# ---- 2. every row status ------------------------------------------------------

def test_classify_rows_covers_every_status():
    before = pd.Series([100, 100, 100, 100, 100])
    predicted = pd.Series([100, 110, 100, 110, 110])
    after = pd.Series([100, 110, 105, 120, 100])
    assert list(classify_rows(before, after, predicted)) == [
        "unchanged", "explained", "side_effect", "wrong_magnitude", "missed"]


# ---- 3. the three named scenarios --------------------------------------------

def test_demo_scenarios_verdicts():
    orders = synth_month(42, 800)
    results = [(name, run_gate(orders, BASE, after, spec), expect)
               for name, spec, after, expect in demo_scenarios(BASE)]
    for name, res, expect in results:
        assert res.passed is expect, name

    intended, side_effect, too_wide = (r for _, r, _ in results)
    assert intended.counts()["explained"] > 0
    # copy-paste side effect: every unexplained row is on the channel nobody meant to touch
    assert set(side_effect.unexplained()["channel"]) == {"channel_A"}
    # effective date ignored: every unexplained row is dated before the effective date
    eff = demo_scenarios(BASE)[0][1].overrides[0].from_date
    assert (too_wide.unexplained()["order_date"] < eff).all()


# ---- 4. coverage and tolerance -------------------------------------------------

def _spec():
    ov = Override("commission_rate", 0.15, channel="channel_B", from_date="2026-01-16")
    return ov, ChangeSpec("test", (ov,))


def test_probe_grid_catches_a_branch_the_month_never_reaches():
    ov, spec = _spec()
    orders = synth_month(3, 400)
    rare = Override("charge_platform_promo", 1, channel="channel_direct", scheme="platform_25",
                    min_gross=88_000)
    month_rows = rare.mask(orders).sum()
    after = BASE.with_overrides(ov, rare)
    if month_rows == 0:                      # the month never hits the branch ...
        assert run_gate(orders, BASE, after, spec, probe=False).passed
    assert not run_gate(orders, BASE, after, spec, probe=True).passed   # ... the grid does


def test_rounding_change_is_invisible_on_round_data_and_to_a_1krw_band():
    ov, spec = _spec()
    orders = synth_month(5, 400)
    after = BASE.with_overrides(ov).with_rounding("floor")
    assert run_gate(orders, BASE, after, spec, probe=False).passed          # month: all whole numbers
    assert not run_gate(orders, BASE, after, spec, eps=0).passed            # probe's off-grid value
    assert run_gate(orders, BASE, after, spec, eps=1).passed                # a 1 KRW band hides it


def test_probe_grid_includes_both_sides_of_the_effective_date():
    _ov, spec = _spec()
    grid = probe_grid(synth_month(1, 50), BASE, spec)
    assert {"2026-01-01", "2026-01-15", "2026-01-16", "2026-01-31"} <= set(grid["order_date"])
