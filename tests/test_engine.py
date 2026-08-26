"""Unit tests for the reconciliation engine.

Two kinds of test:
  1. Deterministic, hand-built frames that pin every anomaly class the
     classifier can emit (no randomness — these must never flake).
  2. Property/regression tests over the synthetic generator (determinism,
     and an end-to-end recall floor that guards the headline number).
"""
from pathlib import Path

import pandas as pd
import pytest

import synth_data
from eval import prf
from reconcile import reconcile_frames
from settlement_rules import ChannelRule, expected_payout
from synth_data import generate, load_rules

RULE = ChannelRule(commission_rate=0.10, merchant_burden=0.5, settlement_lag_days=3)
RULES = {"A": RULE}
TOL = 100


def _order(oid, gross=10_000, discount=1_000, platform_funded=False):
    return dict(order_id=oid, channel="A", gross_amount=gross,
                discount_amount=discount, platform_funded=platform_funded)


# ---- 1. the settlement formula ------------------------------------------------

def test_expected_payout_formula():
    # commission = 10000*0.10 = 1000 ; merchant_discount = 1000*0.5 = 500
    assert expected_payout(10_000, 1_000, RULE) == 8_500
    # a platform-funded promo settles with zero merchant discount
    assert expected_payout(10_000, 0, RULE) == 9_000


# ---- 2. every classifier branch, deterministically ---------------------------

def test_reconcile_flags_cover_every_class():
    orders = pd.DataFrame([
        _order("ORD1"),                                        # ok
        _order("ORD2"),                                        # missing (no official row)
        _order("ORD3"),                                        # duplicate (two official rows)
        _order("ORD4"),                                        # fee_error
        _order("ORD5"),                                        # amount_mismatch
        _order("ORD6", discount=2_000, platform_funded=True),  # burden_error
    ])
    ok = expected_payout(10_000, 1_000, RULE)                       # 8500
    fee_bad = expected_payout(10_000, 1_000, ChannelRule(0.13, 0.5, 3))  # +3%p commission
    burden_bad = expected_payout(10_000, 2_000, RULE)              # merchant wrongly charged the promo

    official = pd.DataFrame([
        dict(order_id="ORD1", channel="A", official_payout=ok),
        dict(order_id="ORD3", channel="A", official_payout=ok),
        dict(order_id="ORD3", channel="A", official_payout=ok),   # duplicated row
        dict(order_id="ORD4", channel="A", official_payout=fee_bad),
        dict(order_id="ORD5", channel="A", official_payout=ok - 4_000),  # unexplained gap
        dict(order_id="ORD6", channel="A", official_payout=burden_bad),
    ])

    df = reconcile_frames(orders, official, RULES, TOL).set_index("order_id")

    assert df.loc["ORD1", "flag"] == "ok"
    assert df.loc["ORD2", "flag"] == "missing"
    assert df.loc["ORD3", "flag"] == "duplicate"
    assert df.loc["ORD4", "flag"] == "fee_error"
    assert df.loc["ORD5", "flag"] == "amount_mismatch"
    assert df.loc["ORD6", "flag"] == "burden_error"

    assert bool(df.loc["ORD1", "is_anomaly"]) is False
    assert df.loc[["ORD2", "ORD3", "ORD4", "ORD5", "ORD6"], "is_anomaly"].all()


def test_within_tolerance_is_not_flagged():
    orders = pd.DataFrame([_order("ORD1")])
    exp = expected_payout(10_000, 1_000, RULE)
    official = pd.DataFrame([dict(order_id="ORD1", channel="A", official_payout=exp + TOL)])
    df = reconcile_frames(orders, official, RULES, TOL).set_index("order_id")
    assert df.loc["ORD1", "flag"] == "ok"          # drift within tolerance is accepted


# ---- 3. the eval helper -------------------------------------------------------

def test_prf_helper():
    p, r, f = prf(tp=8, fp=2, fn=0)
    assert p == pytest.approx(0.8)
    assert r == pytest.approx(1.0)
    assert f == pytest.approx(2 * 0.8 * 1.0 / (0.8 + 1.0))
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)          # no divide-by-zero


# ---- 4. generator properties + end-to-end regression -------------------------

def test_generator_is_deterministic():
    a = generate(n=200, anomaly_rate=0.08, seed=7)
    b = generate(n=200, anomaly_rate=0.08, seed=7)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)
    orders, _official, truth = a
    assert len(orders) == 200
    assert (truth["anomaly"] != "ok").sum() > 0     # anomalies actually planted


def test_end_to_end_recall_floor():
    """Guard the headline: recall must not silently regress on a fresh dataset."""
    orders, official, truth = generate(n=800, anomaly_rate=0.08, seed=42)
    rules, tol = load_rules(Path(synth_data.__file__).with_name("rules.yaml"))
    df = reconcile_frames(orders, official, rules, tol)
    m = df.merge(truth, on="order_id")
    m["true_anomaly"] = m["anomaly"] != "ok"
    tp = int((m.is_anomaly & m.true_anomaly).sum())
    fn = int((~m.is_anomaly & m.true_anomaly).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    assert recall >= 0.90
