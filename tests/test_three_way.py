"""Tests for three-way reconciliation (ledger vs independent recompute vs channel).

1. The majority vote, pinned on hand-built values: every location it can emit.
2. Missing and duplicated rows fall out of the same comparison.
3. End-to-end on one synthetic month: location floor, engine cell found, and the
   two-way engine's blind spot (ledger errors) reproduced.
4. The documented limit — a shared stale master is blamed on the channel — stays true,
   and stays visible at the cell level.
"""
import numpy as np
import pandas as pd

from ruleset import base_ruleset
from three_way import (AMBIGUOUS, CHANNEL, ENGINE, EXPECTED_CALL, LEDGER, OK, generate, locate,
                       reconcile_three_way, run_case, suspect_cells)

TOL = 400


def test_locate_every_pattern():
    c = 10_000
    ledger = np.array([c, c + 5_000, c, c, c + 5_000, c - 350], float)
    engine = np.array([c, c, c, c + 5_000, c - 5_000, c], float)
    official = np.array([c + 15, c, c - 5_000, c, c, c + 350], float)
    assert list(locate(ledger, engine, official, TOL)) == [
        OK, LEDGER, CHANNEL, ENGINE, AMBIGUOUS,
        OK,            # chain within noise: the middle value agrees with both ends
    ]


def test_missing_and_duplicate_rows():
    orders = pd.DataFrame([
        dict(order_id=f"o{i}", channel="channel_A", discount_scheme="none", order_date="2026-01-05",
             gross_amount=20_000, discount_amount=0, platform_funded=False) for i in range(4)])
    pay = 17_600                                  # channel_A: 20,000 - 12% commission
    ledger = pd.DataFrame({"order_id": ["o0", "o1", "o2", "o3", "o3"],
                           "ledger_amount": [pay] * 5})          # o3 booked twice
    official = pd.DataFrame({"order_id": ["o0", "o1", "o3"], "channel": "channel_A",
                             "official_payout": [pay] * 3})     # o2 never settled
    df = reconcile_three_way(orders, ledger, official, base_ruleset(), TOL).set_index("order_id")
    assert df.loc["o0", "location"] == OK
    assert df.loc["o2", "location"] == CHANNEL       # dropped by the channel
    assert df.loc["o3", "location"] == LEDGER        # double booking in our own books


def test_generator_is_deterministic():
    a, b = generate(7, 300), generate(7, 300)
    for x, y in [(a.orders, b.orders), (a.ledger, b.ledger), (a.official, b.official),
                 (a.truth, b.truth)]:
        pd.testing.assert_frame_equal(x, y)
    assert a.bad_cell == b.bad_cell


def test_end_to_end_location_and_engine_cell():
    case = generate(42, 1500, mode="engine_bug")
    df = run_case(case)
    err = df[df["truth"] != OK]
    acc = (err["location"] == err["truth"].map(EXPECTED_CALL)).mean()
    assert acc >= 0.93
    cells = {(r.channel, r.discount_scheme) for r in suspect_cells(df, ENGINE).itertuples()}
    assert cells == {case.bad_cell}
    # the two-way engine cannot see a ledger error: recompute and channel agree on it
    ledger_only = df[df["truth"] == LEDGER]
    assert (ledger_only["two_way"] == OK).all()
    # and it calls an engine error a channel error
    engine_only = df[(df["truth"] == ENGINE) & (df["location"] == ENGINE)]
    assert (engine_only["two_way"] == CHANNEL).all()


def test_shared_master_limit_is_reproduced_and_flagged_as_a_cluster():
    case = generate(42, 1500, mode="shared_master")
    df = run_case(case)
    master = df[df["truth"] == "master"]
    assert (master["location"] == CHANNEL).mean() > 0.9     # the row vote gets this wrong ...
    cells = {(r.channel, r.discount_scheme) for r in suspect_cells(df, CHANNEL).itertuples()}
    assert case.bad_cell in cells                           # ... the cell view still raises it
