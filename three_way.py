"""Three-way reconciliation — not just *that* the numbers disagree, but *where* the error is.

reconcile.py is two-way: independent recompute (R) vs the channel's official settlement
(O). When they disagree it can only say "the channel is off" — it cannot tell a channel
error from a bug in its own rules, and it cannot see an error in the merchant's own
books at all, because the books never enter the comparison.

Three-way adds the ledger (L): the receivable the books recorded for each order, booked
from the contract terms when the sale happened. Three values that are each produced by a
different party give a majority vote on every order:

    L ~ R ~ O            ok
    R ~ O,  L odd        ledger error         (booking typo, missed or double booking)
    L ~ R,  O odd        channel error        (wrong fee, promo charged, dropped/duplicated row)
    L ~ O,  R odd        rule-engine error    (the recompute itself is wrong)
    no two agree         ambiguous            (two sources wrong at once -> manual review)

"~" means within the match tolerance from rules.yaml (400 KRW), the same band the
two-way engine uses. A missing row counts as 0 and a duplicated row sums, so dropped and
doubled entries fall out of the same comparison.

Engine errors are systematic — a wrong rule hits every order in its cell — so
`suspect_cells` aggregates the row votes by channel x discount scheme. A cell where most
rows vote "engine" is a rule to fix, not a list of orders to chase.

**What it cannot do.** The vote assumes the three sources fail independently. If the
ledger and the engine read the same stale contract master, they agree with each other
and outvote the one correct source: every row in that cell is blamed on the channel.
`generate(mode="shared_master")` plants exactly that, and eval_three_way.py measures it.
The row vote is wrong there by construction; what still works is the *cell* view — a
whole cell of "channel" votes is systematic, which is not what random channel errors look
like, and is the cue to check the contract itself (a fourth source).

    python three_way.py               # one synthetic month, confusion matrix + suspect cells
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ruleset import RULES_PATH, Override, RuleSet, base_ruleset, compute_payouts, synth_month
from synth_data import (NOISE_ORDINARY, NOISE_TAIL_HI, NOISE_TAIL_LO, NOISE_TAIL_RATE, SCHEMES,
                        load_rules)

OK, LEDGER, CHANNEL, ENGINE, AMBIGUOUS = "ok", "ledger", "channel", "engine", "ambiguous"
LOCATIONS = [OK, LEDGER, CHANNEL, ENGINE, AMBIGUOUS]
# ground-truth labels: one wrong source, "multiple", or "master" (shared upstream error)
TRUTHS = [OK, LEDGER, CHANNEL, ENGINE, "multiple", "master"]
EXPECTED_CALL = {OK: OK, LEDGER: LEDGER, CHANNEL: CHANNEL, ENGINE: ENGINE,
                 "multiple": AMBIGUOUS, "master": None}   # "master" has no correct row-level call


def default_tolerance() -> int:
    return load_rules(RULES_PATH)[1]


# --------------------------------------------------------------------- classifier

def locate(ledger: np.ndarray, engine: np.ndarray, official: np.ndarray, tol: int) -> np.ndarray:
    """Majority vote over three values per order -> where the error is."""
    lr = np.abs(ledger - engine) <= tol
    lo = np.abs(ledger - official) <= tol
    ro = np.abs(engine - official) <= tol
    agree = lr.astype(int) + lo + ro
    return np.select(
        [agree >= 2, ro & ~lr & ~lo, lr & ~lo & ~ro, lo & ~lr & ~ro],
        [OK, LEDGER, CHANNEL, ENGINE],
        default=AMBIGUOUS,
    )


def reconcile_three_way(orders: pd.DataFrame, ledger: pd.DataFrame, official: pd.DataFrame,
                        engine_rules: RuleSet, tol: int) -> pd.DataFrame:
    """One row per order: ledger / engine / official values and the located error."""
    df = orders.copy()
    df["engine"] = compute_payouts(orders, engine_rules).to_numpy()
    led = ledger.groupby("order_id")["ledger_amount"].agg(["sum", "size"])
    off = official.groupby("order_id")["official_payout"].agg(["sum", "size"])
    df["ledger"] = df["order_id"].map(led["sum"]).fillna(0).to_numpy()
    df["ledger_rows"] = df["order_id"].map(led["size"]).fillna(0).astype(int).to_numpy()
    df["official"] = df["order_id"].map(off["sum"]).fillna(0).to_numpy()
    df["official_rows"] = df["order_id"].map(off["size"]).fillna(0).astype(int).to_numpy()
    df["location"] = locate(df["ledger"].to_numpy(float), df["engine"].to_numpy(float),
                            df["official"].to_numpy(float), tol)
    df["two_way"] = np.where((df["engine"] - df["official"]).abs() > tol, CHANNEL, OK)
    return df


def suspect_cells(df: pd.DataFrame, location: str = ENGINE, min_rows: int = 3,
                  min_share: float = 0.5) -> pd.DataFrame:
    """Channel x scheme cells where most orders vote the same location — a systematic error."""
    g = (df.assign(hit=df["location"] == location)
           .groupby(["channel", "discount_scheme"], as_index=False)
           .agg(orders=("hit", "size"), hits=("hit", "sum")))
    g["share"] = g["hits"] / g["orders"]
    return g[(g["hits"] >= min_rows) & (g["share"] >= min_share)].sort_values("hits", ascending=False)


# --------------------------------------------------------------------- generator

@dataclass
class Case:
    orders: pd.DataFrame
    ledger: pd.DataFrame          # order_id, ledger_amount  (missing / duplicated rows planted)
    official: pd.DataFrame        # order_id, channel, official_payout
    engine_rules: RuleSet         # the rules the recompute actually ran with
    truth: pd.DataFrame           # order_id, truth (TRUTHS), ledger_kind, channel_kind
    bad_cell: tuple | None        # (channel, scheme) of the planted rule / master error


def _cell_override(rng: random.Random, base: RuleSet):
    ch = rng.choice(list(base.channels))
    scheme = rng.choice(list(SCHEMES))
    old = base.channels[ch].commission_rate
    delta = rng.choice([-0.03, 0.03])
    if old + delta < 0:
        delta = -delta
    return (ch, scheme), Override("commission_rate", round(old + delta, 4), channel=ch, scheme=scheme)


def generate(seed: int = 42, n: int = 2000, ledger_rate: float = 0.03, channel_rate: float = 0.03,
             mode: str = "engine_bug") -> Case:
    """Synthetic month with errors planted in all three sources.

    mode = "engine_bug"     one channel x scheme cell runs a wrong commission in the engine
           "shared_master"  the same wrong commission sits in the contract master that BOTH
                            the ledger and the engine read; the channel is right
           "none"           engine and ledger read correct rules
    """
    if mode not in ("engine_bug", "shared_master", "none"):
        raise ValueError(mode)
    rng = random.Random(seed * 104_729 + 1)
    base = base_ruleset()
    orders = synth_month(seed, n)

    bad_cell, engine_rules, master_rules = None, base, base
    if mode != "none":
        bad_cell, ov = _cell_override(rng, base)
        engine_rules = base.with_overrides(ov)
        if mode == "shared_master":
            master_rules = engine_rules

    correct = compute_payouts(orders, base).to_numpy()
    engine = compute_payouts(orders, engine_rules).to_numpy()
    booked = compute_payouts(orders, master_rules).to_numpy()

    ledger_rows, official_rows, truth = [], [], []
    for i, o in enumerate(orders.to_dict("records")):
        c = int(correct[i])
        # ledger: booked to the nearest 10 KRW from the contract master
        lv, l_kind = int(round(booked[i], -1)), "ok"
        l_present, l_dup = True, False
        if rng.random() < ledger_rate:
            l_kind = rng.choice(["typo", "missing", "duplicate"])
            if l_kind == "typo":
                lv += rng.choice([-1, 1]) * rng.randint(30, 15_000)
            elif l_kind == "missing":
                l_present = False
            else:
                l_dup = True
        if l_present:
            ledger_rows.append(dict(order_id=o["order_id"], ledger_amount=lv))
            if l_dup:
                ledger_rows.append(dict(order_id=o["order_id"], ledger_amount=lv))

        # official: correct payout + the same noise profile as synth_data.py
        if rng.random() < NOISE_TAIL_RATE:
            ov_ = c + rng.choice([-1, 1]) * rng.randint(NOISE_TAIL_LO, NOISE_TAIL_HI)
        else:
            ov_ = c + rng.randint(-NOISE_ORDINARY, NOISE_ORDINARY)
        c_kind, c_present, c_dup = "ok", True, False
        if rng.random() < channel_rate:
            c_kind = rng.choice(["fee_error", "burden_error", "amount_mismatch", "missing", "duplicate"])
            rule = base.channels[o["channel"]]
            eff_disc = 0 if o["platform_funded"] else o["discount_amount"]
            gross = o["gross_amount"]
            if c_kind == "fee_error":
                ov_ = round(gross - gross * (rule.commission_rate + 0.03) - eff_disc * rule.merchant_burden)
            elif c_kind == "burden_error":
                ov_ = (round(gross - gross * rule.commission_rate - o["discount_amount"] * rule.merchant_burden)
                       if o["platform_funded"] else c - rng.randint(1_500, 6_000))
            elif c_kind == "amount_mismatch":
                ov_ = c + rng.choice([-1, 1]) * rng.randint(30, 15_000)
            elif c_kind == "missing":
                c_present = False
            else:
                c_dup = True
        if c_present:
            official_rows.append(dict(order_id=o["order_id"], channel=o["channel"], official_payout=ov_))
            if c_dup:
                official_rows.append(dict(order_id=o["order_id"], channel=o["channel"], official_payout=ov_))

        wrong = []
        if l_kind != "ok":
            wrong.append(LEDGER)
        if c_kind != "ok":
            wrong.append(CHANNEL)
        if mode == "engine_bug" and engine[i] != c:
            wrong.append(ENGINE)
        if mode == "shared_master" and booked[i] != c:
            wrong.append("master")
        label = OK if not wrong else wrong[0] if len(wrong) == 1 else "multiple"
        truth.append(dict(order_id=o["order_id"], truth=label, ledger_kind=l_kind, channel_kind=c_kind))

    return Case(orders, pd.DataFrame(ledger_rows), pd.DataFrame(official_rows),
                engine_rules, pd.DataFrame(truth), bad_cell)


def run_case(case: Case, tol: int | None = None) -> pd.DataFrame:
    tol = default_tolerance() if tol is None else tol
    df = reconcile_three_way(case.orders, case.ledger, case.official, case.engine_rules, tol)
    return df.merge(case.truth, on="order_id")


def confusion(df: pd.DataFrame, col: str = "location") -> pd.DataFrame:
    t = pd.crosstab(df["truth"], df[col])
    return t.reindex(index=[x for x in TRUTHS if x in t.index],
                     columns=[x for x in LOCATIONS if x in t.columns], fill_value=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-n", type=int, default=2000)
    ap.add_argument("--mode", default="engine_bug", choices=["engine_bug", "shared_master", "none"])
    a = ap.parse_args()

    case = generate(a.seed, a.n, mode=a.mode)
    df = run_case(case)
    print("=" * 72)
    print(f" Three-way reconciliation - {a.n} synthetic orders, seed {a.seed}, mode={a.mode}")
    print("=" * 72)
    print(" rows = planted truth, columns = located error\n")
    print(confusion(df).to_string())
    if case.bad_cell:
        print(f"\n planted cell: {case.bad_cell[0]} x {case.bad_cell[1]}")
    for loc in (ENGINE, CHANNEL):
        cells = suspect_cells(df, loc)
        for _, c in cells.iterrows():
            print(f" suspect cell ({loc:7s}): {c['channel']} x {c['discount_scheme']}  "
                  f"{int(c['hits'])}/{int(c['orders'])} orders")
    print("=" * 72)


if __name__ == "__main__":
    main()
