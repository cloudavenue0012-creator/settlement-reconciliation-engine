"""Eval for three-way reconciliation: is the error put in the right place?

Each scenario is one synthetic month (three_way.generate) with errors planted in all
three sources — ~3% of ledger rows (typo / missing / double booking), ~3% of channel
rows (the five discrepancy types from synth_data.py) and one channel x scheme cell where
the rule engine runs a commission 3%p off. Rows hit by two at once are "multiple".

Scored row by row against the planted truth:

    location accuracy   share of erroneous rows whose located source is the right one
                        ("multiple" counts as right only when called "ambiguous")
    per-source recall   the same, split by where the error really was
    false alarm         clean rows that get any error location
    engine cell id      scenarios where suspect_cells() names exactly the planted cell

The two-way engine (recompute vs channel) is scored on the same rows for contrast: it
has one call for every disagreement — "the channel is off" — and no view of the ledger.

A second run plants the error in the shared contract master instead (ledger and engine
both read the stale rate, the channel is right). That run is reported, not gated: it is
the case three-way is *built* to get wrong at the row level (see three_way.py).

    python eval_three_way.py                       # 20 synthetic months, seeds 42..61
    python eval_three_way.py --min-location-acc 0.96 --max-false-alarm 0.01 --min-cell-id 1.0
"""
from __future__ import annotations

import argparse

import pandas as pd

from three_way import (AMBIGUOUS, CHANNEL, ENGINE, EXPECTED_CALL, LEDGER, OK, confusion,
                       generate, run_case, suspect_cells)

SOURCES = [LEDGER, CHANNEL, ENGINE, "multiple"]


def _cells(df: pd.DataFrame, loc: str) -> set:
    return {(r.channel, r.discount_scheme) for r in suspect_cells(df, loc).itertuples()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenarios", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42, help="first seed")
    ap.add_argument("-n", type=int, default=2000, help="orders per synthetic month")
    ap.add_argument("--min-location-acc", type=float, default=None)
    ap.add_argument("--max-false-alarm", type=float, default=None)
    ap.add_argument("--min-cell-id", type=float, default=None)
    a = ap.parse_args()
    seeds = range(a.seed, a.seed + a.scenarios)

    # ------------------------------------------------------------ main run: engine bug
    frames, cell_hits, spurious_channel_cells = [], 0, 0
    for s in seeds:
        case = generate(s, a.n, mode="engine_bug")
        df = run_case(case).assign(seed=s)
        frames.append(df)
        cell_hits += _cells(df, ENGINE) == {case.bad_cell}
        spurious_channel_cells += bool(_cells(df, CHANNEL))     # random channel errors only
    m = pd.concat(frames, ignore_index=True)
    m["right"] = m["location"] == m["truth"].map(EXPECTED_CALL)
    m["right_2way"] = (m["truth"] == CHANNEL) & (m["two_way"] == CHANNEL)
    err = m[m["truth"] != OK]
    clean = m[m["truth"] == OK]

    loc_acc = float(err["right"].mean())
    loc_acc_2 = float(err["right_2way"].mean())
    detect = float((err["location"] != OK).mean())
    detect_2 = float((err["two_way"] != OK).mean())
    false_alarm = float((clean["location"] != OK).mean())
    false_alarm_2 = float((clean["two_way"] != OK).mean())
    cell_id = cell_hits / a.scenarios

    print("=" * 74)
    print(f" Three-way reconciliation - eval ({a.scenarios} synthetic months x {a.n} orders, "
          f"seeds {seeds.start}..{seeds.stop - 1})")
    print("=" * 74)
    print(f" {'':26s}{'three-way':>14s}{'two-way':>14s}   rows")
    print("-" * 74)
    for src in SOURCES:
        sub = err[err["truth"] == src]
        print(f" right location: {src:9s}{sub['right'].mean():>14.1%}"
              f"{sub['right_2way'].mean():>14.1%}   {len(sub):>5d}")
    print("-" * 74)
    print(f" {'location accuracy':26s}{loc_acc:>14.1%}{loc_acc_2:>14.1%}   {len(err):>5d}")
    print(f" {'detected (any location)':26s}{detect:>14.1%}{detect_2:>14.1%}   {len(err):>5d}")
    print(f" {'false alarm (clean rows)':26s}{false_alarm:>14.2%}{false_alarm_2:>14.2%}   {len(clean):>5d}")
    print(f" {'engine cell identified':26s}{cell_id:>14.0%}{'n/a':>14s}   {a.scenarios:>5d} months")
    print("-" * 74)
    print(" confusion, all months (rows = planted truth, columns = located error):\n")
    print(confusion(m).to_string())

    # ------------------------------------------------- second run: shared stale master
    frames, flagged_cell = [], 0
    for s in seeds:
        case = generate(s, a.n, mode="shared_master")
        df = run_case(case)
        frames.append(df)
        flagged_cell += case.bad_cell in _cells(df, CHANNEL)
    sm = pd.concat(frames, ignore_index=True)
    master = sm[sm["truth"] == "master"]
    to_channel = float((master["location"] == CHANNEL).mean())
    print("-" * 74)
    print(" known limit - shared stale contract master (ledger + engine wrong, channel right):")
    print(f"   master-error rows blamed on the channel : {to_channel:6.1%}  ({len(master)} rows)")
    print(f"   stale cell raised as a systematic cluster: {flagged_cell}/{a.scenarios} months")
    print(f"   (same cluster rule on the main run, where channel errors are random: "
          f"{spurious_channel_cells}/{a.scenarios} months flagged)")
    print("=" * 74)

    failures = []
    if a.min_location_acc is not None and loc_acc < a.min_location_acc:
        failures.append(f"location accuracy {loc_acc:.3f} < {a.min_location_acc}")
    if a.max_false_alarm is not None and false_alarm > a.max_false_alarm:
        failures.append(f"false alarm {false_alarm:.4f} > {a.max_false_alarm}")
    if a.min_cell_id is not None and cell_id < a.min_cell_id:
        failures.append(f"engine cell id {cell_id:.2f} < {a.min_cell_id}")
    if failures:
        print("FAIL: " + "; ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
