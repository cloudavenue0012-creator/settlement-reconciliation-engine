"""Reproduce the rigged-benchmark trap this repo shipped with — and the fix.

WHAT WENT WRONG
---------------
The generator used to draw its heavy-tail noise as::

    payout = correct + randint(tol + 10, tol * 3)      # tol = the match tolerance

Read that again with the evaluation in mind. The evaluator flags a row when
``|actual - expected| > tol``. The noise was drawn to start at ``tol + 10``, so
*every* noise draw landed outside the tolerance — and, crucially, the whole noise
distribution **moved whenever the tolerance moved**.

That is a benchmark whose difficulty is a function of the parameter under
evaluation. Raise the tolerance and the noise obligingly gets bigger; the
separation between "noise" and "planted anomaly" stays exactly where it was. The
scoreboard looks good at every setting, so tuning appears to work and in fact
measures nothing.

Nobody wrote a bug. The line looks like a reasonable way to say "make the noise
big enough to matter". The coupling is invisible unless you sweep the parameter
and notice the score never degrades.

THE GENERAL SHAPE
-----------------
    If the data generator reads the parameter you are evaluating,
    your benchmark cannot rank settings of that parameter.

It shows up well beyond reconciliation: sampling hard negatives with the
retriever you are testing, choosing an outlier cutoff from the same statistic the
detector thresholds on, generating synthetic questions with the model under test.
The tell is always the same — **a sweep that is suspiciously flat**.

HOW TO CHECK YOUR OWN
---------------------
Sweep the parameter and look at the shape, not the level. A real benchmark has a
peak: too tight and recall collapses, too loose and precision does. A rigged one
is flat, because you moved the goalposts along with the ball.

Run::

    python rigged_benchmark_demo.py

No arguments, no network, ~10 seconds. Synthetic data only.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from faker import Faker

from settlement_rules import ChannelRule, expected_payout
from synth_data import (NOISE_ORDINARY, NOISE_TAIL_HI, NOISE_TAIL_LO, NOISE_TAIL_RATE,
                        gen_orders, load_rules)

TOLERANCES = [50, 100, 200, 400, 800]
N_ORDERS = 4000
SEED = 42
PLANTED_LO, PLANTED_HI = 300, 3000      # size range of a planted amount error


def _payouts(orders: pd.DataFrame, rules, tol: int, coupled: bool):
    """Build one settlement file. `coupled=True` reproduces the old, rigged noise."""
    rng = random.Random(SEED)
    rows = []
    for o in orders.to_dict("records"):
        rule: ChannelRule = rules[o["channel"]]
        eff_discount = 0 if o["platform_funded"] else o["discount_amount"]
        correct = expected_payout(o["gross_amount"], eff_discount, rule)

        if rng.random() < NOISE_TAIL_RATE:
            # The heavy tail is the only place the two generators differ — exactly as
            # in the original bug. Coupling it to `tol` is what breaks the benchmark.
            noise = (rng.choice([-1, 1]) * rng.randint(tol + 10, tol * 3) if coupled
                     else rng.choice([-1, 1]) * rng.randint(NOISE_TAIL_LO, NOISE_TAIL_HI))
        else:
            noise = rng.randint(-NOISE_ORDINARY, NOISE_ORDINARY)

        planted = rng.random() < 0.08
        err = rng.choice([-1, 1]) * rng.randint(PLANTED_LO, PLANTED_HI) if planted else 0
        rows.append({"diff": abs(noise + err), "planted": planted})
    return pd.DataFrame(rows)


def _score(df: pd.DataFrame, tol: int) -> tuple[float, float, float]:
    """precision, recall, and the diagnostic that actually matters: false-positive rate.

    FPR = share of *clean* rows the tolerance flags. In a sound benchmark this must
    fall as the tolerance rises — a wider band swallows more of the noise. If it
    does not, the noise moved with the band, and no sweep over that band is
    informative. F1 is the wrong tell here: both generators produce a similar F1
    spread, so a "flat F1" check looks decisive and is not (measured, not assumed).
    """
    flagged = df["diff"] > tol
    clean = ~df["planted"]
    tp = int((flagged & df["planted"]).sum())
    fp = int((flagged & clean).sum())
    fn = int((~flagged & df["planted"]).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, fp / int(clean.sum())


def _bar(v: float, w: int = 24) -> str:
    return "█" * round(v * w) + "·" * (w - round(v * w))


def main() -> None:
    rules, _ = load_rules(Path("rules.yaml"))
    random.seed(SEED)
    Faker.seed(SEED)
    orders = gen_orders(N_ORDERS, rules, Faker("ko_KR"))

    for coupled, title in ((True, "RIGGED — noise drawn as randint(tol+10, tol*3)"),
                           (False, "FIXED  — noise independent of tolerance")):
        print(f"\n{title}")
        print(f"  {'tol':>5}  {'precision':>9} {'recall':>7}   {'false-positive rate':>19}")
        fprs = []
        for tol in TOLERANCES:
            p, r, fpr = _score(_payouts(orders, rules, tol, coupled), tol)
            fprs.append(fpr)
            print(f"  {tol:>5}  {p:>9.3f} {r:>7.3f}   {fpr:>9.4f}  {_bar(fpr / max(fprs[0], 1e-9))}")
        shrink = max(0.0, 1 - (fprs[-1] / fprs[0])) if fprs[0] else 0.0
        verdict = (f"BROKEN — widening the band {TOLERANCES[0]}→{TOLERANCES[-1]} removed only "
                   f"{shrink:.0%} of the false positives. The noise widened with it, so no "
                   "sweep over this parameter can rank anything."
                   if shrink < 0.5 else
                   f"SOUND — widening the band removed {shrink:.0%} of the false positives, "
                   "which is what a tolerance is supposed to do. The sweep carries information.")
        print(f"  → {verdict}")

    print("\nThe fixed generator is the one this repo ships. The rigged one is kept")
    print("here only so the failure is reproducible rather than a story.\n")


if __name__ == "__main__":
    main()
