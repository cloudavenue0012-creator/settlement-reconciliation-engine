"""Versioned settlement rules + a vectorised payout engine.

reconcile.py answers "is the channel's number right?". Once the engine itself is in
production, two more questions show up, and both need to run the engine under more
than one set of rules:

  * rule_change_gate.py  "I changed the rules. Did *only* what I meant to change, change?"
  * three_way.py         "Ledger, engine and channel disagree. Which one is wrong?"

So the rules become a value (`RuleSet`) instead of a module-level config, and every
scoped exception to the base channel rules becomes an explicit `Override` (a channel,
a discount scheme, an effective date, a minimum order value). A rule *change* is then
just a different `RuleSet`, and two versions can be run side by side on the same data.

`base_ruleset()` reproduces `settlement_rules.expected_payout` exactly, row for row —
tests/test_rule_change_gate.py pins that against reconcile.py, so nothing here can
drift from the engine the headline numbers are measured on.

Also here: `synth_month`, a small order generator pinned to a fixed calendar month.
The existing generator dates orders relative to *today*, which is fine for the
2-way benchmark (dates are unused there) but not for effective-date rules, where
"before / after the 16th" has to mean the same rows on every run. It draws from its
own `random.Random(seed)`, so it never touches the global stream that synth_data.py
and the headline metrics depend on.

100% synthetic. Channel names and rates come from rules.yaml (illustrative values).
"""
from __future__ import annotations

import calendar
import random
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from settlement_rules import ChannelRule
from synth_data import SCHEMES, load_rules

RULES_PATH = Path(__file__).with_name("rules.yaml")

FIELDS = ("commission_rate", "merchant_burden", "charge_platform_promo")


@dataclass(frozen=True)
class Override:
    """One scoped exception to the base channel rules.

    `field` is set to `value` on every order matching *all* the given filters
    (None = no filter). Later overrides in a RuleSet win over earlier ones.

      commission_rate        platform commission on gross
      merchant_burden        share of the discount the merchant funds
      charge_platform_promo  1 = charge the merchant for a platform-funded promo
                             (normally a bug; used to model one)
    """
    field: str
    value: float
    channel: str | None = None
    scheme: str | None = None
    from_date: str | None = None      # ISO date, inclusive
    min_gross: int | None = None
    max_gross: int | None = None      # exclusive

    def __post_init__(self):
        if self.field not in FIELDS:
            raise ValueError(f"unknown override field {self.field!r}")

    def mask(self, orders: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=orders.index)
        if self.channel is not None:
            m &= orders["channel"] == self.channel
        if self.scheme is not None:
            m &= orders["discount_scheme"] == self.scheme
        if self.from_date is not None:
            m &= orders["order_date"].astype(str) >= self.from_date
        if self.min_gross is not None:
            m &= orders["gross_amount"] >= self.min_gross
        if self.max_gross is not None:
            m &= orders["gross_amount"] < self.max_gross
        return m

    def describe(self) -> str:
        scope = [f"channel={self.channel}" if self.channel else "all channels"]
        if self.scheme:
            scope.append(f"scheme={self.scheme}")
        if self.from_date:
            scope.append(f"from {self.from_date}")
        if self.min_gross is not None:
            scope.append(f"gross>={self.min_gross:,}")
        if self.max_gross is not None:
            scope.append(f"gross<{self.max_gross:,}")
        return f"{self.field}={self.value:g} ({', '.join(scope)})"


@dataclass(frozen=True)
class RuleSet:
    """A complete, runnable version of the settlement rules."""
    channels: dict = field(hash=False)            # channel -> ChannelRule
    overrides: tuple = ()
    rounding: str = "half_even"                   # "half_even" (Python round) | "floor"

    def with_overrides(self, *ovs: Override) -> "RuleSet":
        return replace(self, overrides=self.overrides + tuple(ovs))

    def with_rounding(self, mode: str) -> "RuleSet":
        return replace(self, rounding=mode)


def base_ruleset(path: Path = RULES_PATH) -> RuleSet:
    rules, _tol = load_rules(path)
    return RuleSet(channels=dict(rules))


def compute_payouts(orders: pd.DataFrame, rs: RuleSet) -> pd.Series:
    """Expected payout per order under rule version `rs` (int KRW, same index as orders).

    Same arithmetic, in the same order, as settlement_rules.expected_payout:
        gross - gross*commission_rate - effective_discount*merchant_burden
    so version A reproduces the engine bit for bit (half-even rounding = Python round).
    """
    ch = orders["channel"]
    comm = ch.map({k: r.commission_rate for k, r in rs.channels.items()}).to_numpy(float, copy=True)
    burden = ch.map({k: r.merchant_burden for k, r in rs.channels.items()}).to_numpy(float, copy=True)
    charge = np.zeros(len(orders), dtype=bool)
    for ov in rs.overrides:
        m = ov.mask(orders).to_numpy()
        if ov.field == "commission_rate":
            comm[m] = ov.value
        elif ov.field == "merchant_burden":
            burden[m] = ov.value
        else:
            charge[m] = bool(ov.value)

    gross = orders["gross_amount"].to_numpy(float)
    disc = orders["discount_amount"].to_numpy(float)
    platform = orders["platform_funded"].to_numpy(bool)
    eff_disc = np.where(platform & ~charge, 0.0, disc)
    raw = gross - gross * comm - eff_disc * burden
    out = np.rint(raw) if rs.rounding == "half_even" else np.floor(raw)
    return pd.Series(out.astype(np.int64), index=orders.index)


def synth_month(seed: int, n: int, year: int = 2026, month: int = 1,
                channels=None) -> pd.DataFrame:
    """Synthetic orders for one fixed calendar month (same schema as synth_data.gen_orders)."""
    rng = random.Random(seed)
    if channels is None:
        channels = list(base_ruleset().channels)
    days = calendar.monthrange(year, month)[1]
    stores = [f"BR-{i:03d}" for i in range(1, 41)]
    rows = []
    for i in range(n):
        ch = rng.choice(channels)
        scheme = rng.choice(list(SCHEMES))
        gross = rng.randrange(12_000, 90_000, 500)
        rate, platform_funded = SCHEMES[scheme]
        discount = 3_000 if scheme == "coupon_fixed" else round(gross * rate)
        rows.append(dict(
            order_id=f"M{seed:04d}-{i:05d}",
            store_id=rng.choice(stores),
            channel=ch,
            order_date=date(year, month, rng.randint(1, days)).isoformat(),
            gross_amount=gross,
            discount_scheme=scheme,
            discount_amount=discount,
            platform_funded=platform_funded,
            net_amount=gross - discount,
        ))
    return pd.DataFrame(rows)


__all__ = ["ChannelRule", "Override", "RuleSet", "base_ruleset", "compute_payouts", "synth_month"]
