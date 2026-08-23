"""Canonical settlement formula shared by the generator and the reconciler.

A merchant's expected payout from a sales channel, given the order and the
channel's fee/discount-burden rules:

    commission        = gross * commission_rate
    merchant_discount = discount * merchant_burden      # portion of the promo the merchant funds
    expected_payout   = gross - commission - merchant_discount

Everything here is generic revenue-settlement accounting — no company-specific
logic, channel names, or figures.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelRule:
    commission_rate: float      # platform commission on gross order value
    merchant_burden: float      # fraction (0..1) of the discount the merchant funds
    settlement_lag_days: int


def expected_payout(gross: float, discount: float, rule: ChannelRule) -> float:
    """Reconstruct what the channel *should* settle to the merchant for one order."""
    commission = gross * rule.commission_rate
    merchant_discount = discount * rule.merchant_burden
    return round(gross - commission - merchant_discount)
