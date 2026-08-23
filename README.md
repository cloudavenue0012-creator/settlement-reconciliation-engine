# Settlement Reconciliation & Verification Engine

Recompute-and-diff verification for **multi-channel sales settlements**. The engine
never trusts the number a channel reports — it reconstructs the *expected* payout
for every order from first principles (order value + channel fee/discount rules),
diffs it against the channel's official settlement file, and flags the gaps:
wrong commission, mis-assigned discount burden, missing rows, double-counting,
and unexplained amount errors.

> **100% synthetic data.** Orders, stores, channel names, and fee rules are all
> randomly generated / illustrative (see `synth_data.py`, `rules.yaml`). No real
> company data, code, schemas, or figures are included. This repo is a portable
> re-implementation of a production pattern, built to be inspected.

## Why this exists

A merchant selling across delivery apps, partner platforms, and direct/POS
receives a settlement file from each channel and is expected to trust it. But
fees change, promo-funding splits are misapplied, and rows silently go missing or
get duplicated — each one a small, hard-to-notice cash leak. Manually spot-checking
thousands of orders doesn't scale. The fix is a **verification gate**: independently
recompute what each payout *should* be, and only trust matches within tolerance.

## Architecture

```
orders.csv ─┐
            ├─► reconstruct expected payout ─► match vs official ─► classify anomaly ─► verification gate ─► eval
rules.yaml ─┘        (settlement_rules.py)       (reconcile.py)        (fee/burden/          (flag)          (eval.py)
official_settlement.csv ─────────────────────────────────────────────  missing/dup/amount)
```

- **`settlement_rules.py`** — the canonical payout formula (`gross − commission − merchant_discount_burden`), shared by generator and reconciler so neither can drift.
- **`synth_data.py`** — generates orders + an official settlement file with *planted* discrepancies and a ground-truth label file. Noise is mostly sub-tolerance, with a heavy tail so detection faces a real precision/recall trade-off.
- **`reconcile.py`** — the engine: reconstructs expected payout, matches, classifies each anomaly type.
- **`eval.py`** — scores the engine against ground truth (the differentiator).

## Results (one synthetic run, 2,000 orders, ~8% planted anomalies)

| metric | value |
|---|---|
| reconciliation accuracy (clean within tolerance) | 97.7% |
| anomaly recall | 100.0% |
| anomaly precision | 78.5% |
| F1 | 0.879 |
| false-positive rate | 2.3% |
| payout gap recovered | 1.60M / 1.62M KRW (synthetic) |

Recall is perfect but precision is capped by rounding-noise false positives — which
is exactly the signal that the fixed tolerance should be *tuned*, not guessed
(see Roadmap). Numbers vary by `--seed`.

## Run

```bash
pip install -r requirements.txt
python synth_data.py -n 2000      # -> data/orders.csv, official_settlement.csv, ground_truth.csv
python reconcile.py               # -> data/reconciliation.csv
python eval.py                    # -> metrics report
python tune.py                    # -> docs/tuning_curve.png + recommended tolerance
python explain.py --demo-hallucination   # grounded explanations + faithfulness eval
streamlit run app.py              # interactive dashboard
```

## Interactive demo

`streamlit run app.py` — regenerate synthetic settlements and reconcile them live.
Slide the **match tolerance** and watch precision/recall move in real time; the
flagged-anomaly table carries a grounded explanation and a faithfulness score per row.

## Tolerance tuning

The fixed tolerance is a knob, not a guess. `tune.py` sweeps it and scores anomaly
detection at each level, so the operating point is *chosen* from the trade-off:

![tuning curve](docs/tuning_curve.png)

Low tolerance flags rounding noise (precision collapses); high tolerance lets small
real errors through (recall decays). Peak F1 here is **~0.97 at ~150 KRW** — and it
does *not* reach 1.0, because ~30% of amount errors are deliberately placed in the
gray zone near tolerance. No single threshold catches everything; the curve is how
you pick the least-bad one.

## Explanation layer + faithfulness eval

`explain.py` turns each flagged anomaly into a grounded, plain-language reason
("under-paid by 4,120 KRW: expected …, settled …; the merchant was charged for a
platform-funded promo"). The differentiator is the **faithfulness scorer**: every
figure an explanation cites must trace back to the reconciliation record, or it's
counted as ungrounded. `--demo-hallucination` injects a fake total to show the
scorer actually catches it (faithfulness drops below 1.0) rather than rubber-stamping
the output — the same guard keeps an LLM rephrasing layer honest.

## Roadmap

- **Streamlit dashboard** — interactive reconciliation view + drill-down, live demo.
- **LLM rephrase backend** — swap the template explainer for a model call, gated by the existing faithfulness eval.
- **CI** — run the pipeline + assert eval thresholds on every commit.

## License

MIT. Synthetic data only.
