"""Interactive Streamlit demo for the settlement reconciliation engine.

Regenerate synthetic settlements, then reconcile them live and watch precision /
recall move as you slide the match tolerance — the "verification gate" tuning knob.
All data is synthetic; no real company data.

    streamlit run app.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from explain import explain, faithfulness
from reconcile import reconcile
from tune import score

HERE = Path(__file__).parent
DATA = HERE / "data"

st.set_page_config(page_title="Settlement Reconciliation", page_icon="🧾", layout="wide")


@st.cache_data(show_spinner="Generating synthetic settlements...")
def generate(n: int, anomaly_rate: float, seed: int) -> str:
    subprocess.run(
        [sys.executable, "synth_data.py", "-n", str(n),
         "--anomaly-rate", str(anomaly_rate), "--seed", str(seed)],
        cwd=HERE, check=True, capture_output=True,
    )
    return f"{n}-{anomaly_rate}-{seed}"


st.title("🧾 Settlement Reconciliation & Verification Engine")
st.caption("Recompute-and-diff verification for multi-channel sales settlements — 100% synthetic data.")

with st.sidebar:
    st.header("Synthetic data")
    n = st.select_slider("Orders", [500, 1000, 2000, 5000], value=2000)
    rate = st.slider("Anomaly rate", 0.02, 0.20, 0.08, 0.01)
    seed = int(st.number_input("Seed", 0, 9999, 42))
    st.header("Verification gate")
    tol = st.slider("Match tolerance (KRW)", 10, 500, 50, 10,
                    help="Below this, |official - expected| is rounding noise, not an anomaly.")

generate(n, rate, seed)
df = reconcile(DATA / "orders.csv", DATA / "official_settlement.csv",
               HERE / "rules.yaml", tol_override=tol)
truth = pd.read_csv(DATA / "ground_truth.csv")
p, r, f = score(df, truth)

flagged = int(df.is_anomaly.sum())
gap_caught = int(df.loc[df.is_anomaly & df.present, "diff"].abs().sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Orders", f"{len(df):,}")
c2.metric("Flagged anomalies", f"{flagged:,}")
c3.metric("Payout gap caught", f"{gap_caught:,} KRW")
c4.metric("Precision / Recall", f"{p:.0%} / {r:.0%}")
c5.metric("F1", f"{f:.3f}")

left, right = st.columns([3, 2])
with left:
    st.subheader("Flagged anomalies (with grounded explanations)")
    anom = df[df.is_anomaly].copy()
    anom["why"] = anom.apply(explain, axis=1)
    anom["faithful"] = anom.apply(lambda x: faithfulness(x["why"], x)[0], axis=1)
    st.dataframe(
        anom[["order_id", "channel", "flag", "expected_payout", "official_payout", "diff", "why"]],
        use_container_width=True, height=430, hide_index=True,
    )
with right:
    st.subheader("Anomaly types")
    st.bar_chart(anom["flag"].value_counts())
    st.subheader("Tolerance tuning")
    img = HERE / "docs" / "tuning_curve.png"
    if img.exists():
        st.image(str(img), caption="Precision / recall / F1 vs tolerance (run tune.py to refresh)")

st.caption("Synthetic orders, channels, and fee rules only — no real company data. See README.")
