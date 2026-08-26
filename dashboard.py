"""Render a one-glance analytics dashboard (docs/dashboard.png) from a fresh
synthetic run: flag mix, detection scoreboard, tolerance sweep, and the KRW gap
recovered. A static PNG so it renders on GitHub and in the README without a
running server. (The interactive version is app.py / Streamlit.)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import synth_data  # noqa: E402
from reconcile import reconcile_frames  # noqa: E402
from synth_data import generate, load_rules  # noqa: E402

INK, MUTED, ACCENT, GOOD, WARN, GRID = (
    "#1f2933", "#7b8794", "#2b6cb0", "#2f855a", "#c05621", "#e5e9ee",
)


def _score(orders, official, truth, rules, tol):
    df = reconcile_frames(orders, official, rules, tol)
    m = df.merge(truth, on="order_id")
    m["true_anomaly"] = m["anomaly"] != "ok"
    tp = int((m.is_anomaly & m.true_anomaly).sum())
    fp = int((m.is_anomaly & ~m.true_anomaly).sum())
    fn = int((~m.is_anomaly & m.true_anomaly).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return df, m, {"p": p, "r": r, "f": f}


def main():
    orders, official, truth = generate(n=2000, anomaly_rate=0.08, seed=42)
    rules, tol = load_rules(Path(synth_data.__file__).with_name("rules.yaml"))
    df, m, met = _score(orders, official, truth, rules, tol)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Settlement Reconciliation — run overview (100% synthetic data)",
                 fontsize=14, fontweight="bold", color=INK)

    # A — orders by flag
    order = ["ok", "fee_error", "burden_error", "amount_mismatch", "missing", "duplicate"]
    counts = df["flag"].value_counts().reindex(order).fillna(0).astype(int)
    ax[0, 0].bar(range(len(order)), counts.values, color=[MUTED] + [ACCENT] * 5)
    ax[0, 0].set_xticks(range(len(order)))
    ax[0, 0].set_xticklabels(order, rotation=30, ha="right", fontsize=8, color=INK)
    ax[0, 0].set_title("orders by flag", fontsize=10, color=INK, loc="left")
    for i, v in enumerate(counts.values):
        ax[0, 0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7, color=MUTED)

    # B — detection scoreboard
    names, vals = ["precision", "recall", "F1"], [met["p"], met["r"], met["f"]]
    ax[0, 1].barh(range(3), vals, color=[ACCENT, GOOD, INK])
    ax[0, 1].set_yticks(range(3))
    ax[0, 1].set_yticklabels(names, fontsize=9, color=INK)
    ax[0, 1].set_xlim(0, 1.0)
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_title("anomaly detection", fontsize=10, color=INK, loc="left")
    for i, v in enumerate(vals):
        ax[0, 1].text(v, i, f"  {v:.3f}", va="center", fontsize=9, color=INK)

    # C — tolerance sweep (operating-point choice)
    tols = list(range(50, 801, 50))
    f1s, recs = [], []
    for t in tols:
        _, _, s = _score(orders, official, truth, rules, t)
        f1s.append(s["f"])
        recs.append(s["r"])
    ax[1, 0].plot(tols, f1s, marker="o", ms=4, color=ACCENT, label="F1")
    ax[1, 0].plot(tols, recs, marker="s", ms=4, color=GOOD, label="recall")
    ax[1, 0].axvline(tol, color=WARN, ls="--", lw=1.2, label=f"shipped tol = {tol}")
    ax[1, 0].set_ylim(0.5, 1.02)
    ax[1, 0].set_xlabel("match tolerance (KRW)", fontsize=8, color=MUTED)
    ax[1, 0].set_title("precision/recall trade-off vs tolerance", fontsize=10, color=INK, loc="left")
    ax[1, 0].legend(fontsize=8, frameon=False)

    # D — KRW payout gap recovered
    gap_total = int(m.loc[m.present, "diff"].abs().sum())
    gap_caught = int(m.loc[m.is_anomaly & m.present, "diff"].abs().sum())
    share = gap_caught / gap_total if gap_total else 0
    ax[1, 1].bar(["caught", "total"], [gap_caught, gap_total], color=[GOOD, GRID])
    ax[1, 1].set_title(f"payout gap recovered — {share:.1%} flagged (KRW)",
                       fontsize=10, color=INK, loc="left")
    ax[1, 1].margins(y=0.15)
    for i, v in enumerate([gap_caught, gap_total]):
        ax[1, 1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8, color=INK)

    for a in ax.flat:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)

    docs = Path(__file__).with_name("docs")
    docs.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = docs / "dashboard.png"
    fig.savefig(out, dpi=130, facecolor="white")
    print(f"wrote {out}  (F1={met['f']:.3f}, recall={met['r']:.1%})")


if __name__ == "__main__":
    main()
