"""
04_business_value.py — Business Value Translation
Bank Marketing Campaign: Predicting Term Deposit Subscriptions

Translates model predictions into financial decision metrics:
  - Subscriptions won at a fixed call budget: model vs random targeting
  - Net profit curve by how far down the scored list we call
  - Precision@K curves for campaign planning
  - Lift chart

Assumptions (illustrative, easily updated):
  - Average term deposit value:  £10,000
  - Bank margin on deposit:       1.5% per annum
  - Revenue per conversion:      £150
  - Cost per outbound call:      £5
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data"
FIGS    = ROOT / "figures"
OUTPUTS = ROOT / "outputs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import INK, GOLD, GREY, LIGHT, DARK_GOLD, SUBTLE, apply_style, frame, save

apply_style()

# ── Business parameters ────────────────────────────────────────────────────
REVENUE_PER_CONVERSION = 150   # £ margin on average term deposit
COST_PER_CALL          = 5     # £ outbound call cost
BUDGET_SHARE           = 0.30  # share of the list the call centre can reach


def load_data():
    X_test  = pd.read_csv(DATA / "X_test.csv")
    y_test  = pd.read_csv(DATA / "y_test.csv").squeeze()
    with open(OUTPUTS / "best_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    model_name = bundle["name"]
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"Model: {model_name} | Test rows: {len(y_test):,}")
    return y_test.values, y_prob


def _row(label, contacted, conversions, total_conv):
    revenue, cost = conversions * REVENUE_PER_CONVERSION, contacted * COST_PER_CALL
    return {
        "Strategy":            label,
        "Clients Contacted":   int(round(contacted)),
        "Conversions":         int(round(conversions)),
        "Conversion Rate":     conversions / contacted,
        "Converters Captured": conversions / total_conv,
        "Net Profit (£)":      revenue - cost,
        "Cost / Conversion":   cost / conversions,
    }


def compute_strategy_comparison(y_true, y_prob) -> pd.DataFrame:
    """
    Compare targeting strategies on the held-out test set.

    The key comparison is at a FIXED BUDGET: if the call centre can only reach
    30% of the list, who should it call? Random calling is shown at its expected
    value (30% of clients at the overall conversion rate).
    """
    n, total = len(y_true), y_true.sum()
    order = np.argsort(y_prob)[::-1]
    k = int(BUDGET_SHARE * n)
    k15 = int(0.15 * n)
    rows = [
        _row("Call everyone", n, total, total),
        _row("Random 30%", k, k * y_true.mean(), total),
        _row("Model top 30%", k, y_true[order[:k]].sum(), total),
        _row("Model top 15%", k15, y_true[order[:k15]].sum(), total),
    ]
    return pd.DataFrame(rows)


def profit_curve(y_true, y_prob):
    """Net profit if we call the top-x% of the scored list, for every x."""
    y_sorted = y_true[np.argsort(y_prob)[::-1]]
    calls = np.arange(1, len(y_true) + 1)
    return calls / len(y_true), np.cumsum(y_sorted) * REVENUE_PER_CONVERSION - calls * COST_PER_CALL


def plot_strategy_comparison(df_strat: pd.DataFrame, y_true, y_prob) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    fig.subplots_adjust(top=0.78, wspace=0.25)
    d = df_strat.set_index("Strategy")

    # Left: same budget, different targeting
    ax = axes[0]
    sub = d.loc[["Random 30%", "Model top 30%"]]
    ax.bar(["Random\n30%", "Model's\ntop 30%"], sub.Conversions, color=[GREY, GOLD], width=0.55)
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(i, r.Conversions + 12, f"{int(r.Conversions):,}", ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=DARK_GOLD if i else INK)
        ax.text(i, r.Conversions / 2, f"{r['Conversion Rate']:.0%}\nconvert", ha="center", va="center",
                fontsize=9, color="white" if i == 0 else INK)
    ax.set_ylim(0, sub.Conversions.max() * 1.18)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"Subscriptions from {int(sub['Clients Contacted'].iloc[0]):,} calls")

    # Right: profit by depth of calling
    ax = axes[1]
    ax.grid(axis="y", color=LIGHT, lw=1)
    share, profit = profit_curve(y_true, y_prob)
    ax.plot(share * 100, profit, color=INK, lw=2.2)
    rand_profit = share * len(y_true) * (y_true.mean() * REVENUE_PER_CONVERSION - COST_PER_CALL)
    ax.plot(share * 100, rand_profit, color=GREY, lw=1.3, ls="--")
    ax.text(62, rand_profit[int(0.6 * len(share))] - 4000, "calling a random x%", color=GREY, fontsize=9,
            va="top", rotation=12)
    ax.text(16, 60000, "calling the top-scored x%", color=INK, fontsize=9, fontweight="bold",
            ha="left", va="center")
    best = profit.argmax()
    marks = [(int(BUDGET_SHARE * len(share)) - 1, "Top 30%", (0, -34), "center"),
             (best, "Maximum", (-6, -34), "right"),
             (len(share) - 1, "Everyone", (0, 12), "right")]
    for i, lab, offset, ha in marks:
        ax.scatter(share[i] * 100, profit[i], color=GOLD, edgecolor=INK, lw=0.6, s=55, zorder=3)
        ax.annotate(f"{lab}\n£{profit[i]:,.0f}", (share[i] * 100, profit[i]), textcoords="offset points",
                    xytext=offset, ha=ha, fontsize=9, color=DARK_GOLD, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"£{int(x / 1000)}k"))
    ax.set_xlabel("% of clients called, highest scores first")
    ax.set_xlim(0, 101)
    ax.set_title("Net profit by how far down the list we call")

    m, r = sub.Conversions.iloc[1], sub.Conversions.iloc[0]
    frame(fig, f"At the same budget, the model wins {m / r:.1f} times as many subscriptions",
          f"Held-out test set of {len(y_true):,} clients. Assumes £{REVENUE_PER_CONVERSION} revenue per "
          f"subscription and £{COST_PER_CALL} per call.")
    save(fig, FIGS / "09_strategy_comparison.png")


def plot_cumulative_lift(y_true, y_prob) -> None:
    """
    Cumulative gains (lift) curve: what fraction of all converters
    do we capture if we contact the top-X% of the scored list?
    """
    y_sorted = y_true[np.argsort(y_prob)[::-1]]
    n = len(y_true)
    pct_contacted = np.arange(1, n + 1) / n
    pct_captured = np.cumsum(y_sorted) / y_true.sum()
    lift = pct_captured / pct_contacted
    i30 = int(0.30 * n) - 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.subplots_adjust(top=0.78, wspace=0.28)
    for ax in axes:
        ax.grid(axis="y", color=LIGHT, lw=1)

    ax = axes[0]
    ax.fill_between(pct_contacted * 100, pct_captured * 100, pct_contacted * 100, color=GOLD, alpha=0.15, lw=0)
    ax.plot(pct_contacted * 100, pct_captured * 100, color=INK, lw=2.2)
    ax.plot([0, 100], [0, 100], color=GREY, lw=1.2, ls="--")
    ax.text(74, 64, "random calling", color=GREY, fontsize=9, rotation=36, ha="center", va="top")
    ax.vlines(30, 0, pct_captured[i30] * 100, color=GOLD, lw=1.2)
    ax.scatter(30, pct_captured[i30] * 100, color=GOLD, edgecolor=INK, lw=0.6, s=55, zorder=3)
    ax.annotate(f"top 30% of calls reach\n{pct_captured[i30]:.0%} of all subscribers", (30, pct_captured[i30] * 100),
                xytext=(10, -40), textcoords="offset points", fontsize=9, color=DARK_GOLD, fontweight="bold")
    ax.set(xlabel="% of clients called, highest scores first", ylabel="% of all subscribers reached",
           xlim=(0, 100), ylim=(0, 101))
    ax.set_title("Cumulative gains")

    ax = axes[1]
    ax.plot(pct_contacted * 100, lift, color=INK, lw=2.2)
    ax.axhline(1, color=GREY, lw=1.2, ls="--")
    ax.text(60, 0.9, "random calling", color=GREY, fontsize=9, ha="center", va="top")
    ax.scatter(30, lift[i30], color=GOLD, edgecolor=INK, lw=0.6, s=55, zorder=3)
    ax.annotate(f"{lift[i30]:.1f}x at 30%", (30, lift[i30]), xytext=(8, 8), textcoords="offset points",
                fontsize=9, color=DARK_GOLD, fontweight="bold")
    ax.set(xlabel="% of clients called, highest scores first", ylabel="Lift over random calling",
           xlim=(0, 100), ylim=(0, None))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}x"))
    ax.set_title("Lift")

    frame(fig, f"Calling the top 30% reaches {pct_captured[i30]:.0%} of all subscribers",
          "How much of the total a scored call list captures, compared with calling clients at random.")
    save(fig, FIGS / "10_lift_curve.png")


def plot_precision_at_k(y_true, y_prob) -> None:
    """Precision@K: conversion rate among the clients called, at each depth of the list."""
    y_sorted = y_true[np.argsort(y_prob)[::-1]]
    n = len(y_true)
    k_range = np.arange(1, n + 1)
    prec_at_k = np.cumsum(y_sorted) / k_range
    pct_range = k_range / n * 100
    baseline = y_true.mean()
    i15 = int(0.15 * n) - 1

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(top=0.8)
    ax.grid(axis="y", color=LIGHT, lw=1)
    ax.axvspan(0, 15, color=GOLD, alpha=0.12, lw=0)
    ax.plot(pct_range, prec_at_k, color=INK, lw=2.2)
    ax.axhline(baseline, color=GREY, lw=1.2, ls="--")
    ax.text(50, baseline - 0.015, f"average conversion ({baseline:.0%})", color=GREY, fontsize=9,
            ha="center", va="top")
    ax.scatter(15, prec_at_k[i15], color=GOLD, edgecolor=INK, lw=0.6, s=55, zorder=3)
    ax.annotate(f"top 15%: {prec_at_k[i15]:.0%} convert", (15, prec_at_k[i15]), xytext=(10, 10),
                textcoords="offset points", fontsize=9, color=DARK_GOLD, fontweight="bold")
    ax.set_xlabel("% of clients called, highest scores first")
    ax.set_ylabel("Conversion rate among clients called")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, 80)
    ax.set_ylim(0, None)
    frame(fig, f"The top 15% of scores convert at {prec_at_k[i15]:.0%}, "
               f"{prec_at_k[i15] / baseline:.0f} times the average",
          "Conversion rate among the clients called, as the call list goes deeper. The first few hundred "
          "calls are noisy.")
    save(fig, FIGS / "11_precision_at_k.png")


def print_summary(df_strat: pd.DataFrame, y_true, y_prob) -> None:
    show = df_strat.copy()
    for c in ["Conversion Rate", "Converters Captured"]:
        show[c] = show[c].map("{:.1%}".format)
    show["Net Profit (£)"] = show["Net Profit (£)"].map("£{:,.0f}".format)
    show["Cost / Conversion"] = show["Cost / Conversion"].map("£{:.2f}".format)
    print("\n=== Business Value Summary ===")
    print(show.to_string(index=False))

    r, m = df_strat.set_index("Strategy").loc[["Random 30%", "Model top 30%"], "Conversions"]
    share, profit = profit_curve(y_true, y_prob)
    everyone = df_strat.set_index("Strategy").loc["Call everyone", "Net Profit (£)"]
    top30 = df_strat.set_index("Strategy").loc["Model top 30%", "Net Profit (£)"]
    print(f"\n  At a 30% budget the model wins {m / r:.1f}x as many subscriptions as random calling")
    print(f"  Calling only the top 30% keeps {top30 / everyone:.0%} of the profit of calling everyone"
          f" with {1 - BUDGET_SHARE:.0%} fewer calls")
    print(f"  Profit peaks at £{profit.max():,.0f} when calling the top {share[profit.argmax()]:.0%}")
    print(f"\n  Assumptions: £{REVENUE_PER_CONVERSION} revenue per subscription, £{COST_PER_CALL} per call")


if __name__ == "__main__":
    print("=== Business Value Pipeline ===\n")
    y_true, y_prob = load_data()

    df_strat = compute_strategy_comparison(y_true, y_prob)
    plot_strategy_comparison(df_strat, y_true, y_prob)
    plot_cumulative_lift(y_true, y_prob)
    plot_precision_at_k(y_true, y_prob)
    print_summary(df_strat, y_true, y_prob)

    df_strat.to_csv(OUTPUTS / "strategy_comparison.csv", index=False)
    print("\n✓ Business value analysis complete.")
