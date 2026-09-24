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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pickle
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data"
FIGS    = ROOT / "figures"
OUTPUTS = ROOT / "outputs"

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

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
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.4]})

    # Left: same budget, different targeting
    sub = df_strat[df_strat.Strategy.isin(["Random 30%", "Model top 30%"])]
    bars = axes[0].bar(sub.Strategy, sub.Conversions, color=["#94A3B8", "#2563EB"], width=0.5)
    for bar, (_, r) in zip(bars, sub.iterrows()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f"{r.Conversions:,} subscribers\n{r['Conversion Rate']:.0%} conversion",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, sub.Conversions.max() * 1.3)
    axes[0].set_title(f"Same budget ({sub['Clients Contacted'].iloc[0]:,} calls), different targeting",
                      fontweight="bold")
    axes[0].set_ylabel("Subscriptions won")
    sns.despine(ax=axes[0], left=True)

    # Right: profit by depth of calling
    share, profit = profit_curve(y_true, y_prob)
    axes[1].plot(share * 100, profit, color="#2563EB", lw=2.5, label="Call the top-scored x%")
    rand_profit = share * len(y_true) * (y_true.mean() * REVENUE_PER_CONVERSION - COST_PER_CALL)
    axes[1].plot(share * 100, rand_profit, "k--", lw=1, label="Call a random x%")
    best = profit.argmax()
    for x, lab in [(BUDGET_SHARE, "Top 30%"), (share[best], "Maximum profit"), (1.0, "Everyone")]:
        i = min(int(x * len(share)) - 1, len(share) - 1) if lab != "Maximum profit" else best
        axes[1].scatter(share[i] * 100, profit[i], color="#DC2626", zorder=3)
        axes[1].annotate(f"{lab}\n£{profit[i]:,.0f}", (share[i] * 100, profit[i]),
                         textcoords="offset points",
                         xytext={"Top 30%": (0, -40), "Maximum profit": (-55, -40), "Everyone": (0, -40)}[lab],
                         ha="right" if lab == "Everyone" else "center",
                         fontsize=9, color="#DC2626")
    axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"£{int(x):,}"))
    axes[1].set_xlabel("% of clients called (highest scores first)")
    axes[1].set_ylabel("Net profit")
    axes[1].set_title("Net profit by how far down the list we call", fontweight="bold")
    axes[1].legend(loc="lower right")
    sns.despine(ax=axes[1])

    fig.suptitle("Campaign Strategy Comparison: Business Impact", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "09_strategy_comparison.png")
    plt.close()
    print("Saved: 09_strategy_comparison.png")


def plot_cumulative_lift(y_true, y_prob) -> None:
    """
    Cumulative gains (lift) curve: what fraction of all converters
    do we capture if we contact the top-X% of the scored list?
    """
    sort_idx   = np.argsort(y_prob)[::-1]
    y_sorted   = y_true[sort_idx]
    cum_conv   = np.cumsum(y_sorted)
    total_conv = y_true.sum()
    n          = len(y_true)

    pct_contacted = np.arange(1, n + 1) / n
    pct_captured  = cum_conv / total_conv
    random_line   = pct_contacted

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative gains
    axes[0].plot(pct_contacted * 100, pct_captured * 100,
                 color="#2563EB", lw=2.5, label="Model")
    axes[0].plot([0, 100], [0, 100], "k--", lw=1, label="Random baseline")
    axes[0].fill_between(pct_contacted * 100, pct_captured * 100,
                         random_line * 100, alpha=0.08, color="#2563EB")
    axes[0].set_xlabel("% of Clients Contacted")
    axes[0].set_ylabel("% of Converters Captured")
    axes[0].set_title("Cumulative Gains Curve", fontweight="bold")
    axes[0].legend()
    # Annotate 30% mark
    idx_30 = int(0.30 * n)
    axes[0].annotate(f"Top 30%:\n{pct_captured[idx_30]:.0%} of converters",
                     xy=(30, pct_captured[idx_30] * 100),
                     xytext=(45, pct_captured[idx_30] * 100 - 12),
                     arrowprops=dict(arrowstyle="->", color="#DC2626"),
                     color="#DC2626", fontsize=9)
    sns.despine(ax=axes[0])

    # Lift curve
    lift = pct_captured / pct_contacted
    axes[1].plot(pct_contacted * 100, lift,
                 color="#16A34A", lw=2.5, label="Lift")
    axes[1].axhline(1, color="k", lw=1, linestyle="--", label="No lift")
    axes[1].set_xlabel("% of Clients Contacted")
    axes[1].set_ylabel("Lift over Random")
    axes[1].set_title("Lift Curve", fontweight="bold")
    axes[1].legend()
    axes[1].set_xlim(0, 100)
    sns.despine(ax=axes[1])

    fig.suptitle("Model Lift: Efficiency Gains over Random Outreach",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "10_lift_curve.png")
    plt.close()
    print("Saved: 10_lift_curve.png")


def plot_precision_at_k(y_true, y_prob) -> None:
    """Precision@K: how accurate is our targeting at each contact threshold?"""
    sort_idx = np.argsort(y_prob)[::-1]
    y_sorted = y_true[sort_idx]
    n        = len(y_true)
    k_range  = np.arange(1, n + 1)
    prec_at_k = np.cumsum(y_sorted) / k_range

    pct_range = k_range / n * 100
    baseline  = y_true.mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pct_range, prec_at_k, color="#2563EB", lw=2.5, label="Model precision@k")
    ax.axhline(baseline, color="#94A3B8", lw=1.5, linestyle="--",
               label=f"Baseline (overall rate: {baseline:.1%})")

    # Shade top 15% and 30% zones
    ax.axvspan(0, 15, alpha=0.07, color="#16A34A", label="Top 15%")
    ax.axvspan(15, 30, alpha=0.05, color="#2563EB", label="Top 15–30%")

    ax.set_xlabel("% of Scored List Contacted (highest → lowest)")
    ax.set_ylabel("Precision (Conversion Rate Among Contacted)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Precision@K — Targeting Efficiency Curve", fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 80)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGS / "11_precision_at_k.png")
    plt.close()
    print("Saved: 11_precision_at_k.png")


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
