"""
01_eda.py — Exploratory Data Analysis
Bank Marketing Campaign: Predicting Term Deposit Subscriptions

Produces summary statistics, class balance overview, and key
distribution plots saved to the figures/ directory.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import INK, GOLD, GREY, LIGHT, DARK_GOLD, DIV, apply_style, frame, save

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data" / "bank_marketing.csv"
FIGS    = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

apply_style()


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Conversion rate: {(df.y == 'yes').mean():.1%}\n")
    return df


def summary(df: pd.DataFrame) -> None:
    """Print dataset overview."""
    print("=== Dataset Overview ===")
    print(df.dtypes.to_string())
    print("\n=== Missing Values ===")
    missing = df.isin(["unknown"]).sum()
    print(missing[missing > 0].to_string())
    print(f"\n=== Target Distribution ===")
    vc = df["y"].value_counts()
    for label, cnt in vc.items():
        print(f"  {label}: {cnt:,} ({cnt/len(df):.1%})")


def plot_class_balance(df: pd.DataFrame) -> None:
    counts = df["y"].value_counts().reindex(["no", "yes"])
    fig, ax = plt.subplots(figsize=(8, 2.6))
    fig.subplots_adjust(top=0.62, left=0.0, right=1.0)
    left = 0
    for label, col, txt in [("no", GREY, INK), ("yes", GOLD, DARK_GOLD)]:
        n = counts[label]
        ax.barh(0, n, left=left, color=col, height=0.6, edgecolor="white", lw=2)
        name = "Subscribed" if label == "yes" else "Did not subscribe"
        ax.text(left + n / 2 if label == "no" else left + n + 300, 0,
                f"{name}\n{n:,} ({n / len(df):.1%})", ha="center" if label == "no" else "left",
                va="center", fontsize=10, fontweight="bold", color="white" if label == "no" else txt)
        left += n
    ax.set_xlim(0, len(df) * 1.2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    frame(fig, "Only one call in nine ends in a subscription",
          "Outcome of 41,188 phone contacts, May 2008 to November 2010. Accuracy would mislead: "
          "\npredicting 'no' for everyone is right 89% of the time.")
    save(fig, FIGS / "01_class_balance.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = ["duration", "campaign", "age", "euribor3m",
                    "emp.var.rate", "cons.price.idx", "cons.conf.idx"]
    labels = {
        "age": "Age", "duration": "Call duration (s)",
        "campaign": "Contacts this campaign",
        "cons.price.idx": "Consumer price index",
        "cons.conf.idx": "Consumer confidence index",
        "euribor3m": "Euribor 3-month rate",
        "emp.var.rate": "Employment variation rate",
    }
    clip = {"duration": 1500, "campaign": 15}
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    fig.subplots_adjust(top=0.8, left=0.01, right=0.99, hspace=0.45, wspace=0.12)
    axes = axes.flatten()
    for ax, col in zip(axes, numeric_cols):
        x = df[col].clip(upper=clip.get(col, np.inf))
        bins = np.histogram_bin_edges(x, bins=35) if col != "campaign" else np.arange(0.5, 16.5)
        for label, colour, alpha in [("no", GREY, 0.55), ("yes", GOLD, 0.75)]:
            ax.hist(x[df["y"] == label], bins=bins, density=True, color=colour, alpha=alpha, lw=0)
        ax.set_title(labels[col], fontsize=10, color=DARK_GOLD if col == "duration" else INK)
        ax.set_yticks([])
        ax.grid(False)
    axes[0].text(0.98, 0.95, "known only after\nthe call, so excluded", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=8.5, color=DARK_GOLD)
    axes[-1].set_visible(False)
    fig.text(0.0, 0.875, "\u25a0", color=GOLD, fontsize=13, va="center")
    fig.text(0.02, 0.875, "Subscribed", color=DARK_GOLD, fontsize=9.5, fontweight="bold", va="center")
    fig.text(0.12, 0.875, "\u25a0", color=GREY, fontsize=13, va="center")
    fig.text(0.14, 0.875, "Did not subscribe", color=INK, fontsize=9.5, fontweight="bold", va="center")
    frame(fig, "Call duration separates subscribers best, but only after the call is over",
          "Distribution of each feature by outcome (each group scaled to the same area). "
          "The economic indicators take only\na handful of values because they change month by month, "
          "not client by client.")
    save(fig, FIGS / "02_numeric_distributions.png")


def plot_categorical_conversion(df: pd.DataFrame) -> None:
    cats = {
        "poutcome":  "Previous campaign outcome",
        "job":       "Job",
        "month":     "Month of last contact",
        "contact":   "Contact method",
        "education": "Education",
    }
    month_order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    overall = (df["y"] == "yes").mean()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    fig.subplots_adjust(top=0.84, left=0.09, right=0.99, hspace=0.3, wspace=0.65)
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, cats.items()):
        conv = df.groupby(col)["y"].apply(lambda s: (s == "yes").mean())
        if col == "poutcome":
            conv = conv.rename({"nonexistent": "not contacted before"})
        if col == "month":
            conv = conv.reindex(month_order[::-1])
        else:
            conv = conv.sort_values()
        colours = [GOLD if r >= 2 * overall else INK for r in conv]
        ax.barh(conv.index, conv.values, color=colours, height=0.65)
        for i, rate in enumerate(conv.values):
            ax.text(rate + 0.01, i, f"{rate:.0%}", va="center", fontsize=8.5,
                    color=DARK_GOLD if rate >= 2 * overall else INK)
        ax.axvline(overall, color=GREY, lw=1, ls="--")
        ax.set_title(title, fontsize=10.5)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.set_xlim(0, max(conv.max() * 1.25, 0.2))
        ax.tick_params(axis="y", labelsize=9)
    axes[3].text(overall + 0.003, 0.5, "average 11%", color=GREY, fontsize=8.5, va="center")
    axes[-1].set_visible(False)
    frame(fig, "A past subscription is the strongest signal: 65% of those clients subscribed again",
          "Share of contacts that ended in a subscription. Marigold marks groups converting at more than "
          "twice the\naverage (dashed line).")
    save(fig, FIGS / "03_categorical_conversion.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df.select_dtypes(include=np.number).corr()
    n = len(corr)
    rows, cols = corr.index[1:], corr.columns[:-1]   # drop the empty first row and last column
    shown = np.where(np.tril(np.ones((n, n), dtype=bool), k=-1), corr.values, np.nan)[1:, :-1]
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    fig.subplots_adjust(top=0.86, left=0.16, right=0.96)
    ax.imshow(shown, cmap=DIV, vmin=-1, vmax=1)
    econ = {"emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"}
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if np.isnan(shown[i, j]):
                continue
            v = shown[i, j] + 0.0
            strong = abs(v) >= 0.7 and r in econ and c in econ
            ax.text(j, i, f"{v:.2f}".replace("-0.00", "0.00"), ha="center", va="center", fontsize=8.5,
                    color=INK, fontweight="bold" if strong else "normal")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    frame(fig, "The economic indicators move together because they all track time",
          "Correlation between numeric features (marigold positive, blue negative). Euribor, the number "
          "employed\nand the employment variation rate are almost interchangeable (0.9 or more).")
    save(fig, FIGS / "04_correlation_heatmap.png")


if __name__ == "__main__":
    df = load_data()
    summary(df)
    plot_class_balance(df)
    plot_numeric_distributions(df)
    plot_categorical_conversion(df)
    plot_correlation_heatmap(df)
    print("\n✓ EDA complete. Figures saved to figures/")
