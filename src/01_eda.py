"""
01_eda.py — Exploratory Data Analysis
Bank Marketing Campaign: Predicting Term Deposit Subscriptions

Produces summary statistics, class balance overview, and key
distribution plots saved to the figures/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data" / "bank_marketing.csv"
FIGS    = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
PALETTE = {"yes": "#2563EB", "no": "#94A3B8"}
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


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
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["y"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[k] for k in counts.index],
                  edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 400,
                f"{val:,}\n({val/len(df):.1%})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Term Deposit Subscription: Class Balance", fontweight="bold", pad=12)
    ax.set_xlabel("Subscribed?")
    ax.set_ylabel("Number of Clients")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.18)
    sns.despine(left=True)
    fig.savefig(FIGS / "01_class_balance.png")
    plt.close()
    print("Saved: 01_class_balance.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = ["age", "duration", "campaign", "cons.price.idx",
                    "cons.conf.idx", "euribor3m", "emp.var.rate"]
    labels = {
        "age": "Age", "duration": "Call Duration (s)",
        "campaign": "Contacts This Campaign",
        "cons.price.idx": "Consumer Price Index",
        "cons.conf.idx": "Consumer Confidence Index",
        "euribor3m": "Euribor 3-Month Rate",
        "emp.var.rate": "Employment Variation Rate"
    }
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        for label, grp in df.groupby("y"):
            grp[col].hist(ax=ax, bins=35, alpha=0.65,
                          color=PALETTE[label], label=label, density=True)
        ax.set_title(labels[col], fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density" if i % 4 == 0 else "")
        ax.legend(title="Subscribed", fontsize=8)
        sns.despine(ax=ax, left=True)
    axes[-1].set_visible(False)
    fig.suptitle("Numeric Feature Distributions by Subscription Outcome",
                 fontweight="bold", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "02_numeric_distributions.png")
    plt.close()
    print("Saved: 02_numeric_distributions.png")


def plot_categorical_conversion(df: pd.DataFrame) -> None:
    cats = {
        "job":       "Job Type",
        "education": "Education Level",
        "contact":   "Contact Method",
        "month":     "Last Contact Month",
        "poutcome":  "Previous Campaign Outcome"
    }
    month_order = ["mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (col, title) in enumerate(cats.items()):
        ax = axes[i]
        conv = (df.groupby(col)["y"]
                  .apply(lambda s: (s == "yes").mean())
                  .reset_index()
                  .rename(columns={"y": "conversion_rate"}))
        if col == "month":
            conv["month"] = pd.Categorical(conv["month"], categories=month_order, ordered=True)
            conv = conv.sort_values("month")
        else:
            conv = conv.sort_values("conversion_rate", ascending=False)

        bars = ax.barh(conv[col], conv["conversion_rate"],
                       color="#2563EB", alpha=0.85, edgecolor="white")
        for bar, rate in zip(bars, conv["conversion_rate"]):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{rate:.0%}", va="center", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xlim(0, conv["conversion_rate"].max() * 1.25)
        sns.despine(ax=ax, left=True)

    axes[-1].set_visible(False)
    fig.suptitle("Conversion Rate by Categorical Feature",
                 fontweight="bold", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "03_categorical_conversion.png")
    plt.close()
    print("Saved: 03_categorical_conversion.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
                cmap="RdBu_r", center=0, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Correlation Matrix — Numeric Features", fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIGS / "04_correlation_heatmap.png")
    plt.close()
    print("Saved: 04_correlation_heatmap.png")


if __name__ == "__main__":
    df = load_data()
    summary(df)
    plot_class_balance(df)
    plot_numeric_distributions(df)
    plot_categorical_conversion(df)
    plot_correlation_heatmap(df)
    print("\n✓ EDA complete. Figures saved to figures/")
