"""House chart style shared by every script in this project.

Every chart follows the same rules: a bold headline that states the finding,
a grey subtitle that says what is plotted, direct labels instead of legends
where possible, and a small source note. Colours: indigo for the main series,
marigold to highlight, grey for baselines and comparisons.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import offset_copy

INK, GOLD, BLUE, GREY, LIGHT = "#1D1B4C", "#F2A900", "#2F45C9", "#9A99B8", "#ECEBF5"
DARK_GOLD, SUBTLE = "#9A7400", "#5B5A7E"
SOURCE = "Source: UCI Bank Marketing dataset (Moro, Cortez and Rita, 2014), 41,188 contacts."

# sequential (white to indigo) and diverging (blue, white, marigold) colour maps
SEQ = LinearSegmentedColormap.from_list("house_seq", ["#FFFFFF", "#8C8BC0", INK])
DIV = LinearSegmentedColormap.from_list("house_div", [BLUE, "#FFFFFF", GOLD])
LOW_HIGH = LinearSegmentedColormap.from_list("house_low_high", [BLUE, "#B9B8D6", GOLD])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": GREY, "axes.labelcolor": INK, "axes.labelsize": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": LIGHT, "grid.linewidth": 1,
        "axes.axisbelow": True, "axes.facecolor": "white", "figure.facecolor": "white",
        "xtick.color": INK, "ytick.color": INK,
        "ytick.major.size": 0, "xtick.major.size": 0, "legend.frameon": False,
    })


def frame(fig, title: str, subtitle: str = "", source: str = SOURCE, top: float = 1.0) -> None:
    """Headline, subtitle and source note, aligned to the left edge of the figure.

    Call after fig.subplots_adjust(top=...) leaves room above the axes.
    """
    below = lambda pts: offset_copy(fig.transFigure, fig=fig, y=-pts, units="points")
    fig.text(0.0, top, title, fontsize=14, fontweight="bold", color=INK, ha="left", va="bottom")
    if subtitle:
        fig.text(0.0, top, subtitle, fontsize=10.5, color=SUBTLE, ha="left", va="top", transform=below(6))
    fig.text(0.0, 0.0, source, fontsize=8, color=GREY, ha="left", va="top", transform=below(14))


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path.name}")
