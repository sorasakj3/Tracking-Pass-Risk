from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


def risk_map(test_frame: pd.DataFrame, output: str | Path) -> None:
    ranked = test_frame.sort_values("predicted_turnover_risk", ascending=False).head(35)
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#101612")
    ax.set_facecolor("#101612")
    ax.add_patch(Rectangle((0, 0), 105, 68, fill=False, color="#d7ddd5"))
    ax.plot([52.5, 52.5], [0, 68], color="#d7ddd5", lw=0.8)
    for row in ranked.itertuples():
        colour = "#ef4035" if row.receiver_pressured_at_arrival else "#f0b44d"
        width = 0.7 + 2.8 * row.predicted_turnover_risk
        arrow = FancyArrowPatch(
            (row.start_x_m, row.start_y_m),
            (row.end_x_m, row.end_y_m),
            arrowstyle="-|>",
            mutation_scale=8,
            lw=width,
            color=colour,
            alpha=0.6,
        )
        ax.add_patch(arrow)
    ax.text(2, 5, "TRACKING PASS RISK", color="#ef4035", fontsize=11, weight="bold")
    ax.text(2, 10, "Options likely to arrive under pressure", color="#f5f2e9", fontsize=19, weight="bold")
    ax.text(2, 15, "Top test-period predictions · red = defender within 8m at arrival", color="#d7ddd5", fontsize=10)
    ax.set(xlim=(0, 105), ylim=(68, 0), aspect="equal")
    ax.axis("off")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
