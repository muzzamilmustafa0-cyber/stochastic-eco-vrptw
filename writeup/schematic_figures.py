"""Framework/pipeline schematic and decision-focused methodology schematic."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "figures")
os.makedirs(FIG, exist_ok=True)


def _box(ax, x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.4, edgecolor=ec, facecolor=fc))
    ax.text(x + w/2, y + h - 0.22, title, ha="center", va="top", fontsize=10,
            fontweight="bold", color="#14233a")
    ax.text(x + w/2, y + h - 0.62, "\n".join(lines), ha="center", va="top", fontsize=8.3,
            color="#22364f", linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color="#5b6b7d", shrinkA=2, shrinkB=2))


def fig_framework():
    fig, ax = plt.subplots(figsize=(12, 3.6)); ax.set_xlim(0, 24); ax.set_ylim(0, 6); ax.axis("off")
    cols = [("#eaeef2", "#9aa7b5"), ("#dbe9f6", "#5b9bd5"), ("#d7f0ea", "#3bb39b"),
            ("#e7ddf3", "#8e6fc0"), ("#fce6d6", "#e08a4b"), ("#dcefdc", "#5aa860")]
    stages = [
        ("Real data", ["NYC TLC trips", "Austin / DSNY loads", "Dublin, Peshawar,", "Solomon"]),
        ("Prediction layer", ["demand quantiles", "travel-time dist.", "eco-speed", "feasibility"]),
        ("Scenario set", ["omega = {q, tau, z}", "sample-average", "approximation"]),
        ("Decision-focused", ["learn service", "levels tau_i;", "regime selection"]),
        ("Scenario-aware", ["HGLS operators,", "chance constraint,", "recourse + CVaR"]),
        ("Evaluation", ["out-of-sample", "E[cost], CVaR,", "overload"]),
    ]
    w, h, gap = 3.4, 3.2, 0.5; x = 0.3; y = 1.4
    centers = []
    for (title, lines), (fc, ec) in zip(stages, cols):
        _box(ax, x, y, w, h, title, lines, fc, ec); centers.append((x, x + w)); x += w + gap
    for i in range(len(stages) - 1):
        _arrow(ax, centers[i][1], y + h/2, centers[i+1][0], y + h/2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_framework.png"), dpi=300,
                                    bbox_inches="tight"); plt.close(fig)


def fig_method():
    fig, ax = plt.subplots(figsize=(12, 6.2)); ax.set_xlim(0, 23); ax.set_ylim(0, 12); ax.axis("off")
    blue = ("#dbe9f6", "#5b9bd5"); purple = ("#e7ddf3", "#8e6fc0")
    orange = ("#fce6d6", "#e08a4b"); green = ("#dcefdc", "#5aa860"); gray = ("#eaeef2", "#9aa7b5")
    ax.text(11.5, 11.4, "Decision-focused training loop and regime selection",
            ha="center", fontsize=12, fontweight="bold", color="#14233a")
    # top row (left -> right)
    _box(ax, 0.4, 7.6, 4.6, 2.7, "Node features  x_i",
         ["mean demand, CV,", "distance to depot,", "time-window width"], *gray)
    _box(ax, 6.0, 7.6, 5.0, 2.7, "Predictor  tau_i = g_theta(x_i)",
         ["learned per-node", "capacity service level"], *blue)
    _box(ax, 12.0, 7.6, 4.4, 2.7, "Planning demand",
         ["q_plan_i =", "Quantile(demand_i, tau_i)"], *purple)
    _box(ax, 17.4, 7.6, 5.2, 2.7, "Scenario-aware HGLS",
         ["operators; hard capacity", "at q_plan; fuel +", "recourse over scenarios"], *orange)
    # bottom row (right -> left): feedback loop
    _box(ax, 17.4, 3.4, 5.2, 2.7, "Realized cost",
         ["expected fuel + recourse", "on held-out scenarios", "(decision regret)"], *green)
    _box(ax, 6.0, 3.4, 5.0, 2.7, "Evolution strategy",
         ["perturb and update theta", "toward lower", "realized cost"], *blue)
    _box(ax, 0.4, 0.2, 22.2, 2.2, "Regime selection (Section 4.3)",
         ["the learned chance-constrained plan is optimized alongside the recourse-only, "
          "CVaR, quantile, and robust plans;", "the regime with the lowest training cost "
          "is deployed, so the method is never worse than the best standard plan"], *purple)
    # arrows: top row
    _arrow(ax, 5.0, 8.95, 6.0, 8.95)
    _arrow(ax, 11.0, 8.95, 12.0, 8.95)
    _arrow(ax, 16.4, 8.95, 17.4, 8.95)
    # HGLS -> realized cost (down)
    _arrow(ax, 20.0, 7.6, 20.0, 6.1)
    # realized cost -> evolution strategy (left)
    _arrow(ax, 17.4, 4.75, 11.0, 4.75)
    # evolution strategy -> predictor (up)  : clean vertical
    _arrow(ax, 8.5, 6.1, 8.5, 7.6)
    # evaluation -> regime selection (down)
    _arrow(ax, 20.0, 3.4, 20.0, 2.4)
    ax.text(14.0, 5.05, "decision-focused feedback", ha="center", fontsize=8.2,
            style="italic", color="#5b6b7d")
    fig.savefig(os.path.join(FIG, "fig_method_pipeline.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_framework(); fig_method()
    print("wrote fig_framework, fig_method_pipeline")
