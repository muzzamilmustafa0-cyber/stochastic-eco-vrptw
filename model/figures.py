"""Paper-ready figures from campaign + ablation results (PNG, 300 dpi)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results"); FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)
ORDER = ["D-HGLS", "PTO-HGLS", "Q-HGLS", "RO-Eco", "S-HGLS", "LA-SHGLS",
         "DF-det", "DF-CC", "DF-CC*"]
PROP = "DF-CC*"


def fig_method_bars():
    df = pd.read_csv(os.path.join(RES, "results.csv"))
    g = df.groupby("method")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, metric in enumerate(["E_cost", "CVaR_cost", "P_overload"]):
        mean = g[metric].mean().reindex(ORDER); sd = g[metric].std().reindex(ORDER)
        colors = ["#1f77b4" if m != PROP else "#d62728" for m in ORDER]
        ax[k].bar(range(len(ORDER)), mean.values, yerr=sd.values, color=colors, capsize=3)
        ax[k].set_xticks(range(len(ORDER))); ax[k].set_xticklabels(ORDER, rotation=45, ha="right")
        ax[k].set_title(metric); ax[k].grid(axis="y", alpha=0.3)
    fig.suptitle(f"Out-of-sample performance by method ({PROP} in red)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_method_bars.png"), dpi=300)
    plt.close(fig)


def fig_relgap_heatmap():
    df = pd.read_csv(os.path.join(RES, "results.csv"))
    piv = df.groupby(["instance", "method"])["E_cost"].mean().unstack("method")
    rel = piv.sub(piv[PROP], axis=0).div(piv[PROP], axis=0) * 100
    cols = [m for m in ORDER if m in rel.columns and m != PROP]
    rel = rel[cols]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(rel.values, cmap="RdYlGn_r", aspect="auto", vmin=-5, vmax=20)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rel.index))); ax.set_yticklabels(rel.index)
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            ax.text(j, i, f"{rel.values[i,j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label=f"E_cost % above {PROP} (green = {PROP} better)")
    ax.set_title(f"Per-instance cost gap of each method vs {PROP}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_relgap_heatmap.png"), dpi=300)
    plt.close(fig)


def fig_ablation():
    p = os.path.join(RES, "ablation.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    base = df[df["variant"] == "A0"].groupby("instance")["E_cost"].mean()
    g = df.groupby(["variant"])["E_cost"].mean()
    # delta vs A0 (positive = component removal hurts)
    a0 = g["A0"]
    delta = (g - a0) / a0 * 100
    labels = {"A1": "-feasibility", "A2": "-stoch demand", "A3": "-stoch tt",
              "A4": "-recourse", "A5": "-CVaR", "A6": "-decision-focus"}
    vs = [v for v in ["A1", "A2", "A3", "A4", "A5", "A6"] if v in delta.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([labels[v] for v in vs], [delta[v] for v in vs], color="#ff7f0e")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("% E_cost increase vs full model (A0)")
    ax.set_title("Ablation: cost penalty from removing each component")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ablation.png"), dpi=300)
    plt.close(fig)


def main():
    fig_method_bars(); fig_relgap_heatmap(); fig_ablation()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
