"""Figures for the revised study. Data-independent figures (calibration,
dependence, decision timeline) build immediately; campaign figures build once
results2 CSVs exist."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results2")
FIG = os.path.join(RES, "figures")
PROC = os.path.join(ROOT, "data_pipeline", "processed")
os.makedirs(FIG, exist_ok=True)

COLORS = {"DET": "#7f8c8d", "PTO": "#95a5a6", "SAA": "#2980b9", "CVAR": "#16a085",
          "Q90C": "#8e44ad", "RO": "#d35400", "DFR": "#c0392b", "ORT": "#27ae60"}


# ------------------------------------------------ calibration (available now) --
def fig_calibration():
    v = json.load(open(os.path.join(PROC, "prediction_validation.json")))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # (a) demand quantile reliability
    qr = v["demand"]["quantile_reliability"]
    nom = [float(k) for k in qr]; emp = list(qr.values())
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].plot(nom, emp, "o-", color="#2980b9")
    ax[0].set_xlabel("nominal quantile"); ax[0].set_ylabel("empirical frequency")
    ax[0].set_title("(a) Demand quantile reliability")
    ax[0].grid(alpha=0.3)
    # (b) conformal effect on interval coverage
    labels = ["demand raw", "demand conf.", "travel raw", "travel conf."]
    vals = [v["demand"]["cover80_raw"], v["demand"]["cover80_conformal"],
            v["travel_time"]["cover80_raw"], v["travel_time"]["cover80_conformal"]]
    cols = ["#95a5a6", "#2980b9", "#95a5a6", "#2980b9"]
    ax[1].bar(range(4), vals, color=cols)
    ax[1].axhline(0.8, color="k", ls="--", lw=1, label="nominal 0.80")
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labels, rotation=20)
    ax[1].set_ylim(0.5, 1.0); ax[1].set_ylabel("80% interval coverage")
    ax[1].set_title("(b) Conformal calibration effect"); ax[1].legend()
    ax[1].grid(axis="y", alpha=0.3)
    # (c) feasibility reliability curves
    for name, c in [("low", "#27ae60"), ("med", "#e67e22"), ("high", "#c0392b")]:
        rc = v["feasibility"]["reliability_curves"][name]
        ax[2].plot(rc["pred"], rc["obs"], "o-", color=c,
                   label=f"{name} (Brier {v['feasibility']['brier'][name]})")
    ax[2].plot([0, 1], [0, 1], "k--", lw=1)
    ax[2].set_xlabel("predicted feasibility probability")
    ax[2].set_ylabel("observed frequency")
    ax[2].set_title("(c) Achievable-speed feasibility calibration")
    ax[2].legend(); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_calibration.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------ dependence (available now) --
def fig_dependence():
    from . import scenlib as SL
    sc_dep = SL.generate("nyc_manhattan", 1)
    sc_ind = SL.generate("nyc_manhattan", 1, rho_d=0.0, sig_t=0.0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, sc, ttl in [(ax[0], sc_ind, "(a) Independent sampling"),
                       (ax[1], sc_dep, "(b) Calibrated joint sampling")]:
        tot = sc.q[:, 1:].sum(1)
        a.hist(tot, bins=30, color="#2980b9", alpha=0.8)
        a.set_xlabel("total daily demand (m3)"); a.set_ylabel("scenarios")
        a.set_title(f"{ttl}\nstd of total = {tot.std():.1f}")
        a.grid(alpha=0.3)
    fig.suptitle("Cross-node dependence widens the distribution of total demand "
                 "(rho = %.3f from Austin day effects)" % SL.RHO_D)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dependence.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------ decision timeline (now) -----
def fig_timeline():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 24); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, w, y, h, txt, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     linewidth=1.3, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=9, color="#14233a")
    ax.annotate("", xy=(23.6, 1.0), xytext=(0.4, 1.0),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#5b6b7d"))
    ax.text(12, 0.45, "time", ha="center", fontsize=9, color="#5b6b7d")
    box(0.6, 6.8, 2.0, 3.4, "FIRST STAGE (before the shift)\n"
        "routes and sequences\nplanned eco-speed levels\n"
        "capacity service levels", "#dbe9f6", "#5b9bd5")
    box(8.2, 6.6, 2.0, 3.4, "INFORMATION REVELATION\n"
        "bin fill observed at service\nachievable speeds realised\n"
        "by arc and departure time", "#fdf3d9", "#d4a017")
    box(15.6, 7.9, 2.0, 3.4, "SECOND STAGE (recourse)\n"
        "speed reduced to achievable\nemergency depot return when full\n"
        "lateness absorbed at penalty\ndeferment only at shift horizon",
        "#dcefdc", "#5aa860")
    ax.text(4.0, 5.7, "here-and-now decisions x", ha="center", fontsize=8.5,
            style="italic", color="#5b6b7d")
    ax.text(19.5, 5.7, "scenario-dependent recourse y(x, omega)", ha="center",
            fontsize=8.5, style="italic", color="#5b6b7d")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_timeline.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------ campaign figures ------------
def fig_shift_degradation():
    p = os.path.join(RES, "shift_degradation.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    kinds = [("demand_mean", "demand mean shift"),
             ("congestion", "congestion shift")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, (kind, ttl) in zip(ax, kinds):
        sub = d[d["kind"] == kind]
        for m in ["DET", "SAA", "Q90C", "RO", "DFR"]:
            if m in sub.columns:
                a.plot(sub["mag"] * 100, sub[m], "o-", color=COLORS[m], label=m)
        a.set_xlabel("shift magnitude (%)"); a.set_ylabel("cost degradation (%)")
        a.set_title(ttl); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("Out-of-sample degradation under distribution shift "
                 "(flatter is more robust)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_shift.png"), dpi=300)
    plt.close(fig)


def fig_scount():
    p = os.path.join(RES, "scount.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for fam, c in [("nyc_manhattan", "#2980b9"), ("peshawar_real", "#c0392b"),
                   ("solomon_c101", "#27ae60")]:
        sub = d[d["family"] == fam]
        g = sub.groupby("S_train")["E_cost_test"]
        mean, sd = g.mean(), g.std()
        base = mean.iloc[-1]
        ax.errorbar(mean.index, mean / base * 100, yerr=sd / base * 100,
                    fmt="o-", color=c, capsize=3, label=fam)
    ax.set_xlabel("training scenario count"); ax.set_xscale("log")
    ax.set_ylabel("out-of-sample cost (% of S=200 mean)")
    ax.set_title("Scenario-count stability (5 independent sets per point)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_scount.png"), dpi=300)
    plt.close(fig)


def fig_ablation_ladder():
    p = os.path.join(RES, "ablation.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    order = ["L0_det", "L1_scen_blind", "L2_recourse", "L3_dependence",
             "L4_conformal", "L5_dro", "L6_policy", "L7_safeguard"]
    labels = ["deterministic", "+scenarios\n(recourse-blind)", "+recourse\nobjective",
              "+joint\ndependence", "+conformal\nprotection", "+DRO",
              "+learned\npolicy", "+safeguard"]
    piv = d.groupby(["family", "rep", "step"])["E_cost"].mean().unstack("step")
    rel = piv.div(piv["L0_det"], axis=0) * 100
    mean = rel[order].mean(); sd = rel[order].std()
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.errorbar(range(len(order)), mean, yerr=sd, fmt="o-", color="#c0392b",
                capsize=3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("test cost (% of deterministic plan)")
    ax.set_title("Additive component ladder (fixed evaluation environment)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ablation2.png"), dpi=300)
    plt.close(fig)


def fig_frontier():
    p = os.path.join(RES, "main.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    g = d.groupby("method")[["E_cost", "CVaR_cost", "P_trigger", "n_veh"]].mean()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for m, r in g.iterrows():
        star = (m == "DFR")
        for a, yk in [(ax[0], "CVaR_cost"), (ax[1], "P_trigger")]:
            a.scatter(r["E_cost"], r[yk], s=170 if star else 80,
                      c=COLORS.get(m, "#333"), marker="*" if star else "o",
                      edgecolors="k", linewidths=0.5, zorder=3 if star else 2)
            a.annotate(m, (r["E_cost"], r[yk]), fontsize=8,
                       xytext=(4, 4), textcoords="offset points")
    ax[0].set_xlabel("expected cost"); ax[0].set_ylabel("CVaR(0.9) of cost")
    ax[1].set_xlabel("expected cost"); ax[1].set_ylabel("route-failure probability")
    ax[0].set_title("Cost against tail risk"); ax[1].set_title("Cost against failures")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_frontier.png"), dpi=300)
    plt.close(fig)


def build_now():
    fig_calibration(); fig_dependence(); fig_timeline()
    print("built calibration, dependence, timeline")


def build_post():
    fig_shift_degradation(); fig_scount(); fig_ablation_ladder(); fig_frontier()
    print("built campaign figures")


if __name__ == "__main__":
    build_now(); build_post()
