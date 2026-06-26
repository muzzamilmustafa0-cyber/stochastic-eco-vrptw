"""Conceptual model figures (fuel-load-emission model; eco-speed trade-off)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "figures")
os.makedirs(FIG, exist_ok=True)

RHO0, RHOQ = 0.20, 0.50          # L/km empty / full load
PHI = {"low": 0.90, "medium": 1.00, "high": 1.18}
XI = 2.36                         # kg CO2e per litre


def fig_fuel_model():
    frac = np.linspace(0, 1, 50)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for s, c in zip(["low", "medium", "high"], ["#27ae60", "#e67e22", "#c0392b"]):
        rho = (RHO0 + (RHOQ - RHO0) * frac) * PHI[s]
        ax[0].plot(frac, rho, color=c, label=f"{s} speed")
    ax[0].set_xlabel("payload fraction  L / Q")
    ax[0].set_ylabel("fuel rate  (L/km)")
    ax[0].set_title("(a) Load- and speed-dependent fuel rate")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    fuel = np.linspace(0, 60, 50)
    ax[1].plot(fuel, XI * fuel, color="#34495e")
    ax[1].set_xlabel("fuel consumed (L)"); ax[1].set_ylabel("CO2e emissions (kg)")
    ax[1].set_title(f"(b) Emissions proportional to fuel (xi = {XI})")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_fuel_model.png"), dpi=300)
    plt.close(fig)


def fig_speed_levels():
    d = 2.0                       # representative arc length (km)
    v = {"low": 30, "medium": 50, "high": 70}      # nominal cruising km/h
    names = list(v)
    tt = [d / v[s] * 60 for s in names]            # minutes
    rho = [(RHO0 + (RHOQ - RHO0) * 0.5) * PHI[s] for s in names]   # at half load
    fig, ax1 = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(names)); w = 0.35
    b1 = ax1.bar(x - w/2, tt, w, color="#2980b9", label="travel time")
    ax1.set_ylabel("travel time on arc (min)", color="#2980b9")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{s}\n({v[s]} km/h)" for s in names])
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w/2, rho, w, color="#c0392b", label="fuel rate")
    ax2.set_ylabel("fuel rate at half load (L/km)", color="#c0392b")
    ax1.set_title("Eco-speed trade-off: faster service costs more fuel per km")
    ax1.set_xlabel("eco-speed level")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_speed_levels.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_fuel_model(); fig_speed_levels()
    print("wrote fig_fuel_model, fig_speed_levels")
