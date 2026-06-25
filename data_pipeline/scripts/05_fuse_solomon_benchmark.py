"""
05 - Eco-stochastic Solomon BENCHMARK family (controlled comparison + comparability
to classical VRPTW literature).

Keeps the classical Solomon geometry, demand levels, time windows, service times
(BENCHMARK) and injects the stochastic eco layers:
  demand spread <- Austin per-route CV                          REAL-CALIBRATED
  travel time   <- distance x real NYC congestion-by-hour regime REAL-CALIBRATED
  eco-speed feas<- data-driven levels vs achievable speed        REAL-CALIBRATED

Solomon coords are unitless; we treat 1 coord unit = 1 km and use a free-flow base
speed so eco-speed levels are well separated. Output: instances/solomon_<name>/...
"""
import os, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw"); PROC = os.path.join(HERE, "processed")
N_SCEN = 60; FREEFLOW_KMH = 60.0
SHIFT_HOURS = list(range(6, 19))

od = pd.read_parquet(os.path.join(PROC, "nyc_od_traveltime.parquet"))
austin = pd.read_csv(os.path.join(PROC, "austin_route_demand.csv"))
austin_ok = austin[(austin["mean_kg"] > 0) & austin["cv"].between(0.05, 1.0)].reset_index(drop=True)
hourly_mph = od.groupby("hour")["mean_mph"].median()
cong = (hourly_mph / hourly_mph.max()).to_dict()


def parse_solomon(path):
    lines = open(path).read().splitlines()
    name = lines[0].strip()
    cap = None; rows = []; in_cust = False
    for ln in lines:
        s = ln.split()
        if not s:
            continue
        if s[0] == "NUMBER":
            continue
        if len(s) == 2 and cap is None and s[0].isdigit():
            cap = float(s[1]); continue
        if s[0] == "CUST":
            in_cust = True; continue
        if in_cust and s[0].isdigit():
            # CUST XCOORD YCOORD DEMAND READY DUE SERVICE
            rows.append([float(x) for x in s[:7]])
    arr = np.array(rows)
    return name, cap, arr


def build(path, seed):
    rng = np.random.default_rng(seed)
    name, cap, arr = parse_solomon(path)
    # full 100 customers + depot (matches classical Solomon scale)
    arr = arr[:101]
    Np = len(arr)
    x, y = arr[:, 1], arr[:, 2]
    demand = arr[:, 3].copy(); demand[0] = 0.0
    ready, due, serv = arr[:, 4], arr[:, 5], arr[:, 6]
    D = np.sqrt((x[:, None]-x[None, :])**2 + (y[:, None]-y[None, :])**2)

    ach = np.array([FREEFLOW_KMH*cong.get(h, 1.0) for h in SHIFT_HOURS])
    ach_s = np.clip(rng.lognormal(np.log(ach.mean())-0.05, 0.25, 5000), 8, FREEFLOW_KMH+5)
    SPEED = {"low": round(float(np.percentile(ach_s, 30)), 1),
             "med": round(float(np.percentile(ach_s, 60)), 1),
             "high": round(float(np.percentile(ach_s, 85)), 1)}
    sp_kmh = np.array([SPEED[s] for s in ["low", "med", "high"]])

    pick = rng.choice(len(austin_ok), size=Np-1, replace=False)
    node_cv = austin_ok.loc[pick, "cv"].values

    q = np.zeros((N_SCEN, Np)); tt = np.zeros((N_SCEN, Np, Np, 3)); feas = np.zeros((N_SCEN, Np, Np, 3), dtype=np.int8)
    for s in range(N_SCEN):
        for n in range(1, Np):
            cv = node_cv[n-1]; base = max(demand[n], 1.0)
            mu = np.log(base) - 0.5*np.log(1+cv**2)
            q[s, n] = rng.lognormal(mu, np.sqrt(np.log(1+cv**2)))
        hour = int(rng.choice(SHIFT_HOURS)); base_v = FREEFLOW_KMH*cong.get(hour, 1.0)
        for i in range(Np):
            for j in range(Np):
                if i == j: continue
                v = float(np.clip(rng.lognormal(np.log(base_v)-0.03, 0.22), 8, FREEFLOW_KMH+5))
                for si, vlev in enumerate(sp_kmh):
                    tt[s, i, j, si] = D[i, j] / max(min(vlev, v), 1e-3) * 60.0
                    feas[s, i, j, si] = int(v >= vlev - 3)

    out = os.path.join(HERE, "instances", f"solomon_{name.lower()}"); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "scenarios.npz"), q=q, tt=tt, feas=feas, sp_kmh=sp_kmh, dist=D)
    inst = {"name": f"solomon_{name.lower()}", "n_nodes": Np, "depot": 0,
            "x": x.tolist(), "y": y.tolist(), "distance_km": D.round(4).tolist(),
            "vehicle_capacity_m3": float(cap), "service_time_min": serv.tolist(),
            "speed_levels_kmh": SPEED, "base_demand_m3": demand.round(3).tolist(),
            "demand_cv": [0.0]+node_cv.round(3).tolist(),
            "time_window": [[float(ready[i]), float(due[i])] for i in range(Np)]}
    json.dump(inst, open(os.path.join(out, "instance.json"), "w"), indent=2)
    meta = {"name": f"solomon_{name.lower()}", "family": name[:2], "n_nodes": Np, "n_scenarios": N_SCEN,
            "geometry": "BENCHMARK Solomon (classical)", "time_windows": "BENCHMARK Solomon",
            "demand": "BENCHMARK level x REAL-CALIBRATED CV (Austin)",
            "travel_time": "REAL-CALIBRATED distance x NYC congestion regime",
            "speed_levels_kmh": SPEED, "low_feasible_frac": float(feas[..., 0].mean()),
            "high_feasible_frac": float(feas[..., 2].mean())}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"  {name:6s} N={Np} cap={cap:.0f} speeds={SPEED} feas(low/high)={meta['low_feasible_frac']:.2f}/{meta['high_feasible_frac']:.2f}")


if __name__ == "__main__":
    print("Building eco-stochastic Solomon benchmark...")
    for k, f in enumerate(sorted(glob.glob(os.path.join(RAW, "solomon", "*.txt")))):
        build(f, seed=100+k)
