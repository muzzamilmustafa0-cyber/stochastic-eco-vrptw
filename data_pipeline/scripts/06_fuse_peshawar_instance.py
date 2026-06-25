"""
06 - Peshawar WSSP Zone D CASE STUDY instance (REAL geometry/demand/time windows).

Source: raw/peshawar_wssp_zoneD.csv parsed from the manuscript appendix table
(111 real nodes: lat, lon, time windows, demand m3, service minutes).

  geometry      <- REAL WSSP Zone D coordinates
  demand level  <- REAL per-node demand (m3) from register
  demand spread <- Austin per-route CV                       REAL-CALIBRATED
  travel time   <- distance x real NYC congestion-by-hour    REAL-CALIBRATED
  time windows  <- REAL from register
Output: instances/peshawar_real/{instance.json, scenarios.npz, meta.json}
"""
import os, json
import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw"); PROC = os.path.join(HERE, "processed")
SRC = os.path.join(RAW, "peshawar_wssp_zoneD.csv")
OUT = os.path.join(HERE, "instances", "peshawar_real"); os.makedirs(OUT, exist_ok=True)

N_SCEN = 60
PESH_FREEFLOW_KMH = 36.0      # dense South-Asian city urban free-flow
VEH_CAP_M3 = 12.0

df = pd.read_csv(SRC).sort_values("id").reset_index(drop=True)
df["service_min"] = df["service_min"].fillna(df["service_min"].median())
df["demand_m3"] = df["demand_m3"].fillna(df["demand_m3"].median())
# node 0 is depot
lat = df["lat"].values; lon = df["lon"].values
demand = df["demand_m3"].values.copy(); demand[0] = 0.0
tw = df[["tw_start", "tw_end"]].fillna(method="ffill").values
serv = df["service_min"].values.copy(); serv[0] = 0.0
Np = len(df)

od = pd.read_parquet(os.path.join(PROC, "nyc_od_traveltime.parquet"))
austin = pd.read_csv(os.path.join(PROC, "austin_route_demand.csv"))
austin_ok = austin[(austin["mean_kg"] > 0) & austin["cv"].between(0.05, 1.0)].reset_index(drop=True)
hourly = od.groupby("hour")["mean_mph"].median(); cong = (hourly/hourly.max()).to_dict()

def haversine(la1, lo1, la2, lo2):
    R = 6371.0; p1, p2 = np.radians(la1), np.radians(la2)
    dphi = np.radians(la2-la1); dlmb = np.radians(lo2-lo1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
D = np.zeros((Np, Np))
for i in range(Np):
    D[i] = haversine(lat[i], lon[i], lat, lon)
np.fill_diagonal(D, 0.0)

shift_hours = list(range(8, 17))
ach = np.array([PESH_FREEFLOW_KMH*cong.get(h, 1.0) for h in shift_hours])
ach_s = np.clip(rng.lognormal(np.log(ach.mean())-0.05, 0.25, 5000), 5, PESH_FREEFLOW_KMH+5)
SPEED = {"low": round(float(np.percentile(ach_s, 30)), 1),
         "med": round(float(np.percentile(ach_s, 60)), 1),
         "high": round(float(np.percentile(ach_s, 85)), 1)}
sp_kmh = np.array([SPEED[s] for s in ["low", "med", "high"]])

pick = rng.choice(len(austin_ok), size=Np-1, replace=False)
node_cv = austin_ok.loc[pick, "cv"].values

q = np.zeros((N_SCEN, Np)); tt = np.zeros((N_SCEN, Np, Np, 3)); feas = np.zeros((N_SCEN, Np, Np, 3), dtype=np.int8)
for s in range(N_SCEN):
    for n in range(1, Np):
        cv = node_cv[n-1]; base = max(demand[n], 0.2)
        mu = np.log(base) - 0.5*np.log(1+cv**2)
        q[s, n] = rng.lognormal(mu, np.sqrt(np.log(1+cv**2)))
    hour = int(rng.choice(shift_hours)); base_v = PESH_FREEFLOW_KMH*cong.get(hour, 1.0)
    for i in range(Np):
        for j in range(Np):
            if i == j: continue
            v = float(np.clip(rng.lognormal(np.log(base_v)-0.03, 0.22), 5, PESH_FREEFLOW_KMH+5))
            for si, vlev in enumerate(sp_kmh):
                tt[s, i, j, si] = D[i, j]/max(min(vlev, v), 1e-3)*60.0
                feas[s, i, j, si] = int(v >= vlev - 3)

np.savez_compressed(os.path.join(OUT, "scenarios.npz"), q=q, tt=tt, feas=feas, sp_kmh=sp_kmh, dist=D)
inst = {"name": "peshawar_real", "n_nodes": Np, "depot": 0, "lat": lat.tolist(), "lon": lon.tolist(),
        "distance_km": D.round(4).tolist(), "vehicle_capacity_m3": VEH_CAP_M3,
        "service_time_min": serv.tolist(), "shift": [8.0, 16.0], "speed_levels_kmh": SPEED,
        "base_demand_m3": demand.round(3).tolist(), "demand_cv": [0.0]+node_cv.round(3).tolist(),
        "time_window": [[float(tw[i, 0]), float(tw[i, 1])] for i in range(Np)]}
json.dump(inst, open(os.path.join(OUT, "instance.json"), "w"), indent=2)
meta = {"name": "peshawar_real", "n_nodes": Np, "n_scenarios": N_SCEN, "role": "CASE STUDY",
        "geometry": "REAL WSSP Zone D coordinates (manuscript register)",
        "demand": "REAL per-node demand (m3) x REAL-CALIBRATED CV (Austin)",
        "time_windows": "REAL WSSP register", "travel_time": "REAL-CALIBRATED distance x NYC congestion",
        "speed_levels_kmh": SPEED, "low_feasible_frac": float(feas[..., 0].mean()),
        "high_feasible_frac": float(feas[..., 2].mean()), "demand_mean_m3": float(q[:, 1:].mean())}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
print(json.dumps(meta, indent=2))
