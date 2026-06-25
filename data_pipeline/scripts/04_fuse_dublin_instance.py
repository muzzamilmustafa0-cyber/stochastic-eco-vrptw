"""
04 - DATA FUSION: Dublin (Fingal) LA-SEco-ECVRPTW instance + scenario set.

  geometry      <- 190 REAL Fingal solar-bin coordinates                  REAL
  demand level  <- REAL per-bin 'Liters' volume from bin dataset          REAL
  demand spread <- Austin per-route CV                                    REAL-CALIBRATED
  travel time   <- distance x real NYC congestion-by-hour regime          REAL-CALIBRATED
                   (empirical intra-day speed pattern transferred)
  eco-speed feas<- data-driven levels vs achievable speed per arc/hour    REAL-CALIBRATED

Output: instances/dublin_real/{instance.json, scenarios.npz, meta.json}
"""
import os, json
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw"); PROC = os.path.join(HERE, "processed")
OUT = os.path.join(HERE, "instances", "dublin_real"); os.makedirs(OUT, exist_ok=True)

N_NODES = 100; N_SCEN = 60
SHIFT_START, SHIFT_END = 8.0, 16.0
VEH_CAP_M3 = 12.0; SERVICE_MIN = 6.0
DUBLIN_FREEFLOW_KMH = 42.0     # urban free-flow base (off-peak)

bins = pd.read_csv(os.path.join(RAW, "dublin_solar_bins.csv")).dropna(subset=["Latitude", "Longitude", "Liters"])
od = pd.read_parquet(os.path.join(PROC, "nyc_od_traveltime.parquet"))
austin = pd.read_csv(os.path.join(PROC, "austin_route_demand.csv"))
austin_ok = austin[(austin["mean_kg"] > 0) & austin["cv"].between(0.05, 1.0)].reset_index(drop=True)

# real intra-day congestion multiplier (NYC empirical achievable-speed-by-hour, normalised)
hourly_mph = od.groupby("hour")["mean_mph"].median()
cong = (hourly_mph / hourly_mph.max()).to_dict()    # 1.0 = free-flow, <1 = congested

# pick a coherent dense service area: 40 bins nearest the medoid
lat0, lon0 = bins["Latitude"].median(), bins["Longitude"].median()
bins["d0"] = np.hypot(bins["Latitude"] - lat0, bins["Longitude"] - lon0)
sel = bins.sort_values("d0").head(N_NODES + 1).reset_index(drop=True)
lat = sel["Latitude"].values; lon = sel["Longitude"].values
liters = sel["Liters"].values.astype(float)
Np = len(sel)

def haversine(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dphi = np.radians(la2-la1); dlmb = np.radians(lo2-lo1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
D = np.zeros((Np, Np))
for i in range(Np):
    D[i] = haversine(lat[i], lon[i], lat, lon)
np.fill_diagonal(D, 0.0)

# data-driven eco-speed levels from achievable-speed regime over the shift hours
ach = np.array([DUBLIN_FREEFLOW_KMH * cong.get(h, 1.0) for h in range(int(SHIFT_START), int(SHIFT_END)+1)])
# add stochastic spread for percentile estimation
ach_s = np.clip(rng.lognormal(np.log(ach.mean())-0.05, 0.25, 5000), 5, DUBLIN_FREEFLOW_KMH+5)
SPEED = {"low": round(float(np.percentile(ach_s, 30)), 1),
         "med": round(float(np.percentile(ach_s, 60)), 1),
         "high": round(float(np.percentile(ach_s, 85)), 1)}
sp_kmh = np.array([SPEED[s] for s in ["low", "med", "high"]])

# demand: REAL per-bin Liters -> base m3 (normalised to realistic per-stop scale), REAL-CALIBRATED CV
base_dem = (liters[1:] / liters.mean()) * 1.2     # mean ~1.2 m3/stop, real spatial heterogeneity
pick = rng.choice(len(austin_ok), size=Np-1, replace=False)
node_cv = austin_ok.loc[pick, "cv"].values

q = np.zeros((N_SCEN, Np)); tt = np.zeros((N_SCEN, Np, Np, 3)); feas = np.zeros((N_SCEN, Np, Np, 3), dtype=np.int8)
for s in range(N_SCEN):
    for n in range(1, Np):
        cv = node_cv[n-1]; mu = np.log(base_dem[n-1]) - 0.5*np.log(1+cv**2)
        q[s, n] = rng.lognormal(mu, np.sqrt(np.log(1+cv**2)))
    hour = int(np.clip(round(rng.uniform(SHIFT_START, SHIFT_END)), 0, 23))
    base_v = DUBLIN_FREEFLOW_KMH * cong.get(hour, 1.0)
    for i in range(Np):
        for j in range(Np):
            if i == j: continue
            v_kmh = float(np.clip(rng.lognormal(np.log(base_v)-0.03, 0.22), 5, DUBLIN_FREEFLOW_KMH+5))
            for si, vlev in enumerate(sp_kmh):
                v_eff = min(vlev, v_kmh)
                tt[s, i, j, si] = D[i, j] / max(v_eff, 1e-3) * 60.0
                feas[s, i, j, si] = int(v_kmh >= vlev - 3)

np.savez_compressed(os.path.join(OUT, "scenarios.npz"), q=q, tt=tt, feas=feas, sp_kmh=sp_kmh, dist=D)
inst = {"name": "dublin_real", "n_nodes": Np, "depot": 0,
        "lat": lat.tolist(), "lon": lon.tolist(), "distance_km": D.round(4).tolist(),
        "vehicle_capacity_m3": VEH_CAP_M3, "service_time_min": SERVICE_MIN,
        "shift": [SHIFT_START, SHIFT_END], "speed_levels_kmh": SPEED,
        "bin_liters_real": [0.0]+liters[1:].tolist(),
        "base_demand_m3": [0.0]+base_dem.round(3).tolist(),
        "demand_cv": [0.0]+node_cv.round(3).tolist(),
        "time_window": [[SHIFT_START, SHIFT_END]]*Np}
json.dump(inst, open(os.path.join(OUT, "instance.json"), "w"), indent=2)
meta = {"name": "dublin_real", "n_nodes": Np, "n_scenarios": N_SCEN,
        "geometry": "REAL Fingal solar-bin coordinates (190 bins, dense subset)",
        "demand": "REAL per-bin Liters level x REAL-CALIBRATED CV (Austin)",
        "travel_time": "REAL-CALIBRATED distance x NYC empirical congestion-by-hour regime",
        "speed_levels_kmh": SPEED, "low_feasible_frac": float(feas[..., 0].mean()),
        "high_feasible_frac": float(feas[..., 2].mean()),
        "demand_mean_m3": float(q[:, 1:].mean())}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
print(json.dumps(meta, indent=2))
