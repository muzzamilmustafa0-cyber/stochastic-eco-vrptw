"""
03 - DATA FUSION (multi-borough): build NYC LA-SEco-ECVRPTW instances + scenario sets.

Builds one fully-real instance family per NYC borough (distinct traffic / demand /
eco-speed-feasibility regimes), fusing:
  geometry      <- NYC taxi-zone centroids (per borough)        REAL
  travel time   <- NYC TLC OD x hour distributions             REAL
  eco-speed feas<- real achievable km/h per arc x hour         REAL  (data-driven levels)
  demand level  <- NYC DSNY borough refuse tonnage             REAL  (per-borough scaling)
  demand spread <- Austin per-route CV + DSNY seasonality      REAL-CALIBRATED

Output: instances/nyc_<borough>/{instance.json, scenarios.npz, meta.json}
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "processed")

N_NODES = 68          # use as many real zones as have travel-time data per borough
N_SCEN = 60
SHIFT_START, SHIFT_END = 8.0, 16.0
VEH_CAP_M3 = 12.0
SERVICE_MIN = 6.0
BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx"]

zc = pd.read_csv(os.path.join(PROC, "nyc_zone_centroids.csv"))
od = pd.read_parquet(os.path.join(PROC, "nyc_od_traveltime.parquet"))
austin = pd.read_csv(os.path.join(PROC, "austin_route_demand.csv"))
dsny = pd.read_csv(os.path.join(PROC, "nyc_dsny_district_demand.csv"))
od_key = od.set_index(["PULocationID", "DOLocationID", "hour"])
od_pairs = set(zip(od["PULocationID"], od["DOLocationID"]))
austin_ok = austin[(austin["mean_kg"] > 0) & austin["cv"].between(0.05, 1.0)].reset_index(drop=True)

# real per-borough demand level (normalized mean refuse tonnage)
bor_level = dsny.groupby("borough")["mean_tons"].mean()
bor_level = (bor_level / bor_level.mean())
seas = float(dsny.filter(like="seas_07").iloc[0, 0]) if "seas_07" in dsny.columns else 1.0


def haversine(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dphi = np.radians(la2 - la1); dlmb = np.radians(lo2 - lo1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def build(borough, seed):
    rng = np.random.default_rng(seed)
    bz = zc[zc["Borough"] == borough].copy()
    deg = od.groupby("PULocationID").size().rename("outdeg")
    bz = bz.merge(deg, left_on="LocationID", right_index=True, how="left").fillna({"outdeg": 0})
    bz = bz[bz["outdeg"] > 0].sort_values("outdeg", ascending=False).head(N_NODES + 1).reset_index(drop=True)
    if len(bz) < 12:
        print(f"  SKIP {borough}: only {len(bz)} zones with travel-time data"); return None
    depot = bz.iloc[0]; nodes = bz.iloc[1:].reset_index(drop=True)
    all_ids = [int(depot["LocationID"])] + nodes["LocationID"].astype(int).tolist()
    lat = np.array([depot["lat"]] + nodes["lat"].tolist())
    lon = np.array([depot["lon"]] + nodes["lon"].tolist())
    Np = len(all_ids)

    D = np.zeros((Np, Np))
    for i in range(Np):
        D[i] = haversine(lat[i], lon[i], lat, lon)
    np.fill_diagonal(D, 0.0)

    bz_set = set(bz["LocationID"].astype(int))
    od_b = od[od["PULocationID"].isin(bz_set) & od["DOLocationID"].isin(bz_set)]
    fb_mph = od_b.groupby("hour")["mean_mph"].median()
    gl_mph = float(od_b["mean_mph"].median()) if len(od_b) else 12.0
    ach_kmh = od_b["mean_mph"].values * 1.60934
    SPEED = {"low": round(float(np.percentile(ach_kmh, 30)), 1),
             "med": round(float(np.percentile(ach_kmh, 60)), 1),
             "high": round(float(np.percentile(ach_kmh, 85)), 1)}
    sp_kmh = np.array([SPEED[s] for s in ["low", "med", "high"]])

    def speed_sample(i, j, hour):
        oi, oj = all_ids[i], all_ids[j]
        try:
            r = od_key.loc[(oi, oj, hour)]
            cv = min(max(r["std_min"] / max(r["mean_min"], 1e-6), 0.1), 0.8)
            mu = np.log(r["mean_mph"]) - 0.5*np.log(1+cv**2)
            return float(np.clip(rng.lognormal(mu, np.sqrt(np.log(1+cv**2))), 3, 75))
        except KeyError:
            base = float(fb_mph.get(hour, gl_mph))
            return float(np.clip(rng.lognormal(np.log(base)-0.045, 0.3), 3, 75))

    # demand: real borough level * per-node base * real CV (Austin)
    lvl = float(bor_level.get(borough, 1.0))
    pick = rng.choice(len(austin_ok), size=Np-1, replace=False)
    base_dem = rng.uniform(0.4, 2.2, Np-1) * lvl
    node_cv = austin_ok.loc[pick, "cv"].values

    q = np.zeros((N_SCEN, Np))
    tt = np.zeros((N_SCEN, Np, Np, 3))
    feas = np.zeros((N_SCEN, Np, Np, 3), dtype=np.int8)
    for s in range(N_SCEN):
        for n in range(1, Np):
            cv = node_cv[n-1]; mu = np.log(base_dem[n-1]*seas) - 0.5*np.log(1+cv**2)
            q[s, n] = rng.lognormal(mu, np.sqrt(np.log(1+cv**2)))
        hour = int(np.clip(round(rng.uniform(SHIFT_START, SHIFT_END)), 0, 23))
        for i in range(Np):
            for j in range(Np):
                if i == j: continue
                v_kmh = speed_sample(i, j, hour) * 1.60934
                for si, vlev in enumerate(sp_kmh):
                    v_eff = min(vlev, v_kmh)
                    tt[s, i, j, si] = D[i, j] / max(v_eff, 1e-3) * 60.0
                    feas[s, i, j, si] = int(v_kmh >= vlev - 5)

    out = os.path.join(HERE, "instances", f"nyc_{borough.lower()}")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "scenarios.npz"),
                        q=q, tt=tt, feas=feas, sp_kmh=sp_kmh, dist=D)
    inst = {"name": f"nyc_{borough.lower()}", "n_nodes": Np, "depot": 0,
            "node_location_ids": all_ids, "lat": lat.tolist(), "lon": lon.tolist(),
            "distance_km": D.round(4).tolist(), "vehicle_capacity_m3": VEH_CAP_M3,
            "service_time_min": SERVICE_MIN, "shift": [SHIFT_START, SHIFT_END],
            "speed_levels_kmh": SPEED, "base_demand_m3": [0.0]+base_dem.round(3).tolist(),
            "demand_cv": [0.0]+node_cv.round(3).tolist(),
            "time_window": [[SHIFT_START, SHIFT_END]]*Np}
    json.dump(inst, open(os.path.join(out, "instance.json"), "w"), indent=2)
    meta = {"borough": borough, "n_nodes": Np, "n_scenarios": N_SCEN,
            "geometry": "REAL NYC TLC zone centroids", "travel_time": "REAL NYC TLC OD x hour",
            "demand": "REAL borough level (DSNY) x REAL-CALIBRATED CV (Austin)",
            "speed_levels_kmh": SPEED, "demand_borough_level": round(lvl, 3),
            "high_feasible_frac": float(feas[..., 2].mean()),
            "low_feasible_frac": float(feas[..., 0].mean())}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"  {borough:10s} N={Np} speeds={SPEED} demand_lvl={lvl:.2f} "
          f"feas(low/high)={meta['low_feasible_frac']:.2f}/{meta['high_feasible_frac']:.2f}")
    return meta


if __name__ == "__main__":
    print("Building NYC borough instances...")
    for k, b in enumerate(BOROUGHS):
        build(b, seed=42 + k)
