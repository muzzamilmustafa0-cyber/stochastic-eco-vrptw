"""
09 - Precompute per-family arc-speed lookup tables so scenario replicates can be
generated on the fly from seeds (reproducible, no bulk tensor storage).

NYC families: [N, N, 24] mean achievable mph and coefficient of variation from the
real OD x hour cells, with hour-median fallback where a cell is missing.
Profile families (dublin, solomon, peshawar): the global congestion-by-hour profile.

Output: processed/arclut_<family>.npz and processed/congestion_profile.json
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "processed")
INST = os.path.join(HERE, "instances")

od = pd.read_parquet(os.path.join(PROC, "nyc_od_traveltime.parquet"))
od["cv"] = np.clip(od["std_min"] / np.maximum(od["mean_min"], 1e-6), 0.1, 0.8)

# global congestion-by-hour profile (median mph normalised to its max)
hourly = od.groupby("hour")["mean_mph"].median()
cong = (hourly / hourly.max())
json.dump({str(h): round(float(cong.get(h, 1.0)), 4) for h in range(24)},
          open(os.path.join(PROC, "congestion_profile.json"), "w"), indent=2)

for fam in ["nyc_manhattan", "nyc_queens", "nyc_brooklyn"]:
    inst = json.load(open(os.path.join(INST, fam, "instance.json")))
    ids = inst["node_location_ids"]; N = len(ids)
    mean_mph = np.zeros((N, N, 24), np.float32)
    cv = np.zeros((N, N, 24), np.float32)
    sub = od[od["PULocationID"].isin(ids) & od["DOLocationID"].isin(ids)]
    fb_mph = sub.groupby("hour")["mean_mph"].median()
    fb_all = float(sub["mean_mph"].median()) if len(sub) else 12.0
    fb_cv = float(sub["cv"].median()) if len(sub) else 0.3
    for h in range(24):
        mean_mph[:, :, h] = float(fb_mph.get(h, fb_all))
        cv[:, :, h] = fb_cv
    idx = {loc: k for k, loc in enumerate(ids)}
    cells = od[od["PULocationID"].isin(ids) & od["DOLocationID"].isin(ids)]
    n_real = 0
    for row in cells.itertuples():
        i, j, h = idx[row.PULocationID], idx[row.DOLocationID], int(row.hour)
        mean_mph[i, j, h] = row.mean_mph; cv[i, j, h] = row.cv; n_real += 1
    np.savez_compressed(os.path.join(PROC, f"arclut_{fam}.npz"),
                        mean_mph=mean_mph, cv=cv)
    print(f"{fam}: N={N}, real cells={n_real}, lut size ~{mean_mph.nbytes*2/1e6:.1f} MB")
print("congestion profile hours 7-18:",
      [round(float(cong.get(h, 1)), 2) for h in range(7, 19)])
