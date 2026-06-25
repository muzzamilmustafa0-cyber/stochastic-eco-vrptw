"""
02 - Real waste-demand variability from NYC DSNY (district-month tonnage) and
Austin (per-route daily load weights).

Output:
  processed/nyc_dsny_district_demand.csv : borough, cd, mean_tons, std_tons, cv,
                                           p10,p50,p90, plus month seasonal factors
  processed/austin_route_demand.csv      : route, n_days, mean_kg, std_kg, cv, dow factors
  processed/demand_variability_summary.csv : pooled CV stats used to calibrate nodes

These give the REAL coefficient-of-variation / quantile structure of waste demand,
which the fusion step maps onto instance nodes (REAL-CALIBRATED demand scenarios).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw")
PROC = os.path.join(HERE, "processed")

# ---------- NYC DSNY ----------
d = pd.read_csv(os.path.join(RAW, "nyc_dsny_tonnage.csv"))
d = d.dropna(subset=["refusetonscollected"])
d["year"] = d["month"].str.split("/").str[0].str.strip().astype(int)
d["mon"] = d["month"].str.split("/").str[1].str.strip().astype(int)
d["tons"] = pd.to_numeric(d["refusetonscollected"], errors="coerce")
d = d[(d["tons"] > 0) & d["tons"].notna()]
# keep recent decade for stationarity
d = d[d["year"] >= 2015]

key = ["borough", "communitydistrict"]
agg = d.groupby(key)["tons"].agg(
    n="count", mean_tons="mean", std_tons="std",
    p10=lambda s: s.quantile(.10), p50="median", p90=lambda s: s.quantile(.90),
).reset_index()
agg["cv"] = agg["std_tons"] / agg["mean_tons"]
agg = agg[agg["n"] >= 24]
# seasonal (month-of-year) factor pooled across districts
seas = d.groupby("mon")["tons"].mean()
seas = (seas / seas.mean()).round(4)
for m in range(1, 13):
    agg[f"seas_{m:02d}"] = seas.get(m, 1.0)
agg.to_csv(os.path.join(PROC, "nyc_dsny_district_demand.csv"), index=False)
print("DSNY districts:", len(agg), "| median CV:", round(agg["cv"].median(), 3))

# ---------- Austin ----------
a = pd.read_csv(os.path.join(RAW, "austin_waste_loads.csv"))
a = a.dropna(subset=["load_weight"])
a["load_weight"] = pd.to_numeric(a["load_weight"], errors="coerce")
a = a[a["load_weight"] > 0]
# focus on residential garbage / recycling / organics collection routes
waste_types = a["load_type"].str.upper().fillna("")
mask = waste_types.str.contains("GARBAGE|RECYCL|ORGANIC|COMPOST|TRASH")
a = a[mask].copy()
a["date"] = pd.to_datetime(a["report_date"], errors="coerce").dt.date
a["dow"] = pd.to_datetime(a["report_date"], errors="coerce").dt.dayofweek
# daily load per route (kg) -- sum loads same route same day
daily = a.groupby(["route_number", "date", "dow"])["load_weight"].sum().reset_index()
rt = daily.groupby("route_number")["load_weight"].agg(
    n_days="count", mean_kg="mean", std_kg="std",
    p10=lambda s: s.quantile(.10), p50="median", p90=lambda s: s.quantile(.90),
).reset_index()
rt["cv"] = rt["std_kg"] / rt["mean_kg"]
rt = rt[rt["n_days"] >= 20]
# day-of-week factor pooled
dowf = daily.groupby("dow")["load_weight"].mean()
dowf = (dowf / dowf.mean()).round(4)
for k in range(7):
    rt[f"dow_{k}"] = dowf.get(k, 1.0)
rt.to_csv(os.path.join(PROC, "austin_route_demand.csv"), index=False)
print("Austin routes:", len(rt), "| median CV:", round(rt["cv"].median(), 3))

# ---------- pooled variability summary (for calibration) ----------
summary = pd.DataFrame({
    "source": ["NYC_DSNY_district", "Austin_route"],
    "n_units": [len(agg), len(rt)],
    "cv_p25": [agg["cv"].quantile(.25), rt["cv"].quantile(.25)],
    "cv_median": [agg["cv"].median(), rt["cv"].median()],
    "cv_p75": [agg["cv"].quantile(.75), rt["cv"].quantile(.75)],
})
summary.to_csv(os.path.join(PROC, "demand_variability_summary.csv"), index=False)
print(summary.to_string(index=False))
