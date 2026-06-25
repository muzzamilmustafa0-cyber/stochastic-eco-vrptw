"""
01 - Real travel-time distributions from NYC TLC yellow-taxi trips.

Output:
  processed/nyc_zone_centroids.csv      : LocationID, Borough, Zone, lat, lon
  processed/nyc_od_traveltime.parquet   : PU, DO, hour, n, mean_min, std_min,
                                          p10,p50,p90, mean_mph  (OD x hour cells)
Travel-time distribution per (origin zone, destination zone, hour-of-day) is the
REAL stochastic travel-time source for the LA-SEco-ECVRPTW scenario generator.
"""
import glob, os
import numpy as np
import pandas as pd
import geopandas as gpd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw")
PROC = os.path.join(HERE, "processed")
os.makedirs(PROC, exist_ok=True)

# ---- 1. zone centroids from shapefile (real geometry) ----
import zipfile
zpath = os.path.join(RAW, "nyc_taxi_zones.zip")
with zipfile.ZipFile(zpath) as z:
    z.extractall(os.path.join(RAW, "taxi_zones_shp"))
shp = gpd.read_file(os.path.join(RAW, "taxi_zones_shp", "taxi_zones", "taxi_zones.shp"))
shp = shp.to_crs(4326)
cent = shp.geometry.representative_point()
lookup = pd.read_csv(os.path.join(RAW, "nyc_taxi_zone_lookup.csv"))
zc = pd.DataFrame({
    "LocationID": shp["LocationID"].astype(int),
    "lat": cent.y.values, "lon": cent.x.values,
}).merge(lookup, on="LocationID", how="left")
zc.to_csv(os.path.join(PROC, "nyc_zone_centroids.csv"), index=False)
print("zone centroids:", len(zc))

# ---- 2. trips -> travel-time cells (yellow + green taxi) ----
frames = []
files = sorted(glob.glob(os.path.join(RAW, "nyc_yellow_*.parquet")) +
               glob.glob(os.path.join(RAW, "nyc_green_*.parquet")))
for f in files:
    pre = "tpep" if "yellow" in os.path.basename(f) else "lpep"
    cols = [f"{pre}_pickup_datetime", f"{pre}_dropoff_datetime",
            "PULocationID", "DOLocationID", "trip_distance"]
    df = pd.read_parquet(f, columns=cols).rename(columns={
        f"{pre}_pickup_datetime": "pu", f"{pre}_dropoff_datetime": "do"})
    df["dur_min"] = (df["do"] - df["pu"]).dt.total_seconds() / 60.0
    df["hour"] = df["pu"].dt.hour
    # quality filters: realistic urban trips only
    df = df[(df["dur_min"].between(1, 120)) &
            (df["trip_distance"].between(0.2, 30)) &
            (df["PULocationID"].between(1, 263)) &
            (df["DOLocationID"].between(1, 263)) &
            (df["PULocationID"] != df["DOLocationID"])]
    df["mph"] = df["trip_distance"] / (df["dur_min"] / 60.0)
    df = df[df["mph"].between(1, 70)]          # drop GPS/clock errors
    frames.append(df[["PULocationID", "DOLocationID", "hour", "dur_min", "mph"]])
    print("loaded", os.path.basename(f), len(df))

trips = pd.concat(frames, ignore_index=True)
print("total clean trips:", len(trips))

g = trips.groupby(["PULocationID", "DOLocationID", "hour"])
od = g["dur_min"].agg(
    n="count", mean_min="mean", std_min="std",
    p10=lambda s: s.quantile(.10), p50="median", p90=lambda s: s.quantile(.90),
).reset_index()
od["mean_mph"] = g["mph"].mean().values
od = od[od["n"] >= 20]                          # reliable cells only
od["std_min"] = od["std_min"].fillna(0.0)
od.to_parquet(os.path.join(PROC, "nyc_od_traveltime.parquet"), index=False)
print("OD x hour reliable cells:", len(od),
      "| unique OD pairs:", od.groupby(["PULocationID", "DOLocationID"]).ngroups)
