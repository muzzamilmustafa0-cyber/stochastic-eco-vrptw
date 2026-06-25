"""99 - Scan all built instances and emit a consolidated dataset index (CSV + Markdown)."""
import os, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INST = os.path.join(HERE, "instances")

rows = []
for d in sorted(glob.glob(os.path.join(INST, "*"))):
    ij = os.path.join(d, "instance.json"); mj = os.path.join(d, "meta.json"); sz = os.path.join(d, "scenarios.npz")
    if not os.path.exists(ij):
        continue
    inst = json.load(open(ij)); meta = json.load(open(mj)) if os.path.exists(mj) else {}
    z = np.load(sz)
    q = z["q"]; feas = z["feas"]
    rows.append({
        "instance": inst["name"], "nodes": inst["n_nodes"],
        "scenarios": q.shape[0],
        "speeds_kmh": "/".join(str(v) for v in inst["speed_levels_kmh"].values()),
        "demand_mean_m3": round(float(q[:, 1:].mean()), 2),
        "demand_cv": round(float(q[:, 1:].std()/q[:, 1:].mean()), 2),
        "feas_low": round(float(feas[..., 0].mean()), 2),
        "feas_high": round(float(feas[..., 2].mean()), 2),
        "geometry": meta.get("geometry", ""),
        "travel_time": meta.get("travel_time", ""),
        "demand_src": meta.get("demand", ""),
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "DATASET_INDEX.csv"), index=False)

with open(os.path.join(HERE, "DATASET_INDEX.md"), "w", encoding="utf-8") as f:
    f.write("# Dataset Index\n\n")
    f.write(f"Built {len(df)} stochastic instances. Each has instance.json + scenarios.npz "
            "(q[S,N], tt[S,N,N,3], feas[S,N,N,3], dist[N,N]) + meta.json.\n\n")
    f.write(df[["instance", "nodes", "scenarios", "speeds_kmh", "demand_mean_m3",
                "demand_cv", "feas_low", "feas_high"]].to_markdown(index=False))
    f.write("\n\n## Provenance\n\n")
    for _, r in df.iterrows():
        f.write(f"- **{r['instance']}**: geom={r['geometry']}; tt={r['travel_time']}; demand={r['demand_src']}\n")

print(df.to_string(index=False))
print("\nWrote DATASET_INDEX.csv and DATASET_INDEX.md")
