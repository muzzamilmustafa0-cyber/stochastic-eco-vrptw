"""
06 - Peshawar WSSP Zone D case-study loader (PENDING real data).

The manuscript references 109 fixed containers (Zone D) but only 3 sample nodes are
embedded in the .docx appendix. To keep the case study REAL (no fabrication), this
loader expects the full WSSP file from the collaborator (Salman) at:
    raw/peshawar_wssp_zoneD.csv
with columns: id, lat, lon, demand_m3, tw_start, tw_end, service_min

When that file is provided, run this script to build instances/peshawar_real/ using the
SAME stochastic eco layers as the other families (real-calibrated demand CV from Austin,
congestion regime, data-driven eco-speed levels). Until then it is a no-op with a notice.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "raw", "peshawar_wssp_zoneD.csv")

if not os.path.exists(SRC):
    print("PENDING: peshawar_wssp_zoneD.csv not found.")
    print("  -> Obtain the full 109-node Zone D register (coords, demand, time windows,")
    print("     service times) from WSSP / collaborator, save to raw/peshawar_wssp_zoneD.csv,")
    print("     then re-run. Do NOT fabricate the case-study data.")
else:
    df = pd.read_csv(SRC)
    print(f"Found WSSP file with {len(df)} nodes. (build logic mirrors 04/05.)")
    # build logic to be wired identically to dublin/solomon fusion once data arrives
