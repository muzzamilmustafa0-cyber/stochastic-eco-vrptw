"""
Recourse-weight sensitivity: re-run the proposed DF-CC* and the strong baselines
(S-HGLS, LA-SHGLS) under perturbed recourse cost weights, in BOTH optimisation and
evaluation, to show the win is robust to the (necessarily chosen) weights and not an
artefact of one setting. Resumable per-row CSV.

Sweeps the overload weight C_OVER and lateness weight C_LATE around their base values.
"""
import os, csv, time
import numpy as np
from . import ecvrptw as E, experiments as X, decision_focus as DF

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results"); OUT = os.path.join(RES, "sensitivity.csv")
INSTANCES = ["nyc_manhattan", "peshawar_real", "solomon_c101"]
# (C_OVER, C_LATE) settings; base is (2.0, 0.05)
SETTINGS = [(1.0, 0.05), (2.0, 0.05), (4.0, 0.05), (2.0, 0.025), (2.0, 0.10)]
FIELDS = ["instance", "c_over", "c_late", "seed", "method", "E_cost", "CVaR_cost",
          "P_overload", "n_vehicles", "secs"]


def _done():
    s = set()
    if os.path.exists(OUT):
        import pandas as pd
        for _, r in pd.read_csv(OUT).iterrows():
            s.add((r["instance"], float(r["c_over"]), float(r["c_late"]), int(r["seed"]), r["method"]))
    return s


def run(instances=INSTANCES, seeds=(1, 2), budget=1200):
    done = _done()
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS)
    if new: w.writeheader()
    for name in instances:
        inst, sc = E.load(name)
        for (co, cl) in SETTINGS:
            E.C_OVER, E.C_LATE = co, cl          # perturb module weights (read at runtime)
            for seed in seeds:
                sc_tr, sc_te = X.split_scenarios(sc, seed=seed)
                for mth in ["S-HGLS", "LA-SHGLS", "DF-CC*"]:
                    if (name, co, cl, seed, mth) in done:
                        continue
                    t0 = time.time()
                    if mth == "DF-CC*":
                        m, _ = DF.run_proposed(inst, sc_tr, sc_te, budget=budget,
                                               train_budget=350, iters=40, seed=seed)
                    else:
                        m, _ = X.run_method(mth, inst, sc_tr, sc_te, budget=budget, seed=seed)
                    row = {k: m.get(k, "") for k in FIELDS}
                    row.update(instance=name, c_over=co, c_late=cl, seed=seed, method=mth,
                               secs=round(time.time()-t0, 1))
                    w.writerow(row); f.flush()
                    print(f"{name:14s} co={co} cl={cl} s{seed} {mth:8s} cost={m['E_cost']:.1f} ({row['secs']}s)", flush=True)
    E.C_OVER, E.C_LATE = 2.0, 0.05               # restore base
    f.close()


if __name__ == "__main__":
    run()
