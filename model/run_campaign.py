"""
Full experimental campaign: every method x instance x seed, evaluated out-of-sample.
Resumable: appends one row per (instance, method, seed) to results/results.csv so a
long run can be monitored / restarted without recomputation.

Methods:
  D-HGLS, PTO-HGLS, Q-HGLS, RO-Eco, S-HGLS, LA-SHGLS   (experiments.run_method)
  DF-det   learned planning + deterministic optimisation
  DF-CC    learned capacity constraint + scenario HGLS
  DF-CC*   planning-regime selection (run_proposed)
"""
import os, sys, csv, time
import numpy as np
from . import ecvrptw as E, experiments as X, decision_focus as DF

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(HERE, "results"); os.makedirs(RESDIR, exist_ok=True)
CSV = os.path.join(RESDIR, "results.csv")

METHODS = ["D-HGLS", "PTO-HGLS", "Q-HGLS", "RO-Eco", "S-HGLS", "LA-SHGLS",
           "DF-det", "DF-CC", "DF-CC*"]
FIELDS = ["instance", "method", "seed", "n_vehicles", "E_cost", "E_fuel", "E_emission",
          "E_recourse", "CVaR_cost", "worst_cost", "P_late", "E_late_min",
          "P_overload", "E_overload_m3", "missed_nodes", "infeas_speed_arcs",
          "tau_mean", "tau_std", "secs"]


def _done():
    done = set()
    if os.path.exists(CSV):
        import pandas as pd
        d = pd.read_csv(CSV)
        for _, r in d.iterrows():
            done.add((r["instance"], r["method"], int(r["seed"])))
    return done


def run(instances, seeds=(1, 2, 3), budget=1200, df_train_budget=400, df_iters=45):
    done = _done()
    new = not os.path.exists(CSV)
    f = open(CSV, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for name in instances:
        inst, sc = E.load(name)
        for seed in seeds:
            sc_tr, sc_te = X.split_scenarios(sc, seed=seed)
            for mth in METHODS:
                if (name, mth, seed) in done:
                    continue
                t0 = time.time()
                if mth == "DF-det":
                    m, _ = DF.run_df(inst, sc_tr, sc_te, budget=budget,
                                     train_budget=df_train_budget, iters=df_iters,
                                     seed=seed, mode="det", name="DF-det")
                elif mth == "DF-CC":
                    m, _ = DF.run_df(inst, sc_tr, sc_te, budget=budget,
                                     train_budget=df_train_budget, iters=df_iters,
                                     seed=seed, mode="cc", name="DF-CC")
                elif mth == "DF-CC*":
                    m, _ = DF.run_proposed(inst, sc_tr, sc_te, budget=budget,
                                           train_budget=df_train_budget, iters=df_iters,
                                           seed=seed, name="DF-CC*")
                else:
                    m, _ = X.run_method(mth, inst, sc_tr, sc_te, budget=budget, seed=seed)
                row = {k: m.get(k, "") for k in FIELDS}
                row.update(instance=name, method=mth, seed=seed, secs=round(time.time()-t0, 1))
                w.writerow(row); f.flush()
                print(f"{name:16s} {mth:9s} s{seed} cost={m['E_cost']:.2f} "
                      f"CVaR={m['CVaR_cost']:.2f} ({row['secs']}s)", flush=True)
    f.close()


if __name__ == "__main__":
    allinst = ["nyc_manhattan", "nyc_queens", "nyc_brooklyn", "dublin_real",
               "peshawar_real", "solomon_c101", "solomon_c201", "solomon_r102",
               "solomon_r202", "solomon_rc101", "solomon_rc201"]
    args = sys.argv[1:]
    insts = args if args else allinst
    run(insts)
