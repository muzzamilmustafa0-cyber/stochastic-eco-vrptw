"""
Ablation study: isolate each component's contribution to the proposed method.
Trains the decision-focused chance-constraint (tau) ONCE per instance/seed, then
runs each ablated variant from it. All variants scored by the identical full-truth
out-of-sample evaluator. Resumable (per-row CSV).

  A0  full DF-CC (learned chance-constraint, scenarios, recourse, eco-speed feasibility)
  A1  - eco-speed feasibility   (w_infeas = 0 in search)
  A2  - stochastic demand        (optimise on mean demand)
  A3  - stochastic travel time   (optimise on mean travel time)
  A4  - recourse                 (recourse weights = 0 in search)
  A5  - chance constraint        (no learned cap -> recourse-only S)
"""
import os, csv, time
import numpy as np
from . import ecvrptw as E, construct as C, hgls as H, experiments as X, decision_focus as DF

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results"); OUT = os.path.join(RES, "ablation.csv")
INSTANCES = ["nyc_manhattan", "peshawar_real", "dublin_real", "solomon_c101", "solomon_r202"]
VARIANTS = ["A0", "A1", "A2", "A3", "A4", "A5"]
FIELDS = ["instance", "variant", "seed", "E_cost", "CVaR_cost", "E_fuel",
          "E_recourse", "P_overload", "P_late", "n_vehicles", "secs"]


def _done():
    s = set()
    if os.path.exists(OUT):
        import pandas as pd
        for _, r in pd.read_csv(OUT).iterrows():
            s.add((r["instance"], r["variant"], int(r["seed"])))
    return s


def run(instances=INSTANCES, seeds=(1, 2, 3), budget=1200, train_budget=400, iters=45):
    done = _done()
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS)
    if new: w.writeheader()
    for name in instances:
        inst, sc = E.load(name)
        for seed in seeds:
            if all((name, v, seed) in done for v in VARIANTS):
                continue
            sc_tr, sc_te = X.split_scenarios(sc, seed=seed)
            tt_mean = sc_tr.tt.mean(0)
            theta, Xn = DF.train_df(inst, sc_tr, budget=train_budget, iters=iters, seed=seed, mode="cc")
            qp, _ = DF.plan_demand(theta, Xn, sc_tr)
            for v in VARIANTS:
                if (name, v, seed) in done:
                    continue
                t0 = time.time(); ekw = {}; cap = qp
                if v == "A1": ekw = {"w_infeas": 0.0}
                elif v == "A2": ekw = {"use_mean_demand": True}
                elif v == "A3": ekw = {"mean_tt": tt_mean}
                elif v == "A4": ekw = {"w_late": 0.0, "w_over": 0.0, "w_miss": 0.0}
                elif v == "A5": cap = None
                demand = qp if cap is not None else np.quantile(sc_tr.q, 0.75, axis=0)
                r0, s0 = C.nearest_feasible(inst, sc_tr, demand=demand, tt=tt_mean)
                br, bs, _ = H.hgls(inst, sc_tr, r0, s0, max_iter=budget, no_improve=budget // 3,
                                   beta=0.0, seed=seed, cap_demand=cap, eval_kwargs=ekw)
                m = E.evaluate(inst, sc_te, br, bs, beta=0.5, alpha=0.9)
                row = {k: m.get(k, "") for k in FIELDS}
                row.update(instance=name, variant=v, seed=seed,
                           n_vehicles=len([x for x in br if x]), secs=round(time.time()-t0, 1))
                w.writerow(row); f.flush()
                print(f"{name:14s} {v} s{seed} cost={m['E_cost']:.1f} over={m['P_overload']:.2f} ({row['secs']}s)", flush=True)
    f.close()


if __name__ == "__main__":
    run()
