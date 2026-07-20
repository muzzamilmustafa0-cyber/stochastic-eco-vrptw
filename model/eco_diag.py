"""Diagnostic for the eco-speed ablation.

The ablation shows a fixed low-speed policy beating optimized speed on most
families under an equal wall-clock budget. Two explanations are possible and
they demand different wording in the paper:

  (a) search-budget effect: speed moves consume budget that structural moves
      would use more productively, so the optimized run is also worse on the
      TRAINING objective it actually optimizes;
  (b) overfitting: the optimized run wins on training and loses on the locked
      test scenarios.

This script records the training and test objective of the same plans, plus a
double-budget optimized run to see whether the gap closes with more time.
"""
import os, csv, time
import numpy as np
from . import ecvrptw as E, scenlib as SL, policy as P, hgls as H
from .eco_ablation import run_policy, FAMS, REPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results2", "eco_diag.csv")
FIELDS = ["family", "rep", "seed", "policy", "train_obj", "test_cost", "secs"]
BIG = 10**9
# Same reason as in eco_ablation: wall-clock budgets make a single run
# non-reproducible, so every configuration is repeated over seeds.
SEEDS = [0, 1, 2, 3, 4]


def opt_full_long(inst, tr, seed=0, budget=45.0):
    qp = np.quantile(tr.q, 0.75, axis=0)
    r0, s0 = P.construct(inst, tr, qp, ortools_s=5, seed=seed)
    br, bs, _ = H.hgls(inst, tr, r0, s0, max_iter=BIG, no_improve=BIG,
                       beta=0.0, seed=seed, time_limit=budget)
    return br, bs


def main():
    done = set()
    if os.path.exists(OUT):
        import pandas as pd
        for _, r in pd.read_csv(OUT).iterrows():
            done.add((r["family"], int(r["rep"]), int(r["seed"]), r["policy"]))
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for fam in FAMS:
        for rep in REPS:
            inst, tr, ca, te = SL.load_replicate(fam, rep)
            for sd in SEEDS:
                jobs = [
                    ("fixed_low",
                     lambda s=sd: run_policy("fixed_low", inst, tr, seed=s)),
                    ("opt_full",
                     lambda s=sd: run_policy("opt_full", inst, tr, seed=s)),
                    ("opt_full_3x",
                     lambda s=sd: opt_full_long(inst, tr, seed=s)),
                ]
                for name, fn in jobs:
                    if (fam, rep, sd, name) in done:
                        continue
                    t0 = time.perf_counter()
                    r, s = fn()
                    tro = E.evaluate(inst, tr, r, s)["fitness"]
                    tec = E.evaluate(inst, te, r, s)["E_cost"]
                    w.writerow(dict(family=fam, rep=rep, seed=sd, policy=name,
                                    train_obj=round(float(tro), 3),
                                    test_cost=round(float(tec), 3),
                                    secs=round(time.perf_counter() - t0, 1)))
                    f.flush()
                    print(f"[diag] {fam} r{rep} s{sd} {name}: "
                          f"train={tro:.1f} test={tec:.1f}", flush=True)
    f.close()
    print("ECO DIAG COMPLETE", flush=True)


if __name__ == "__main__":
    main()
