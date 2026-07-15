"""Submission extras: 30-seed repeats on three representative families and
anytime traces for the refinement search. Resumable."""
import os, csv, json, time
import numpy as np
from . import ecvrptw as E, scenlib as SL, experiments2 as X2, policy as P
from . import hgls as H, external as XT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results2")
FAMS = ["nyc_manhattan", "peshawar_real", "solomon_c101"]
MET_KEYS = ["E_cost", "E_fuel", "E_emission", "E_recourse", "CVaR_cost",
            "P_trigger", "E_returns", "E_return_km", "P_late", "E_late_min",
            "P_defer", "n_veh"]


def _done(path, keycols):
    done = set()
    if os.path.exists(path):
        import pandas as pd
        for _, r in pd.read_csv(path).iterrows():
            done.add(tuple(str(r[k]) for k in keycols))
    return done


def repeats30():
    thetas = json.load(open(os.path.join(RES, "policy_theta.json")))
    from .run_final import theta_for
    path = os.path.join(RES, "repeats30.csv")
    fields = ["family", "method", "seed"] + MET_KEYS + ["secs"]
    done = _done(path, ["family", "method", "seed"])
    new = not os.path.exists(path)
    f = open(path, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    for fam in FAMS:
        inst, tr, ca, te = SL.load_replicate(fam, 1)
        for method in ["SAA", "Q90C", "DFR"]:
            for seed in range(30):
                if (fam, method, str(seed)) in done:
                    continue
                th = theta_for(fam, thetas) if method == "DFR" else None
                t0 = time.perf_counter()
                r, s, _ = X2.run_regime(method, inst, tr, ca, seed=seed, theta=th)
                m = X2.score(inst, te, r, s)
                row = {k: round(float(m[k]), 5) for k in MET_KEYS}
                row.update(family=fam, method=method, seed=seed,
                           secs=round(time.perf_counter() - t0, 1))
                w.writerow(row); f.flush()
        print(f"[rep30] {fam} done", flush=True)
    f.close()


def anytime():
    out = os.path.join(RES, "anytime.json")
    data = json.load(open(out)) if os.path.exists(out) else {}
    for fam in FAMS:
        if fam in data:
            continue
        inst, tr, ca, te = SL.load_replicate(fam, 1)
        q75 = np.quantile(tr.q, 0.75, axis=0)
        rec = {}
        # HGLS from greedy construction, 60 s trace
        import model.construct as C
        r0, s0 = C.nearest_feasible(inst, tr, demand=q75, tt=tr.tt.mean(0))
        trace = []
        H.hgls(inst, tr, r0, s0, max_iter=10**9, no_improve=10**9, beta=0.0,
               seed=0, time_limit=60.0, trace=trace)
        rec["hgls_greedy"] = trace
        # HGLS from OR-Tools construction (constructor time added to the clock)
        t0 = time.perf_counter()
        rx, sx = P.construct(inst, tr, q75, ortools_s=5, seed=0)
        off = time.perf_counter() - t0
        trace2 = []
        H.hgls(inst, tr, rx, sx, max_iter=10**9, no_improve=10**9, beta=0.0,
               seed=0, time_limit=55.0, trace=trace2)
        rec["hgls_ort"] = [(t + off, v) for t, v in trace2]
        # external engine at increasing budgets (final value each)
        pts = []
        for b in [5, 10, 20, 40]:
            t0 = time.perf_counter()
            r, s = XT.solve(inst, tr, q_plan=q75, time_limit_s=b, seed=0)
            el = time.perf_counter() - t0
            m = E.evaluate(inst, tr, r, s)
            pts.append((el, m["fitness"]))
        rec["ort_points"] = pts
        data[fam] = rec
        json.dump(data, open(out, "w"))
        print(f"[anytime] {fam} done", flush=True)


if __name__ == "__main__":
    anytime()
    repeats30()
    print("EXTRAS COMPLETE", flush=True)
