"""Eco-speed contribution ablation (pre-submission suggestion B1).

Five speed policies under an otherwise identical SAA pipeline (same OR-Tools
construction on the 0.75 planning quantile, same 15 s refinement budget, same
locked-test scoring):

  fixed_low / fixed_med / fixed_high : all arcs planned at one level; the
      refinement search uses structural moves only
  opt_nofeas : speed levels optimized, but the search objective ignores the
      speed-infeasibility penalty (deterministic feasibility view)
  opt_full   : the standard regime, speed levels optimized with the
      feasibility-aware objective

Reported per policy: expected cost, fuel, CO2e, lateness, deferment, the share
of planned arcs whose level exceeds the realized achievable speed, and the
planned speed mix. Resumable CSV: results2/eco_ablation.csv.
"""
import os, csv, time
import numpy as np
from . import ecvrptw as E, scenlib as SL, policy as P, hgls as H

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results2")
OUT = os.path.join(RES, "eco_ablation.csv")
FAMS = ["nyc_manhattan", "peshawar_real", "solomon_c101", "solomon_r202"]
# All five operational replicates per family. With only two replicates the
# eight paired observations cap the Holm-corrected Wilcoxon p at 0.031, which
# is significant only if every instance agrees; twenty instances give the test
# adequate power.
REPS = [1, 2, 3, 4, 5]
# Wall-clock budgets make a single run non-reproducible: at a fixed seed the
# realized cost varies by up to ~12% on the less stable families, which is the
# same order as the effect under study. Every configuration is therefore run
# over several seeds and compared on seed medians.
SEEDS = [0, 1, 2, 3, 4]
POLICIES = ["fixed_low", "fixed_med", "fixed_high", "opt_nofeas", "opt_full"]
FIELDS = ["family", "rep", "seed", "policy", "E_cost", "E_fuel", "E_emission",
          "E_late_min", "P_late", "P_defer", "P_trigger", "n_veh",
          "arc_shortfall_share", "pct_low", "pct_med", "pct_high", "secs"]
BIG = 10**9


def speed_mix(speeds):
    flat = [s for row in speeds for s in row]
    n = max(len(flat), 1)
    return (100 * flat.count(0) / n, 100 * flat.count(1) / n,
            100 * flat.count(2) / n)


def shortfall_share(inst, sc, routes, speeds):
    """Share of planned arc traversals whose level exceeds achievable speed,
    averaged over scenarios."""
    tot = 0; cnt = 0.0
    for r, route in enumerate(routes):
        if not route:
            continue
        path = [inst.depot] + list(route) + [inst.depot]
        for k in range(1, len(path)):
            i, j = path[k-1], path[k]
            lv = speeds[r][k-1]
            cnt += (1 - sc.feas[:, i, j, lv]).mean()
            tot += 1
    return cnt / max(tot, 1)


def _lock(op, lv):
    """Wrap a structural operator so every arc keeps the locked speed level.

    Necessary because hgls._fix_speeds pads lengthened routes with the medium
    level and op_relocate seeds new routes with [1, 1]; without this wrapper a
    nominally fixed-speed policy silently drifts toward medium.
    """
    def wrapped(routes, speeds, rng):
        r, s = op(routes, speeds, rng)
        return r, [[lv] * len(x) for x in s]
    wrapped.__name__ = op.__name__
    return wrapped


def run_policy(policy, inst, tr, seed=0):
    qp = np.quantile(tr.q, 0.75, axis=0)
    r0, s0 = P.construct(inst, tr, qp, ortools_s=5, seed=seed)
    if policy.startswith("fixed"):
        lv = {"fixed_low": 0, "fixed_med": 1, "fixed_high": 2}[policy]
        s0 = [[lv] * (len(r) + 1) for r in r0]
        bak = H.OPS[:]
        H.OPS = [_lock(op, lv) for op in bak
                 if "speed" not in op.__name__]
        try:
            br, bs, _ = H.hgls(inst, tr, r0, s0, max_iter=BIG, no_improve=BIG,
                               beta=0.0, seed=seed, time_limit=15.0)
        finally:
            H.OPS = bak
        bs = [[lv] * len(x) for x in bs]
        assert all(v == lv for x in bs for v in x), "speed lock failed"
        return br, bs
    ekw = {"w_infeas": 0.0} if policy == "opt_nofeas" else {}
    br, bs, _ = H.hgls(inst, tr, r0, s0, max_iter=BIG, no_improve=BIG,
                       beta=0.0, seed=seed, time_limit=15.0, eval_kwargs=ekw)
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
                for pol in POLICIES:
                    if (fam, rep, sd, pol) in done:
                        continue
                    t0 = time.perf_counter()
                    r, s = run_policy(pol, inst, tr, seed=sd)
                    m = E.evaluate(inst, te, r, s)
                    lo, md, hi = speed_mix(s)
                    row = dict(
                        family=fam, rep=rep, seed=sd, policy=pol,
                        E_cost=round(m["E_cost"], 3),
                        E_fuel=round(m["E_fuel"], 3),
                        E_emission=round(m["E_emission"], 3),
                        E_late_min=round(m["E_late_min"], 3),
                        P_late=round(m["P_late"], 4),
                        P_defer=round(m["P_defer"], 4),
                        P_trigger=round(m["P_trigger"], 4),
                        n_veh=len([x for x in r if x]),
                        arc_shortfall_share=round(
                            shortfall_share(inst, te, r, s), 4),
                        pct_low=round(lo, 1), pct_med=round(md, 1),
                        pct_high=round(hi, 1),
                        secs=round(time.perf_counter() - t0, 1))
                    w.writerow(row); f.flush()
                    print(f"[eco] {fam} r{rep} s{sd} {pol}: "
                          f"cost={m['E_cost']:.1f} "
                          f"mix={lo:.0f}/{md:.0f}/{hi:.0f} "
                          f"({row['secs']}s)", flush=True)
    f.close()
    print("ECO ABLATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
