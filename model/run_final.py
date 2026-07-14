"""
Final experiment campaign (roadmap Parts III-IV). Stages, all resumable:

  1 policy   : train the global service-level policy on two family folds
  2 main     : 11 families x 5 replicates x 8 regimes, equal wall-clock
  3 exact    : optimality gaps on 8-customer sub-instances (proven optima)
  4 shifts   : re-evaluate stored plans under demand/traffic/dependence shifts
  5 scount   : scenario-count convergence (SAA stability)
  6 ablation : additive component ladder on replicates 1-2 of every family
  7 repeats  : 10-seed repeats of SAA and DFR on six representative instances
  8 sens     : recourse-weight sensitivity with re-optimisation

Outputs under results2/.  Plans are stored as JSON so shift studies re-evaluate
without re-solving.  Every row records the stage, so one CSV per stage.
"""
import os, sys, csv, json, time
import numpy as np
from . import ecvrptw as E, scenlib as SL, experiments2 as X2, policy as P, exact as XA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results2")
PLANS = os.path.join(RES, "plans")
os.makedirs(PLANS, exist_ok=True)

FOLD_A_TRAIN = ["nyc_queens", "dublin_real", "solomon_c101", "solomon_r102",
                "solomon_rc101"]
FOLD_B_TRAIN = ["nyc_manhattan", "nyc_brooklyn", "peshawar_real", "solomon_c201",
                "solomon_r202", "solomon_rc201"]
REP_FAMS = ["nyc_manhattan", "nyc_queens", "dublin_real", "peshawar_real",
            "solomon_c101", "solomon_rc101"]

MET_KEYS = ["E_cost", "E_fuel", "E_emission", "E_recourse", "CVaR_cost",
            "P_trigger", "E_returns", "E_return_km", "P_late", "E_late_min",
            "P_defer", "n_veh"]


def _csv(path, fields):
    new = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    return f, w


def _done(path, keycols):
    done = set()
    if os.path.exists(path):
        import pandas as pd
        d = pd.read_csv(path)
        for _, r in d.iterrows():
            done.add(tuple(str(r[k]) for k in keycols))
    return done


def _plan_path(fam, rep, method, seed=0):
    return os.path.join(PLANS, f"{fam}_r{rep}_{method}_s{seed}.json")


def _save_plan(path, routes, speeds):
    json.dump({"routes": routes, "speeds": speeds}, open(path, "w"))


def _load_plan(path):
    d = json.load(open(path))
    return d["routes"], d["speeds"]


# ---------------------------------------------------------------- 1 policy ----
def stage_policy():
    out = os.path.join(RES, "policy_theta.json")
    thetas = json.load(open(out)) if os.path.exists(out) else {}
    for fold, fams in [("A", FOLD_A_TRAIN), ("B", FOLD_B_TRAIN)]:
        if fold in thetas:
            continue
        print(f"[policy] training fold {fold} on {fams}", flush=True)
        train = []
        for fam in fams:
            inst, tr, ca, te = SL.load_replicate(fam, 1)
            train.append((inst, tr, ca))
        theta, hist = P.train_global(train, iters=25, pop=6, budget_iter=300,
                                     seed=0, verbose=True)
        thetas[fold] = {"theta": list(map(float, theta)), "loss_history":
                        [float(h) for h in hist]}
        json.dump(thetas, open(out, "w"), indent=2)
    return thetas


def theta_for(fam, thetas):
    """Policy trained on the fold that did NOT contain fam."""
    fold = "B" if fam in FOLD_A_TRAIN else "A"
    return np.array(thetas[fold]["theta"])


# ---------------------------------------------------------------- 2 main ------
def stage_main(thetas, seed=0):
    path = os.path.join(RES, "main.csv")
    fields = ["family", "rep", "method", "seed", "secs"] + MET_KEYS + \
             ["kappa", "rho", "tau_mean", "selected"]
    done = _done(path, ["family", "rep", "method", "seed"])
    f, w = _csv(path, fields)
    for fam in SL.FAMILIES:
        for rep in SL.REPLICATES:
            inst, tr, ca, te = SL.load_replicate(fam, rep)
            for method in X2.METHODS:
                if (fam, str(rep), method, str(seed)) in done:
                    continue
                t0 = time.perf_counter()
                th = theta_for(fam, thetas) if method == "DFR" else None
                r, s, info = X2.run_regime(method, inst, tr, ca, seed=seed,
                                           theta=th)
                secs = time.perf_counter() - t0
                m = X2.score(inst, te, r, s)
                _save_plan(_plan_path(fam, rep, method, seed), r, s)
                row = {k: round(float(m[k]), 5) for k in MET_KEYS}
                row.update(family=fam, rep=rep, method=method, seed=seed,
                           secs=round(secs, 1),
                           kappa=round(info.get("kappa", np.nan), 4) if info.get("kappa") else "",
                           rho=round(info.get("rho", np.nan), 4) if info.get("rho") else "",
                           tau_mean=round(info.get("tau_mean", np.nan), 3) if info.get("tau_mean") else "",
                           selected=info.get("selected", ""))
                w.writerow(row); f.flush()
                print(f"[main] {fam} r{rep} {method}: cost={m['E_cost']:.1f} "
                      f"trig={m['P_trigger']:.2f} ({secs:.0f}s)", flush=True)
    f.close()


# ---------------------------------------------------------------- 3 exact -----
def stage_exact():
    path = os.path.join(RES, "exact.csv")
    fields = ["family", "n_cust", "S", "exact_cost", "hgls_cost", "gap_pct",
              "explored", "optimal", "exact_secs"]
    done = _done(path, ["family", "n_cust", "S"])
    f, w = _csv(path, fields)
    import model.hgls as HH
    for fam in REP_FAMS:
        if (fam, "8", "10") in done:
            continue
        inst0, tr0, ca0, te0 = SL.load_replicate(fam, 1)
        nodes = list(range(1, 9))
        inst, sc = XA.subinstance(inst0, tr0, nodes)
        sc = E.Scenarios(sc.q[:10], sc.tt[:10], sc.feas[:10])
        t0 = time.perf_counter()
        br, bc, nexp, opt = XA.exact_solve(inst, sc, time_limit=1200)
        ex_s = time.perf_counter() - t0
        # HGLS at the same fixed speed, structural moves only
        import model.construct as C
        r0, s0 = C.nearest_feasible(inst, sc)
        s0 = [[XA.SPEED] * (len(r) + 1) for r in r0]
        bak = HH.OPS[:]
        HH.OPS = [op for op in bak if "speed" not in op.__name__]
        bh, bs, _ = HH.hgls(inst, sc, r0, s0, max_iter=BIG_ITER, no_improve=4000,
                            beta=0.0, seed=1)
        HH.OPS = bak
        mh = E.evaluate(inst, sc, bh, bs)
        gap = 100 * (mh["E_cost"] - bc) / bc
        w.writerow(dict(family=fam, n_cust=8, S=10, exact_cost=round(bc, 4),
                        hgls_cost=round(mh["E_cost"], 4), gap_pct=round(gap, 3),
                        explored=nexp, optimal=opt, exact_secs=round(ex_s, 1)))
        f.flush()
        print(f"[exact] {fam}: exact={bc:.3f} hgls={mh['E_cost']:.3f} "
              f"gap={gap:.2f}% ({ex_s:.0f}s)", flush=True)
    f.close()


BIG_ITER = 20000


# ---------------------------------------------------------------- 4 shifts ----
def _shift_scenarios(sc, kind, mag):
    q, tt, feas = sc.q.copy(), sc.tt.copy(), sc.feas.copy()
    if kind == "demand_mean":
        q = q * (1 + mag)
    elif kind == "demand_var":
        mu = q.mean(0, keepdims=True)
        q = np.maximum(mu + (q - mu) * np.sqrt(1 + mag), 0.0)
    elif kind == "congestion":
        tt = tt * (1 + mag)
    elif kind == "feas_drop":
        rng = np.random.default_rng(7)
        mask = rng.random(feas[..., 2].shape) < mag
        feas[..., 2] = np.where(mask, 0, feas[..., 2])
    return E.Scenarios(q, tt, feas)


SHIFTS = [("demand_mean", 0.10), ("demand_mean", 0.20), ("demand_mean", 0.30),
          ("demand_var", 0.25), ("demand_var", 0.50),
          ("congestion", 0.10), ("congestion", 0.20),
          ("feas_drop", 0.20)]


def stage_shifts(seed=0):
    path = os.path.join(RES, "shifts.csv")
    fields = ["family", "rep", "method", "kind", "mag"] + MET_KEYS
    done = _done(path, ["family", "rep", "method", "kind", "mag"])
    f, w = _csv(path, fields)
    for fam in SL.FAMILIES:
        for rep in [1, 2]:
            inst, tr, ca, te = SL.load_replicate(fam, rep)
            for method in ["DET", "SAA", "Q90C", "RO", "DFR"]:
                pp = _plan_path(fam, rep, method, seed)
                if not os.path.exists(pp):
                    continue
                r, s = _load_plan(pp)
                for kind, mag in SHIFTS:
                    if (fam, str(rep), method, kind, str(mag)) in done:
                        continue
                    m = X2.score(inst, _shift_scenarios(te, kind, mag), r, s)
                    row = {k: round(float(m[k]), 5) for k in MET_KEYS}
                    row.update(family=fam, rep=rep, method=method, kind=kind,
                               mag=mag)
                    w.writerow(row)
            f.flush()
            print(f"[shifts] {fam} r{rep} done", flush=True)
    f.close()


# ---------------------------------------------------------------- 5 scount ----
def stage_scount(seed=0):
    path = os.path.join(RES, "scount.csv")
    fields = ["family", "S_train", "set_id", "E_cost_test", "secs"]
    done = _done(path, ["family", "S_train", "set_id"])
    f, w = _csv(path, fields)
    for fam in ["nyc_manhattan", "peshawar_real", "solomon_c101"]:
        inst = SL.load_instance(fam)
        big = SL.generate(fam, 99, S=1000)      # common large evaluation set
        for S_train in [20, 40, 60, 100, 200]:
            for set_id in range(5):
                if (fam, str(S_train), str(set_id)) in done:
                    continue
                sc = SL.generate(fam, 100 + set_id, S=S_train)
                t0 = time.perf_counter()
                r, s, _ = X2.run_regime("SAA", inst, sc, sc, seed=seed)
                secs = time.perf_counter() - t0
                m = X2.score(inst, big, r, s)
                w.writerow(dict(family=fam, S_train=S_train, set_id=set_id,
                                E_cost_test=round(m["E_cost"], 4),
                                secs=round(secs, 1)))
                f.flush()
            print(f"[scount] {fam} S={S_train} done", flush=True)
    f.close()


# ---------------------------------------------------------------- 6 ablation --
ABL_STEPS = ["L0_det", "L1_scen_blind", "L2_recourse", "L3_dependence",
             "L4_conformal", "L5_dro", "L6_policy", "L7_safeguard"]


def _abl_plan(step, fam, rep, inst, thetas, seed=0):
    import model.robust as R
    tr_ind = None
    if step in ("L1_scen_blind", "L2_recourse"):
        sc_all = SL.generate(fam, rep, rho_d=0.0, sig_t=0.0)   # independent scenarios
    else:
        sc_all = SL.generate(fam, rep)
    mk = lambda sl: E.Scenarios(sc_all.q[sl], sc_all.tt[sl], sc_all.feas[sl])
    tr, ca = mk(SL.SPLITS["train"]), mk(SL.SPLITS["calib"])
    q75 = np.quantile(tr.q, 0.75, axis=0)
    if step == "L0_det":
        return X2.run_regime("DET", inst, tr, ca, seed=seed)[:2]
    if step == "L1_scen_blind":     # scenarios, but recourse-blind search objective
        r0, s0 = P.construct(inst, tr, q75, ortools_s=X2.ORT_S, seed=seed)
        import model.hgls as H
        fit = lambda rr, ss: E.evaluate(inst, tr, rr, ss, w_late=0, w_ret=0,
                                        w_miss=0, w_infeas=0)["fitness"]
        r, s, _ = H.hgls(inst, tr, r0, s0, max_iter=10**9, no_improve=10**9,
                         beta=0.0, seed=seed, fitness_fn=fit,
                         time_limit=X2.REFINE_S)
        return r, s
    if step == "L2_recourse":       # + recourse-aware objective (independent scen)
        r0, s0 = P.construct(inst, tr, q75, ortools_s=X2.ORT_S, seed=seed)
        import model.hgls as H
        r, s, _ = H.hgls(inst, tr, r0, s0, max_iter=10**9, no_improve=10**9,
                         beta=0.0, seed=seed, time_limit=X2.REFINE_S)
        return r, s
    if step == "L3_dependence":     # + joint scenarios (= SAA on dependent scen)
        return X2.run_regime("SAA", inst, tr, ca, seed=seed)[:2]
    if step == "L4_conformal":      # + conformal quantile protection
        return X2.run_regime("Q90C", inst, tr, ca, seed=seed)[:2]
    if step == "L5_dro":            # + DRO objective on the conformal plan
        r0, s0 = P.construct(inst, tr, q75, ortools_s=X2.ORT_S, seed=seed)
        import model.hgls as H
        r_ref, s_ref, _ = H.hgls(inst, tr, r0, s0, max_iter=10**9, no_improve=10**9,
                                 beta=0.0, seed=seed, time_limit=X2.REFINE_S / 3)
        z_tr = E.evaluate(inst, tr, r_ref, s_ref, return_z=True)["z"]
        z_ca = E.evaluate(inst, ca, r_ref, s_ref, return_z=True)["z"]
        rho = R.calibrate_rho(z_tr, z_ca)
        kap = R.conformal_kappa(inst, ca, r_ref, eps=0.20)
        qp = np.minimum(np.quantile(tr.q, 0.90, axis=0) * kap, inst.cap)
        fit = R.robust_fitness_fn(inst, tr, rho)
        r0, s0 = P.construct(inst, tr, qp, ortools_s=X2.ORT_S, seed=seed)
        r, s, _ = H.hgls(inst, tr, r0, s0, max_iter=10**9, no_improve=10**9,
                         beta=0.0, seed=seed, cap_demand=qp, fitness_fn=fit,
                         time_limit=X2.REFINE_S)
        return r, s
    if step == "L6_policy":         # + learned service levels (no safeguard)
        th = theta_for(fam, thetas)
        bak = P.SAFEGUARD_DELTA
        P.SAFEGUARD_DELTA = -1e9          # always deploy the learned plan
        r, s, _ = P.deploy(inst, tr, ca, th, seed=seed, time_limit=X2.REFINE_S,
                           ortools_s=X2.ORT_S)
        P.SAFEGUARD_DELTA = bak
        return r, s
    if step == "L7_safeguard":      # full proposed method
        th = theta_for(fam, thetas)
        r, s, _ = P.deploy(inst, tr, ca, th, seed=seed, time_limit=X2.REFINE_S,
                           ortools_s=X2.ORT_S)
        return r, s
    raise ValueError(step)


def stage_ablation(thetas, seed=0):
    path = os.path.join(RES, "ablation.csv")
    fields = ["family", "rep", "step"] + MET_KEYS + ["secs"]
    done = _done(path, ["family", "rep", "step"])
    f, w = _csv(path, fields)
    for fam in SL.FAMILIES:
        for rep in [1, 2]:
            inst, tr, ca, te = SL.load_replicate(fam, rep)
            for step in ABL_STEPS:
                if (fam, str(rep), step) in done:
                    continue
                t0 = time.perf_counter()
                r, s = _abl_plan(step, fam, rep, inst, thetas, seed)
                m = X2.score(inst, te, r, s)
                row = {k: round(float(m[k]), 5) for k in MET_KEYS}
                row.update(family=fam, rep=rep, step=step,
                           secs=round(time.perf_counter() - t0, 1))
                w.writerow(row); f.flush()
                print(f"[abl] {fam} r{rep} {step}: {m['E_cost']:.1f}", flush=True)
    f.close()


# ---------------------------------------------------------------- 7 repeats ---
def stage_repeats(thetas):
    path = os.path.join(RES, "repeats.csv")
    fields = ["family", "rep", "method", "seed"] + MET_KEYS + ["secs"]
    done = _done(path, ["family", "rep", "method", "seed"])
    f, w = _csv(path, fields)
    for fam in REP_FAMS:
        inst, tr, ca, te = SL.load_replicate(fam, 1)
        for method in ["SAA", "DFR"]:
            for seed in range(10):
                if (fam, "1", method, str(seed)) in done:
                    continue
                th = theta_for(fam, thetas) if method == "DFR" else None
                t0 = time.perf_counter()
                r, s, _ = X2.run_regime(method, inst, tr, ca, seed=seed, theta=th)
                m = X2.score(inst, te, r, s)
                row = {k: round(float(m[k]), 5) for k in MET_KEYS}
                row.update(family=fam, rep=1, method=method, seed=seed,
                           secs=round(time.perf_counter() - t0, 1))
                w.writerow(row); f.flush()
        print(f"[repeats] {fam} done", flush=True)
    f.close()


# ---------------------------------------------------------------- 8 sens ------
def stage_sens(thetas, seed=0):
    path = os.path.join(RES, "sens.csv")
    fields = ["family", "param", "mult", "method"] + MET_KEYS
    done = _done(path, ["family", "param", "mult", "method"])
    f, w = _csv(path, fields)
    base = {"C_LATE": E.C_LATE, "C_RET": E.C_RET, "C_MISS": E.C_MISS}
    for fam in ["nyc_manhattan", "peshawar_real", "solomon_c101"]:
        inst, tr, ca, te = SL.load_replicate(fam, 1)
        for param in ["C_LATE", "C_RET", "C_MISS"]:
            for mult in [0.5, 1.0, 2.0]:
                setattr(E, param, base[param] * mult)
                for method in ["SAA", "DFR"]:
                    if (fam, param, str(mult), method) in done:
                        continue
                    th = theta_for(fam, thetas) if method == "DFR" else None
                    r, s, _ = X2.run_regime(method, inst, tr, ca, seed=seed,
                                            theta=th)
                    m = X2.score(inst, te, r, s)
                    row = {k: round(float(m[k]), 5) for k in MET_KEYS}
                    row.update(family=fam, param=param, mult=mult, method=method)
                    w.writerow(row); f.flush()
                setattr(E, param, base[param])
            print(f"[sens] {fam} {param} done", flush=True)
    f.close()


def main(stages=None):
    stages = stages or ["policy", "main", "exact", "shifts", "scount",
                        "ablation", "repeats", "sens"]
    thetas = stage_policy()
    if "main" in stages:
        stage_main(thetas)
    if "exact" in stages:
        stage_exact()
    if "shifts" in stages:
        stage_shifts()
    if "scount" in stages:
        stage_scount()
    if "ablation" in stages:
        stage_ablation(thetas)
    if "repeats" in stages:
        stage_repeats(thetas)
    if "sens" in stages:
        stage_sens(thetas)
    print("CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
