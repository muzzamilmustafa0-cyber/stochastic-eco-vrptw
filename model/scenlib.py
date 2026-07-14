"""
Joint scenario generation with calibrated cross-node / cross-arc dependence.

Replicates are generated deterministically from (family, replicate) seeds, so no
bulk scenario tensors need to be stored: every experiment regenerates identical
scenarios from the instance geometry, the arc-speed lookups, and two dependence
parameters calibrated from real data (data_pipeline/scripts/08):

  rho_demand : share of log-demand variance common to all nodes on a day (Austin)
  sigma_traf : across-date std of log median achievable speed at fixed hour (TLC)

Demand:   log q_i = mu_i + sigma_i * (sqrt(rho) * Z_day + sqrt(1-rho) * Z_i)
          (marginal mean and CV per node preserved; cross-node correlation rho)
Traffic:  one operating hour and one common day factor exp(sigma_traf * W - .5s^2)
          per scenario shift ALL arc speeds together; arc-level lognormal noise
          around the (real, per OD x hour where available) mean speed.

Each replicate provides disjoint scenario blocks:
  train 0:60 | calibration 60:160 | test 160:360   (indices via SPLITS)
"""
import os, json
import numpy as np
from . import ecvrptw as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data_pipeline", "processed")
INST = os.path.join(ROOT, "data_pipeline", "instances")

S_TOTAL = 360
SPLITS = {"train": slice(0, 60), "calib": slice(60, 160), "test": slice(160, 360)}

_dep = json.load(open(os.path.join(PROC, "dependence_calibration.json")))
RHO_D = float(_dep["rho_demand_intraday"])
SIG_T = float(_dep["sigma_traffic_common_log"])
_cong = json.load(open(os.path.join(PROC, "congestion_profile.json")))
CONG = np.array([_cong[str(h)] for h in range(24)], float)

# per-family travel-time regime (mirrors the original fuse scripts)
PROFILE = {
    "dublin_real":  dict(freeflow=42.0, arc_sig=0.22, tol=3.0, shift=(8, 16)),
    "peshawar_real": dict(freeflow=36.0, arc_sig=0.22, tol=3.0, shift=(8, 16)),
}
SOLOMON = dict(freeflow=60.0, arc_sig=0.22, tol=3.0, shift=(6, 18))
NYC = dict(tol=5.0, shift=(8, 16))
MPH2KMH = 1.60934

_lut_cache = {}


def _meta(family):
    return json.load(open(os.path.join(INST, family, "instance.json")))


def load_instance(family):
    inst, _ = E.load(family)
    return inst


def _arclut(family):
    if family not in _lut_cache:
        z = np.load(os.path.join(PROC, f"arclut_{family}.npz"))
        _lut_cache[family] = (z["mean_mph"].astype(float), z["cv"].astype(float))
    return _lut_cache[family]


def generate(family, replicate, S=S_TOTAL):
    """Generate the scenario set for (family, replicate). Deterministic."""
    meta = _meta(family)
    rng = np.random.default_rng(abs(hash((family, int(replicate)))) % (2**32))
    N = meta["n_nodes"]
    D = np.array(meta["distance_km"], float)
    base = np.array(meta["base_demand_m3"], float)
    cv = np.array(meta["demand_cv"], float)
    sp_kmh = np.array(list(meta["speed_levels_kmh"].values()), float)

    # ---- demand with common day factor (marginals preserved) ----
    sig = np.sqrt(np.log1p(cv**2))
    mu = np.where(base > 0, np.log(np.maximum(base, 1e-9)) - 0.5 * sig**2, -np.inf)
    z_day = rng.standard_normal(S)[:, None]
    z_i = rng.standard_normal((S, N))
    q = np.exp(mu[None, :] + sig[None, :] * (np.sqrt(RHO_D) * z_day
                                             + np.sqrt(1 - RHO_D) * z_i))
    q[:, meta["depot"]] = 0.0
    q[np.isinf(mu)[None, :].repeat(S, 0)] = 0.0

    # ---- achievable speed field per scenario ----
    if family.startswith("nyc_"):
        reg = NYC
        mean_mph, arc_cv = _arclut(family)
    elif family.startswith("solomon_"):
        reg = SOLOMON
    else:
        reg = PROFILE[family]
    lo, hi = reg["shift"]
    hours = rng.integers(lo, hi + 1, S)
    w_day = np.exp(SIG_T * rng.standard_normal(S) - 0.5 * SIG_T**2)

    tt = np.empty((S, N, N, 3), np.float32)
    feas = np.empty((S, N, N, 3), np.int8)
    eye = np.eye(N, dtype=bool)
    for s in range(S):
        h = int(hours[s])
        if family.startswith("nyc_"):
            m_kmh = mean_mph[:, :, h] * MPH2KMH
            c = arc_cv[:, :, h]
            noise = np.exp(np.sqrt(np.log1p(c**2)) * rng.standard_normal((N, N))
                           - 0.5 * np.log1p(c**2))
            v = m_kmh * noise * w_day[s]
            v = np.clip(v, 3.0, 120.0)
        else:
            base_v = reg["freeflow"] * CONG[h]
            sgl = reg["arc_sig"]
            noise = np.exp(sgl * rng.standard_normal((N, N)) - 0.5 * sgl**2)
            v = np.clip(base_v * noise * w_day[s], 5.0, reg["freeflow"] + 10.0)
        for li, vl in enumerate(sp_kmh):
            veff = np.minimum(vl, v)
            with np.errstate(divide="ignore"):
                tt[s, :, :, li] = np.where(eye, 0.0, D / np.maximum(veff, 1e-3) * 60.0)
            feas[s, :, :, li] = (v >= vl - reg["tol"])
    return E.Scenarios(q.astype(float), tt.astype(float), feas)


def load_replicate(family, replicate, S=S_TOTAL):
    """Return (instance, sc_train, sc_calib, sc_test) for one operational replicate."""
    inst = load_instance(family)
    sc = generate(family, replicate, S)
    mk = lambda sl: E.Scenarios(sc.q[sl], sc.tt[sl], sc.feas[sl])
    return inst, mk(SPLITS["train"]), mk(SPLITS["calib"]), mk(SPLITS["test"])


FAMILIES = ["nyc_manhattan", "nyc_queens", "nyc_brooklyn", "dublin_real",
            "peshawar_real", "solomon_c101", "solomon_c201", "solomon_r102",
            "solomon_r202", "solomon_rc101", "solomon_rc201"]
REPLICATES = [1, 2, 3, 4, 5]
