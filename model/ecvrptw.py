"""
Core LA-SEco-ECVRPTW model: instance loader, fuel/emission model, and a fully
vectorised scenario-aware evaluator (fuel, time-window reliability, capacity,
recourse, CVaR). Pickup convention (waste collection): vehicle load grows along route.

Units: all times in MINUTES; distances in KM; demand/capacity in m3 (or Solomon units);
fuel in litres; emissions in kg CO2e.

A *solution* is a list of routes; each route is a list of customer node indices
(depot 0 implied at both ends) with a parallel list of per-arc speed-level indices
(0=low,1=med,2=high). Arc k of a route is (route[k-1] -> route[k]); arc 0 is
(depot -> route[0]); final arc is (route[-1] -> depot).
"""
import os, json
import numpy as np

# physics fuel model (manuscript): rho(load,speed) = (rho0 + (rhoQ-rho0)*loadfrac) * phi_s
RHO0 = 0.20            # empty-load L/km
RHOQ = 0.50            # full-load  L/km
PHI = np.array([0.90, 1.00, 1.18])    # speed factor low/med/high (faster -> more fuel/km)
XI = 2.36             # kg CO2e per litre (manuscript)

# recourse parameters (litres-equivalent where monetised, so all terms are
# commensurable with fuel). The emergency depot return is SIMULATED, not priced:
# its distance, fuel, and time are added physically to the route.
C_LATE = 0.05         # per minute of post-recourse lateness
C_RET = 2.00          # fixed cost per emergency return (labour/administration)
C_MISS = 50.0         # per deferred or unserved customer
C_INFEAS_SPEED = 0.5  # per arc using an infeasible eco-speed (soft penalty)
UNLOAD_MIN = 10.0     # depot unload time during an emergency return (minutes)
DEPOT_SPEED = 1       # speed level used on emergency depot legs (medium)


class Instance:
    def __init__(self, name, D, demand_base, cap, tw, service, sp_kmh, depot=0):
        self.name = name
        self.D = D                      # [N,N] km
        self.N = D.shape[0]
        self.demand_base = demand_base  # [N] nominal demand (depot=0)
        self.cap = cap
        self.tw = tw                    # [N,2] minutes
        self.service = service          # [N] minutes
        self.sp_kmh = sp_kmh            # [3]
        self.depot = depot


class Scenarios:
    def __init__(self, q, tt, feas):
        self.q = q                      # [S,N]
        self.tt = tt                    # [S,N,N,3] minutes
        self.feas = feas                # [S,N,N,3] {0,1}
        self.S = q.shape[0]


def load(name, root=None):
    root = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data_pipeline", "instances")
    d = os.path.join(root, name)
    inst = json.load(open(os.path.join(d, "instance.json")))
    z = np.load(os.path.join(d, "scenarios.npz"))
    N = inst["n_nodes"]
    D = np.array(inst["distance_km"], float)
    tw = np.array(inst["time_window"], float)
    if tw.max() <= 24.0 + 1e-9:         # hours -> minutes
        tw = tw * 60.0
    sv = inst["service_time_min"]
    service = np.full(N, float(sv)) if np.isscalar(sv) else np.array(sv, float)
    service[inst["depot"]] = 0.0
    sp = np.array(inst["speed_levels_kmh"], float) if isinstance(inst["speed_levels_kmh"], list) \
        else np.array(list(inst["speed_levels_kmh"].values()), float)
    instance = Instance(name, D, np.array(inst["base_demand_m3"], float),
                        float(inst["vehicle_capacity_m3"]), tw, service, sp, inst["depot"])
    sc = Scenarios(z["q"].astype(float), z["tt"].astype(float), z["feas"].astype(np.int8))
    return instance, sc


def _arc_fuel(d_km, loadfrac, s_idx):
    """Litres on one arc given load fraction (0..1) and speed-level index."""
    rho = (RHO0 + (RHOQ - RHO0) * np.clip(loadfrac, 0, 1)) * PHI[s_idx]
    return d_km * rho


def evaluate(inst: Instance, sc: Scenarios, routes, speeds, beta=0.0, alpha=0.9,
             use_mean_demand=False, mean_tt=None,
             w_late=None, w_ret=None, w_miss=None, w_infeas=None,
             return_z=False):
    """
    Vectorised scenario-aware evaluation with SIMULATED recourse.

    Recourse timeline (second stage, per scenario): the route and planned speeds are
    fixed. Service proceeds sequentially. On arrival at customer j, if the realised
    demand q_j(omega) does not fit the residual capacity, the vehicle performs an
    emergency depot round trip from j (travel j->0 loaded, unload for UNLOAD_MIN,
    travel 0->j empty), then serves j. If a planned speed level is not achievable,
    the realised travel time already reflects the lower achievable speed. Service
    that cannot start before the shift horizon is deferred at a penalty. Lateness
    within the horizon is penalised per minute.

    routes  : list of routes (each a list of customer indices, no depot)
    speeds  : per-arc speed-level index lists; len(speeds[r]) == len(routes[r])+1
    beta    : CVaR weight in the fitness (0 = expected cost only)
    alpha   : CVaR level
    use_mean_demand : evaluate under mean demand (deterministic planning view)
    mean_tt : optional [N,N,3] travel time instead of scenarios (deterministic view)
    return_z: also return the per-scenario cost vector (for DRO / calibration)

    Reported failure metrics are distinguished per roadmap:
      P_trigger  : probability that at least one emergency return occurs (route failure)
      E_returns  : expected number of emergency returns per day
      E_return_km: expected extra distance driven due to returns
      P_late     : probability of any post-recourse lateness
      P_defer    : probability that at least one customer is deferred
    """
    # default to module-level weights at call time (enables sensitivity sweeps)
    w_late = C_LATE if w_late is None else w_late
    w_ret = C_RET if w_ret is None else w_ret
    w_miss = C_MISS if w_miss is None else w_miss
    w_infeas = C_INFEAS_SPEED if w_infeas is None else w_infeas

    S, N = sc.S, inst.N
    if use_mean_demand:
        q = sc.q.mean(0)[None, :].repeat(S, 0)
    else:
        q = sc.q
    depot = inst.depot
    shift_end = inst.tw[depot, 1]

    fuel = np.zeros(S); late = np.zeros(S); infeas = np.zeros(S)
    returns = np.zeros(S); return_km = np.zeros(S); defer = np.zeros(S)
    served = np.zeros(N, bool); served[depot] = True

    def _tt(i, j, s_idx):
        if mean_tt is not None:
            return np.full(S, mean_tt[i, j, s_idx])
        return sc.tt[:, i, j, s_idx]

    for r, route in enumerate(routes):
        if len(route) == 0:
            continue
        sp = speeds[r]
        path = [depot] + list(route) + [depot]
        load = np.zeros(S)
        t = np.full(S, float(inst.tw[depot, 0]))
        for k in range(1, len(path)):
            i, j = path[k-1], path[k]
            s_idx = sp[k-1]
            infeas += (1 - sc.feas[:, i, j, s_idx])
            loadfrac = np.clip(load / inst.cap, 0, 1)
            fuel += _arc_fuel(inst.D[i, j], loadfrac, s_idx)
            t = t + _tt(i, j, s_idx)
            if j == depot:
                continue
            served[j] = True
            # --- emergency return recourse: q_j does not fit residual capacity ---
            trig = (load + q[:, j] > inst.cap + 1e-9)
            if trig.any():
                d_leg = inst.D[j, depot]
                lf = np.clip(load / inst.cap, 0, 1)
                # loaded leg j->0 + empty leg 0->j, at the depot speed level
                extra_fuel = (_arc_fuel(d_leg, lf, DEPOT_SPEED)
                              + _arc_fuel(inst.D[depot, j], 0.0, DEPOT_SPEED))
                extra_t = (_tt(j, depot, DEPOT_SPEED) + _tt(depot, j, DEPOT_SPEED)
                           + UNLOAD_MIN)
                fuel += np.where(trig, extra_fuel, 0.0)
                t = t + np.where(trig, extra_t, 0.0)
                return_km += np.where(trig, d_leg + inst.D[depot, j], 0.0)
                returns += trig
                load = np.where(trig, 0.0, load)
            # --- service, deferment, lateness (post-recourse) ---
            start = np.maximum(t, inst.tw[j, 0])
            dfr = (start > shift_end + 1e-9)
            if dfr.any():
                defer += dfr
                # deferred customers are skipped: no service time, no pickup
                late += np.where(dfr, 0.0, np.maximum(0.0, start - inst.tw[j, 1]))
                t = np.where(dfr, t, start + inst.service[j])
                load = np.where(dfr, load, load + q[:, j])
            else:
                late += np.maximum(0.0, start - inst.tw[j, 1])
                t = start + inst.service[j]
                load = load + q[:, j]

    unrouted = float((~served).sum())
    emission = XI * fuel
    recourse = (w_late * late + w_ret * returns + w_miss * (defer + unrouted)
                + w_infeas * infeas)
    cost = fuel + recourse
    z = cost
    cvar = _cvar(z, alpha)
    fitness = z.mean() + beta * cvar if beta > 0 else z.mean()

    out = {
        "fitness": float(fitness),
        "E_fuel": float(fuel.mean()), "E_emission": float(emission.mean()),
        "E_recourse": float(recourse.mean()), "E_cost": float(cost.mean()),
        "CVaR_cost": float(cvar), "worst_cost": float(z.max()),
        "P_trigger": float((returns > 0).mean()), "E_returns": float(returns.mean()),
        "E_return_km": float(return_km.mean()),
        "P_late": float((late > 1e-6).mean()), "E_late_min": float(late.mean()),
        "P_defer": float((defer > 0).mean()), "E_deferred": float(defer.mean()),
        "unrouted": unrouted,
        "infeas_speed_arcs": float(infeas.mean()),
    }
    if return_z:
        out["z"] = z
    return out


def _cvar(z, alpha):
    """CVaR_alpha = mean of worst (1-alpha) fraction of z."""
    k = max(1, int(np.ceil((1 - alpha) * len(z))))
    return float(np.sort(z)[-k:].mean())
