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

# recourse unit costs (litres-equivalent, so everything is comparable in fuel terms)
C_LATE = 0.05         # per minute of lateness
C_OVER = 2.00         # per m3 of overload (forces a depot return)
C_MISS = 50.0         # per unserved node
C_INFEAS_SPEED = 0.5  # per arc using an infeasible eco-speed (soft penalty)


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
             w_late=None, w_over=None, w_miss=None, w_infeas=None):
    # default to current module-level weights at call time (enables sensitivity sweeps)
    w_late = C_LATE if w_late is None else w_late
    w_over = C_OVER if w_over is None else w_over
    w_miss = C_MISS if w_miss is None else w_miss
    w_infeas = C_INFEAS_SPEED if w_infeas is None else w_infeas
    """
    Vectorised scenario-aware evaluation.

    routes  : list of routes (each a list of customer indices, no depot)
    speeds  : list of per-arc speed-level index lists; len(speeds[r]) == len(routes[r])+1
    beta    : CVaR weight (0 = expected only)
    alpha   : CVaR level
    use_mean_demand : if True, evaluate deterministic mean-value model (D-HGLS baseline)
    mean_tt : optional [N,N,3] travel-time to use instead of scenarios (for PTO/det.)

    Returns dict with expected fuel/emission/recourse, CVaR, reliability metrics, fitness.
    """
    S, N = sc.S, inst.N
    q = inst.demand_base[None, :].repeat(S, 0) if use_mean_demand else sc.q       # [S,N]
    if use_mean_demand:
        q = sc.q.mean(0)[None, :].repeat(S, 0)

    fuel = np.zeros(S); late = np.zeros(S); over = np.zeros(S)
    infeas = np.zeros(S); miss = np.zeros(S)
    served = np.zeros(N, bool); served[inst.depot] = True

    for r, route in enumerate(routes):
        if len(route) == 0:
            continue
        sp = speeds[r]
        path = [inst.depot] + list(route) + [inst.depot]
        load = np.zeros(S)                 # current load per scenario (pickup grows)
        t = np.full(S, inst.tw[inst.depot, 0])   # depart depot at window open
        route_q = q[:, route]              # [S, len(route)]
        total_route_q = route_q.sum(1)     # for capacity check
        for k in range(1, len(path)):
            i, j = path[k-1], path[k]
            s_idx = sp[k-1]
            # travel time
            if mean_tt is not None:
                ttk = np.full(S, mean_tt[i, j, s_idx])
            else:
                ttk = sc.tt[:, i, j, s_idx]
            # eco-speed feasibility (soft)
            infeas += (1 - sc.feas[:, i, j, s_idx])
            # load fraction on this arc (load BEFORE servicing j)
            loadfrac = np.clip(load / inst.cap, 0, 1)
            fuel += _arc_fuel(inst.D[i, j], loadfrac, s_idx)
            # arrive, wait to window open, service
            t = t + ttk
            if j != inst.depot:
                served[j] = True
                t = np.maximum(t, inst.tw[j, 0])
                late += np.maximum(0.0, t - inst.tw[j, 1])
                t = t + inst.service[j]
                load = load + q[:, j]
        # capacity overload recourse (load can exceed Q under high-demand scenarios)
        over += np.maximum(0.0, total_route_q - inst.cap)

    # unserved nodes (customers not in any route)
    miss[:] = float((~served).sum())

    emission = XI * fuel
    recourse = w_late * late + w_over * over + w_miss * miss + w_infeas * infeas
    cost = fuel + recourse                      # litres-equivalent total
    z = cost
    cvar = _cvar(z, alpha) if beta > 0 else 0.0
    fitness = z.mean() + beta * cvar

    return {
        "fitness": float(fitness),
        "E_fuel": float(fuel.mean()), "E_emission": float(emission.mean()),
        "E_recourse": float(recourse.mean()), "E_cost": float(cost.mean()),
        "CVaR_cost": float(cvar) if beta > 0 else float(_cvar(z, alpha)),
        "worst_cost": float(z.max()),
        "P_late": float((late > 1e-6).mean()), "E_late_min": float(late.mean()),
        "P_overload": float((over > 1e-6).mean()), "E_overload_m3": float(over.mean()),
        "missed_nodes": float(miss.mean()),
        "infeas_speed_arcs": float(infeas.mean()),
    }


def _cvar(z, alpha):
    """CVaR_alpha = mean of worst (1-alpha) fraction of z."""
    k = max(1, int(np.ceil((1 - alpha) * len(z))))
    return float(np.sort(z)[-k:].mean())
