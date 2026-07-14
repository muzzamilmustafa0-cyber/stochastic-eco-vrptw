"""Unit tests for the recourse simulator against hand-computed values.

Toy layout: depot 0 and customers 1, 2, all pairwise 10 km apart. Every arc takes
12 minutes at every speed level. Capacity 10. Service 10 min. Wide windows [0, 480].
Fuel model: rho(load) = 0.2 + 0.3 * loadfrac, times phi = [0.9, 1.0, 1.18].
"""
import numpy as np
from . import ecvrptw as E


def _toy(q_row, tw=None, tt_const=12.0):
    N = 3
    D = np.full((N, N), 10.0); np.fill_diagonal(D, 0.0)
    tw = np.array(tw if tw is not None else [[0, 480]] * N, float)
    inst = E.Instance("toy", D, np.array([0.0, 5.0, 5.0]), 10.0, tw,
                      np.array([0.0, 10.0, 10.0]), np.array([30.0, 50.0, 70.0]))
    S = len(q_row)
    q = np.array(q_row, float)
    tt = np.full((S, N, N, 3), tt_const)
    feas = np.ones((S, N, N, 3), np.int8)
    return inst, E.Scenarios(q, tt, feas)


def test_no_recourse():
    inst, sc = _toy([[0, 4, 4]])
    m = E.evaluate(inst, sc, [[1, 2]], [[1, 1, 1]])
    # fuel: 0->1 empty 2.0; 1->2 lf .4 -> rho .32 -> 3.2; 2->0 lf .8 -> rho .44 -> 4.4
    assert abs(m["E_fuel"] - 9.6) < 1e-9, m["E_fuel"]
    assert m["E_returns"] == 0 and m["P_trigger"] == 0 and m["P_late"] == 0
    assert abs(m["E_cost"] - 9.6) < 1e-9


def test_emergency_return():
    inst, sc = _toy([[0, 6, 6]])
    m = E.evaluate(inst, sc, [[1, 2]], [[1, 1, 1]])
    # 0->1: 2.0 | 1->2 lf .6: 3.8 | return from 2 loaded lf .6: 3.8 + empty 2.0
    # after reset serve 2 (load 6); 2->0 lf .6: 3.8  => fuel 15.4
    assert abs(m["E_fuel"] - 15.4) < 1e-9, m["E_fuel"]
    assert m["E_returns"] == 1 and m["P_trigger"] == 1
    assert abs(m["E_return_km"] - 20.0) < 1e-9
    # cost = fuel + C_RET * 1
    assert abs(m["E_cost"] - (15.4 + E.C_RET)) < 1e-9
    assert m["P_late"] == 0 and m["P_defer"] == 0


def test_lateness():
    inst, sc = _toy([[0, 4, 4]], tw=[[0, 480], [0, 480], [0, 30]])
    m = E.evaluate(inst, sc, [[1, 2]], [[1, 1, 1]])
    # arrive 2 at t = 12 + 10 + 12 = 34 > b=30 -> 4 min late (within shift)
    assert abs(m["E_late_min"] - 4.0) < 1e-9, m["E_late_min"]
    assert m["P_late"] == 1 and m["P_defer"] == 0
    assert abs(m["E_cost"] - (9.6 + E.C_LATE * 4.0)) < 1e-9


def test_deferment():
    inst, sc = _toy([[0, 4, 4]], tt_const=250.0)
    m = E.evaluate(inst, sc, [[1, 2]], [[1, 1, 1]])
    # arrive 1 at 250 (ok), serve till 260; arrive 2 at 510 > shift end 480 -> deferred
    assert m["P_defer"] == 1 and m["E_deferred"] == 1
    # deferred customer picked up nothing: fuel legs 0->1 empty 2.0, 1->2 lf .4 3.2,
    # 2->0 still lf .4 (no pickup at 2) 3.2 => 8.4
    assert abs(m["E_fuel"] - 8.4) < 1e-9, m["E_fuel"]
    assert abs(m["E_cost"] - (8.4 + E.C_MISS)) < 1e-9


def test_scenario_mix():
    # scenario 1 fits, scenario 2 triggers a return: metrics are scenario means
    inst, sc = _toy([[0, 4, 4], [0, 6, 6]])
    m = E.evaluate(inst, sc, [[1, 2]], [[1, 1, 1]])
    assert abs(m["E_returns"] - 0.5) < 1e-9
    assert abs(m["P_trigger"] - 0.5) < 1e-9
    assert abs(m["E_fuel"] - (9.6 + 15.4) / 2) < 1e-9


if __name__ == "__main__":
    test_no_recourse(); test_emergency_return(); test_lateness()
    test_deferment(); test_scenario_mix()
    print("all recourse unit tests passed")
