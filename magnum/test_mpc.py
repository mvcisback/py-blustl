import stl
import traces

from magnum.mpc import mpc, solutions_to_stl

def test_mpc_smoke():
    from magnum.examples.feasible_example import feasible_example as g
    
    results = mpc(g)
    for _ in range(10):
        next(results)


def test_solutions_to_stl():
    solutions = [
        {
            'u': traces.TimeSeries([(0, 0)]), 
            'w': traces.TimeSeries([(0, 0)])
        },
        {
            'u': traces.TimeSeries([(0, 0), (1, 1)]),
            'w': traces.TimeSeries([(0, 0), (1, 2)])
        },
        {
            'u': traces.TimeSeries([(0, 0), (1, 1), (2, 3)]),
            'w': traces.TimeSeries([(0, 0), (1, 2), (2, -1)])
        }
    ]
    phi = solutions_to_stl(['u'], solutions, 1)
    psi = stl.parse('(u = 0) & (X(u = 1)) & (X(X(u = 3)))')
    assert phi == psi

    phi = solutions_to_stl(['u', 'w'], solutions, 1)
    psi2 = stl.parse('(w = 0) & (X(w = 2)) & (X(X(w = -1)))')
    assert phi == psi & psi2
