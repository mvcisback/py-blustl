"""Module for taking a 1 off game and turning into a MPC games."""
# TODO: incorporate external measurements
from collections import deque
from itertools import product

import stl
import traces
import funcy as fn
from lenses import bind

from magnum.utils import solution_to_stl
from magnum.game import Game, Specs
from magnum.solvers.cegis import solve


def horizons(N, dt):
    i = 0
    while True:
        yield i*dt
        if i < N-1:
            i += 1


def mpc_games(g):
    for dH in horizons(len(g.times), g.model.dt):
        yield g.new_horizon(dH)


def mpc(orig: Game, *, endless=True, use_smt=False, shift_output_time=True):
    solution_spec = stl.TOP
    N = orig.scope
    for i, g in enumerate(mpc_games(orig)):
        g = bind(g).specs.learned.set(solution_spec)

        # Forget about initial condition
        if i > 0:
            g = bind(g).specs.init.set(stl.TOP)

        # TODO: check if we can reuse/extend counter_examples
        res, _ = solve(g, use_smt=use_smt)

        if not res.feasible:
            return

        offset = i > N
        j = min(i, N-1)
        times = range(j+1)



        inputs = g.model.vars.state
        dt = g.model.dt
        solution_spec = solution_to_stl(inputs, res.solution, dt, times[:N], 
                                        offset=offset)

        solution = res.solution
        if shift_output_time and offset:
            solution = fn.walk_values(
                lambda trace: traces.TimeSeries([(t+(i-N)*dt, v) for t, v in trace.items()]), 
                solution)

        yield solution
