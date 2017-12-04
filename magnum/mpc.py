"""Module for taking a 1 off game and turning into a MPC games."""
# TODO: incorporate external measurements
from collections import deque
from itertools import product

import stl
import traces
import funcy as fn
from lenses import bind

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


def solution_to_stl(inputs, sol, dt, times, offset=False):
    def _lineq(var_t):
        var, i = var_t
        t = i+1 if offset else i
        phi = stl.parse(f"{var} = {sol[var][dt*(t)]:f}")
        return stl.utils.next(phi, i=i)

    return stl.andf(*map(_lineq, product(inputs, times)))


def mpc(orig: Game, *, endless=True):
    solution_spec = stl.TOP
    N = len(orig.times)
    for i, g in enumerate(mpc_games(orig)):
        g = bind(g).specs.learned.modify(lambda x: x & solution_spec)

        # Forget about initial condition
        if i > 0:
            g = bind(g).specs.init.set(stl.TOP)

        # TODO: check if we can reuse/extend counter_examples
        res, _ = solve(g)

        if not res.feasible:
            return

        if i+1 < N:
            times = range(i+1)
            offset=False
        else:
            times = range(N)
            offset=True

        inputs = g.model.vars.state
        dt = g.model.dt
        solution_spec = solution_to_stl(inputs, res.solution, dt, times, offset=offset)

        yield res.solution
