"""Module for taking a 1 off game and turning into a MPC games."""
# TODO: incorporate external measurements
from collections import deque
from itertools import product

import stl
import sympy as sym
import funcy as fn
from lenses import bind

from magnum.game import Game, Specs
from magnum.solvers.cegis import solve


def horizons(N, dt):
    i = 0
    while True:
        yield i*dt
        if i < 2*N:
            i += 1


def mpc_games(g):
    for dH in horizons(len(g.times), g.model.dt):
        yield g.new_horizon(dH)


def solutions_to_stl(inputs, solutions, dt):
    def _lineq(var_t_sol):
        var, (t, sol) = var_t_sol
        phi = stl.parse(f"{var} = {sol[var][dt*t]}")
        return stl.utils.next(phi, i=t)

    timed_solutions = enumerate(solutions)

    return stl.andf(*map(_lineq, product(inputs, timed_solutions)))


def mpc(orig: Game, *, endless=True):
    solutions = deque(maxlen=2*len(orig.times))
    for i, g in enumerate(mpc_games(orig)):
        g = bind(g).specs.learned.modify(
            lambda x: x & solutions_to_stl(
                fn.cat(g.model.vars), solutions, g.model.dt))

        # TODO: check if we can reuse/extend counter_examples
        res, _ = solve(g)

        if not res.feasible:
            return

        yield res.solution
        solutions.append(res.solution)

    
