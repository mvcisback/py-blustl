from itertools import product

import stl
import funcy as fn
from lenses import lens

import magnum
from magnum.solvers import smt
from magnum.solvers import milp


class MaxRoundsError(Exception):
    pass


def combined_solver(*args, **kwargs):
    res = milp.encode_and_run(*args, **kwargs)
    # If milp can't decide use smt
    if res.cost == 0:
        res = smt.encode_and_run(*args, **kwargs)
    return res


def round_counter(max_rounds):
    i = 0
    while i < max_rounds:
        yield i
        i += 1


def solve(g, max_rounds=4, use_smt=False, max_ce=float('inf')):
    """CEGIS for dominant/robust strategy.
    ∃u∀w . (x(u, w, t), u, w) ⊢ φ
    """
    # Create player for sys and env resp.
    g_inv = g.invert()

    counter_examples = []
    solve = smt.encode_and_run if use_smt else combined_solver
    for _ in round_counter(max_rounds):
        play = solve(g, counter_examples=counter_examples)
        if not play.feasible:
            return play, counter_examples
        
        solution = fn.project(play.solution, g.model.vars.input)
        counter = solve(g_inv, counter_examples=[solution])
        if not counter.feasible:
            return play, counter_examples

        move = fn.project(counter.solution, g.model.vars.env)
        if len(counter_examples) < max_ce:
            counter_examples.append(move)
        else:
            # TODO: grow learned clauses
            pass

    raise MaxRoundsError


def encode_refuted_rec(refuted_input, radius, times):
    def _encode_refuted(name_time):
        u, t = name_time
        val = refuted_input[u][t]
        lo, hi = val - radius, val + radius
        phi = stl.BOT
        if lo > 0:
            phi |= stl.parse(f'{u} >= {lo}')

        if hi < 1:
            phi |= stl.parse(f'{u} <= {hi}')

        if phi == stl.BOT:
            return stl.TOP
        else:
            return stl.utils.next(phi, i=t)
    
    return stl.andf(*map(_encode_refuted, product(refuted_input.keys(), times)))
