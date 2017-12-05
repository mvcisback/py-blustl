from itertools import product

import stl
import funcy as fn
from lenses import bind

import magnum
from magnum.solvers import smt
from magnum.solvers import milp
from magnum.utils import encode_refuted_rec


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


def solve(g, max_rounds=4, use_smt=False, max_ce=float('inf'), 
          refuted_recs=True, bloat=0):
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
        elif refuted_recs:
            r = find_refuted_radius(g, solution, move) + bloat
            times = list(g.times)[:-1]
            phi = encode_refuted_rec(solution, r, times, dt=g.model.dt)
            g = bind(g).specs.learned.modify(lambda x: x & phi)

    raise MaxRoundsError


@fn.autocurry
def smt_radius_oracle(counter, play, g, r):
    rec = ~encode_refuted_rec(play, r, g.times, dt=g.model.dt)
    if rec == stl.BOT:
        rec = stl.TOP

    g = bind(g).specs.learned.set(rec)
    return smt.encode_and_run(g, counter_examples=[counter]).feasible


def find_refuted_radius(g, u_star, w_star, tol=1e-2):
    oracle = smt_radius_oracle(counter=w_star, play=u_star, g=g)
    r_low, r_high, r_mid = 0, 1, 1

    while r_high - r_low > tol:
        if not oracle(r=r_mid):
            r_low = r_mid
        else:
            r_high = r_mid
        r_mid = (r_low + r_high) / 2.

    return r_high
