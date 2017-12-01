import funcy as fn
from lenses import lens

import magnum
from magnum.solvers import smt
from magnum.solvers import milp
from magnum.utils import Result


class MaxRoundsError(Exception):
    pass


def combined_solver(*args, **kwargs):
    res = milp.encode_and_run(*args, **kwargs)
    # If milp can't decide use smt
    if res.cost == 0:
        res = smt.encode_and_run(*args, **kwargs)
    return res


def cegis_loop(g, max_rounds=10, use_smt=False, max_ce=float('inf')):
    """CEGIS for dominant/robust strategy.
    ∃u∀w . (x(u, w, t), u, w) ⊢ φ
    """
    # Create player for sys and env resp.
    g_inv = g.invert()

    counter_examples = []
    solve = smt.encode_and_run if use_smt else combined_solver
    for i in range(max_rounds):
        play = solve(g, counter_examples=counter_examples)
        if not play.feasible:
            return Result(False, None, None), counter_examples
        
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


