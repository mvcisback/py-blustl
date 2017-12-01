import funcy as fn
from lenses import lens

import magnum
from magnum.solvers import smt
from magnum.solvers import milp
from magnum.utils import Result


def cegis_loop(g, max_rounds=10, use_smt=False):
    """CEGIS for dominant/robust strategy.
    ∃u∀w . (x(u, w, t), u, w) ⊢ φ
    """
    # Create player for sys and env resp.
    g_inv = g.invert()

    counter_examples = []

    solve = smt.encode_and_run if use_smt else milp.encode_and_run
    for i in range(max_rounds):
        play = solve(g, counter_examples=counter_examples)
        if not play.feasible:
            return Result(False, None, None), counter_examples
        
        solution = fn.project(play.solution, g.model.vars.input)
        counter = solve(g_inv, counter_examples=[solution])
        if not counter.feasible:
            return play, counter_examples

        counter_examples.append(fn.project(counter.solution, g.model.vars.env))

    raise NotImplementedError
    


def banned_square(prev_input, cost, g, eps=1):
    if g.meta.drdu == oo:
        return prev_input

    R = (cost + eps) / g.meta.drdu
    delta = R - prev_input.const
    lower = lens(lens(prev_input).op.set("<")).const.set(delta)
    upper = lens(lens(prev_input).op.set(">")).const.set(-delta)
    return lower & upper
