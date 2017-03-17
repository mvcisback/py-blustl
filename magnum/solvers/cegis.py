from itertools import cycle

import stl
import funcy as fn
from lenses import lens

import magnum
from magnum.solvers.milp import encode_and_run as predict
from magnum.utils import project_solution_stl


def cegis(g):
    return fn.last(cegis_loop(g))


def cegis_loop(g):
    """CEGIS for dominant/robust strategy.
    ∃u∀w . (x(u, w, t), u, w) ⊢ φ
    """
    # Create player for sys and env resp.
    p1 = player(g, is_sys=True)
    p2 = player(magnum.game.invert_game(g), is_sys=False)

    # Start Co-Routines
    next(p1)
    next(p2)

    # Take turns providing counter examples
    response = set()
    for p in cycle([p1, p2]):
        # Tell p about previous response and get p's response.
        response = p.send(response)
        yield response  # generator for monitoring/testing

        # p failed to respond, thus the game ends.
        if not response.feasible:
            return None if p == p1 else response.solution


def player(g, is_sys=True):
    """Player co-routine.
    Receives counter example and then returns response. Remembers
    previous inputs that the adv responded to and won't play them
    again.
    """
    inputs, adv_inputs = g.model.vars.input, g.model.vars.env
    counter_example = yield

    # We must be the first player. Give unconstrained response.
    if not counter_example:
        counter_example = yield predict(g)
    banned_inputs = set()

    learned_lens = lens(g).spec.learned
    while True:
        # They gave a response w, we cannot use previous solutions.
        sol = counter_example.solution
        prev_input = project_solution_stl(sol, inputs, g.model.t)
        response = project_solution_stl(sol, adv_inputs, g.model.t)
        # Step 1) prev input had counter strategy, so ban it.
        if prev_input is not stl.TOP:
            banned_inputs.add(prev_input)

        # Step 2) respond to w's response.
        learned = response & ~stl.orf(*banned_inputs)
        prediction = predict(learned_lens.set(learned))

        # Step 3) If not the system, need to consider old inputs
        # upon failure
        # TODO: bench mark if it's better to relax problem
        # or constrained problem
        if not prediction.feasible and not is_sys:
            learned = response & stl.orf(*banned_inputs)
            prediction = predict(learned_lens.set(learned))

        # Step 4) Yield response
        counter_example = yield prediction
