from itertools import cycle
from collections import namedtuple

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
    phi1 = magnum.game_to_stl(g)
    phi2 = magnum.game_to_stl(g, invert_game=True)
    p1 = player(phi1, g, g.model.vars.input, g.model.vars.env)
    p2 = player(phi2, g, g.model.vars.env, g.model.vars.input,
                is_sys=False)

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


def player(phi, g, inputs, adv_inputs, is_sys=True):
    """Player co-routine.
    Receives counter example and then returns response. Remembers
    previous inputs that the adv responded to and won't play them
    again.
    """
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
        learned = response
        if banned_inputs:
            learned &= ~stl.orf(*banned_inputs)

        prediction = predict(learned_lens.set(learned))

        # Step 3) If not the system, need to consider old inputs
        # upon failure
        if not prediction.feasible and not is_sys:
            prediction = predict(learned_lens.set(response))

        # Step 4) Yield response
        counter_example = yield prediction
