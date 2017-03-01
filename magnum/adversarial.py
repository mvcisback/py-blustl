from itertools import cycle
from collections import namedtuple

import stl
from stl import orf

from magnum.non_adversarial import predict
from magnum.utils import project_solution_stl


def cegis(phi, g, t):
    """CEGIS for dominant/robust strategy.
    ∃u∀w . (x(u, w, t), u, w) ⊢ φ
    """
    # Create player for sys and env resp.
    p1 = player(phi, g, t, g.model.vars.input, g.model.vars.env)
    p2 = player(~phi, g, t, g.model.vars.env, g.model.vars.input,
                is_sys=False)

    # Start Co-Routines
    next(p1)
    next(p2)

    # Take turns providing counter examples
    response = set()
    for p in cycle([p1, p2]):
        # Tell p about previous response and get p's response.
        response = p.send(response)

        # p failed to respond, thus the game ends.
        if not response.feasible:
            return None if p == p1 else response.solution


def player(phi, g, t, inputs, adv_inputs, is_sys=True):
    """Player co-routine.
    Receives counter example and then returns response. Remembers
    previous inputs that the adv responded to and won't play them
    again.
    """
    counter_example = yield

    # We must be the first player. Give unconstrained response.
    if not counter_example:
        counter_example = yield predict(phi, g, t)
    banned_inputs = set()

    while True:
        # They gave a response w, we cannot use previous solutions.
        sol = counter_example.solution
        prev_input = project_solution_stl(sol, inputs, t)
        response = project_solution_stl(sol, adv_inputs, t)
        # Step 1) prev input had counter strategy, so ban it.
        if prev_input is not stl.TOP:
            banned_inputs.add(prev_input)

        # Step 2) respond to w's response.
        psi = phi & response
        if banned_inputs:
            psi &= ~orf(*banned_inputs)

        prediction = predict(psi, g, t)

        # Step 3) If not the system, need to consider old inputs
        # upon failure
        if not prediction.feasible and not is_sys:
            prediction = predict(phi & response, g, t)

        # Step 4) Yield response
        counter_example = yield prediction
