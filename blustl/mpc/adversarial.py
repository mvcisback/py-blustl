from itertools import cycle
from collections import namedtuple

import funcy as fn

from blustl.mpc.non_adversarial import predict
from blustl.utils import project_solution_stl

def cegis(phi, g, t):
    """CEGIS for dominate strategy.
    ∃u∀w . (x(u, w, t), u, w) ⊢ φ
    """
    # Create player for sys and env resp.
    p1 = player(phi, g, t, g.model.vars.inputs, g.model.vars.env)
    p2 = player(~phi, g, t, g.model.vars.env, g.model.vars.inputs)

    # Start Co-Routines
    next(p1); next(p2)

    # Take turns providing counter examples
    response = set()
    for p in cycle([p1, p2]):
        # Tell p about previous response and get p's response.
        response = p.send(response)

        # p failed to respond, thus the game ends.
        if not respond.feasible:
            return None if p == p1 else response.solution


def player(phi, g, t, inputs, adv_inputs):
    """Player co-routine. 
    Receives counter example and then returns response. Remembers
    previous inputs that the adv responded to and won't play them
    again.
    """
    counter_example = yield

    # We must be the first player. Give unconstrained response.
    if not counter_example:
        counter_example = yield predict(phi, g, t)

    while True:
        # They gave a response w, we cannot use previous solutions.
        sol = counter_example.solution
        prev_input = project_solution_stl(sol, inputs)
        response = project_solution_stl(sol, adv_inputs)

        # Step 1) prev input had counter strategy, so ban it.
        phi &= ~prev_input

        # Step 2) respond to w's response.
        counter_example = yield predict(psi & response, g, t)
