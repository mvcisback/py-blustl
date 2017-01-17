from itertools import cycle
from collections import namedtuple

import funcy as fn

from blustl.mpc.non_adversarial import predict
from blustl.utils import to_lineq

def cegis(phi, g, t, *, p1:bool):
    banned_ws, banned_us = set(), set()

    turns = cycle(
        [(True, banned_us, phi, g.model.vars.inputs),
         (False, banned_ws, ~phi, g.model.vars.env)]
    )

    for p1_turn, banned_inputs, banned_adv_inputs, spec, input_symbols in turns:
        psi = ~stl.orf(*banned_inputs)
        success, inputs = best_response(psi, g, t, input_symbols)
        
        if success:
            banned_adv_inputs.add(inputs)
        elif not p1_turn:
            # TODO: return Maybe inputs
            raise Exception("No Solution Exists")
        else:
            # TODO: pass back learned lemmas that are relevant to next MPC loop
            return inputs


def best_response(psi, g, t, input_symbols):
    res = predict(psi, g, t)
    inputs = fn.project(res.solution.get(t, {}), input_symbols)
    return res.feasible, to_lineq(inputs)
