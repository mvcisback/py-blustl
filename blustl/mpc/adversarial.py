from itertools import cycle
from collections import namedtuple

import funcy as fn

from blustl.mpc.non_adversarial import predict
from blustl.utils import to_lineq

def cegis(phi, g, t, *, p1:bool):
    banned_ws, banned_us = set(), set()
    turns = [(True, banned_us, phi, g.model.vars.inputs),
             (False, banned_ws, ~phi, g.model.vars.env)]
    for p1_turn, banned_inputs, banned_adv_inputs, spec, input_symbols in turns:
        psi = stl.andf(*banned_inputs)
        success, inputs = best_response(psi, g, t, input_symbols)
        
        if success:
            banned_adv_inputs.add(inputs)
        else:
            return p1_turn, inputs, banned_adv_inputs


def best_response(psi, g, t, input_symbols):
    res = predict(psi, g, t)
    inputs = fn.project(res.solution.get(t, {}), input_symbols)
    return res.feasible, to_lineq(inputs)
