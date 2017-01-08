from itertools import cycle
from collections import namedtuple

import funcy as fn

from blustl.mpc.non_adversarial import predict
from blustl.utils import to_lineq

def cegis(phi, g, t):
    banned_us = set()
    input_symbols = g.model.vars.inputs
    while True:
        psi = ~stl.orf(*banned_us)
        success, inputs = best_response(psi, g, t, input_symbols)
        return  inputs, banned_adv_inputs


def best_response(psi, g, t, input_symbols):
    res = predict(psi, g, t)
    inputs = fn.project(res.solution.get(t, {}), input_symbols)
    return res.feasible, to_lineq(inputs)
