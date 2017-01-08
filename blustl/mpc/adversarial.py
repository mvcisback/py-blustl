from itertools import cycle
from collections import namedtuple

import funcy as fn

from blustl.mpc.non_adversarial import predict
from blustl.utils import to_lineq

def cegis(phi, g, t):
    banned_us = set()
    while True:
        psi = ~stl.orf(*banned_us)
        success, inputs = best_response(psi, g, t)
        return  inputs, banned_adv_inputs


def best_response(psi, g, t):
    res = predict(psi, g, t)
    inputs = fn.project(res.solution.get(t, {}), g.model.vars.inputs)
    return res.feasible, to_lineq(inputs)
