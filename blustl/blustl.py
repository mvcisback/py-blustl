"""
"""
from functools import partial
from copy import deepcopy

from numpy import hstack

import stl
import milp
from constraint_kinds import Kind as K

oo = float('inf')

def p2_params(params):
    params2 = deepcopy(params)
    n_sys = params2['num_sys_inputs'] = params['num_env_inputs']
    B = params['state_space']['B']
    params2['state_space']['B'] = hstack([B[:, n_sys:], B[:, :n_sys]])
    params2['sys'] = [stl.Neg(stl.And(tuple(params['sys'])))]
    return params2


def controller_oracle(params):
    J0, u2, u1, iis0 = best_response(params, w=None)
    if J0 != -oo and params['num_env_inputs'] > 0: # adversarial
        params1 = params
        params2 = p2_params(params)
        
        p1 = cegis(params1, u2)
        p2 = cegis(params2, u1)
        (J1, w1, u1, iis1) = next(p1)
        (J2, w2, u2, iis2) = next(p2)

        env_repeated = sys_repeated = False        
        while not (env_repeated and sys_repeated):
            env_repeated = sys_repeated = False
            if J1 == -oo:
                return False, w1, iis1 # iis
            elif J2 == -oo:
                return True, w2, iis2 # iis, Solved

            try:
                (J1, w1, u1, iis1) = p1.send(u2)
            except StopIteration:
                env_repeated = True
            try:
                (J2, w2, u2, iis2) = p2.send(u1)
            except StopIteration:
                sys_repeated = True
        return True, u2, u1, None
    else:
        return (J0 != -oo), u2, u1, iis0


def cegis(params, w0):
    ws = {w0}
    costs = {}
    while True:
        responses = [best_response(params, w=w) for w in ws]
        
        J, w, u, iis = min(responses)
        w = yield J, w, u, iis
        if w in ws:
            break

        costs[u] = J
        ws.add(w)


# TODO: memomize
def best_response(params, w=None):
    feasible, output = milp.encode_and_run(params, w=w)
    if not feasible:
        iis = output
        return -oo, w, None, iis
    else:
        J, inputs = output
        return J, inputs['w'], inputs['u'], None


def learn_spec(params):
    r = repair_oracle(params)
    next(r)
    # TODO termination condition
    while True:
        success, u, iis = get_controller(params)
        if success:
            return params, u 
        r.send(iis)
        params = next(r)
