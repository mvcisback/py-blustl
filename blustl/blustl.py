"""
"""
from functools import partial

from numpy import hstack

import stl
import milp
from constraint_kinds import Kind as K

oo = float('inf')

def p2_params(params):
    params2 = params.copy()    
    n_sys = params2['num_sys_inputs'] = params['num_env_inputs']
    B = params['state_space']['B']
    params2['state_space']['B'] = hstack([B[:, n_sys:], B[:, :n_sys]])
    params2['sys'] = stl.Neg(params['sys'])
    return params2


def controller_oracle(params):
    if params['num_env_inputs'] > 0: # adversarial
        params1 = params
        params2 = p2_params(params)
        
        p1 = cegis(params1)
        p2 = cegis(params2)
        while True:
            (J1, w1, iis_or_u1), (J2, w2, iis_or_u2) = next(p1), next(p2)
            if J1 == -oo:
                return False, w1, iis_or_u1 # iis
            elif J2 == -oo:
                return True, w2, iis_or_u2 # iis, Solved
            p1.send(iis_or_u2)
            p2.send(iis_or_u1)

    else:
        J, w, iis = best_response(params, w=None)
        return (J != -oo), w, iis


def cegis(params):
    ws = {[0]*params['num_env_inputs']}
    costs = {}
    while True:
        responses = map(partial(best_response, partial), ws)
        J, w, iis_or_u = min(responses)
        yield J, w, iis_or_u

        if J == -oo:
            break # stop

        cost[u] = iis_or_u # u
        w = yield
        ws.add(w)


# TODO: memomize
def best_response(params, w):
    feasible, output = milp.encode_and_run(params, w=w)
    if not feasible:
        iis = output
        return -oo, w, iis
    else:
        J, u = output
        return J, w, u


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
