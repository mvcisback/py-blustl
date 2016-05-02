from __future__ import division

from math import ceil
from itertools import product

from singledispatch import singledispatch
import gurobipy as gpy

import stl

M = 100 # TODO
eps = 0.01 # TODO

def get_vars(phi, model, steps, problem):
    var_map = {}
    for i, x in enumerate(stl.walk(phi)):
        vtype = gpy.GRB.BINARY if isinstance(x, stl.Pred) else gpy.GRB.CONTINUOUS
        var_map[x] = [model.addVar(vtype=vtype, name="{}_{}".format(i, t)) for t in range(steps)]

    n = problem['params']['num_vars']
    var_map['x'] = {(i, t): model.addVar(vtype=gpy.GRB.CONTINUOUS, name="x{}_{}".format(i, t)) for i, t in product(range(n), range(steps))}
    model.update()

    return var_map


@singledispatch
def encode(problem):
    """STL -> MILP"""
    env = reduce(stl.And, problem['env'])
    sys = reduce(stl.And, problem['sys'])
    phi = stl.Or(stl.Neg(env), sys)
    H = ceil(problem['params']['time_horizon'])
    dt = problem['params']['dt']
    steps = int(ceil(H / dt))

    model = gpy.Model("milp")
    var_map = get_vars(phi, model, steps, problem)
    
    for psi in stl.walk(phi):
        encode(psi, model, var_map, 0, H, dt)

    import ipdb; ipdb.set_trace()

    # TODO: add state evolution constraints

@encode.register(stl.Pred)
def _(psi, model, var_map, t, H, dt):
    const = psi.const
    x = var_map['x'][psi.lit, t]
    z_t = var_map[psi][t]
    if psi.op in ("<", "<=", "="):
        model.addConstr(const - x <= M*z_t - eps)
        model.addConstr(x - const <= M*(1 - z_t) - eps)

    if psi.op in (">", ">=", "="):
        model.addConstr(x - const <= M*z_t - eps)
        model.addConstr(const - x <= M*(1 - z_t) - eps)


@encode.register(stl.Or)
def _(psi, model, var_map, t, H, dt):
    z_psi = var_map[psi][t]
    z_phiL = var_map[psi.left][t]
    z_phiR = var_map[psi.right][t]
    
    model.addConstr(z_psi >= z_phiL)
    model.addConstr(z_psi >= z_phiR)
    model.addConstr(z_psi <= z_phiL + z_phiR)

@encode.register(stl.And)
def _(psi, model, var_map, t, H, dt):
    z_psi = var_map[psi][t]
    z_phiL = var_map[psi.left][t]
    z_phiR = var_map[psi.right][t]
    
    model.addConstr(z_psi <= z_phiL)
    model.addConstr(z_psi <= z_phiR)
    model.addConstr(z_psi >= z_phiL + z_phiR - 1)


@encode.register(stl.F)
def _(psi, model, var_map, t, H, dt):
    z_psi = var_map[psi][t]
    a, b = psi.interval.lower, psi.interval.upper
    f = lambda x: int(ceil(x / dt))
    a, b = f(max(t + a, H)), f(max(t + b, H))
    
    for i in range(a, b):
        model.addConstr(z_psi >= var_map[psi.arg][i])
    model.addConstr(z_psi <= sum(var_map[psi.arg][i] for i in range(a, b)))


@encode.register(stl.G)
def _(psi, model, var_map, t, H, dt):
    z_psi = var_map[psi][t]
    a, b = psi.interval.lower, psi.interval.upper
    f = lambda x: int(ceil(x / dt))
    a, b = f(max(t + a, H)), f(max(t + b, H))
    
    for i in range(a, b):
        model.addConstr(z_psi <= var_map[psi.arg][i])
    model.addConstr(z_psi >= 1 - (b-a) + sum(var_map[psi.arg][i] for i in range(a, b)))


@encode.register(stl.FG)
def _(psi, model, var_map, t, H, dt):
    raise NotImplementedError


@encode.register(stl.GF)
def _(psi, model, var_map, t, H, dt):
    raise NotImplementedError


@encode.register(stl.Neg)
def _(psi, model, var_map, t, H, dt):
    z_psi = var_map[psi][t]
    z_phi = var_map[psi.arg][t]
    model.addConstr(z_psi == 1 - z_phi)
