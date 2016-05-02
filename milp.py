from __future__ import division

from math import ceil

from singledispatch import singledispatch

import stl
# import gurobipy as gpy

def get_vars(phi, model, steps):
    """phi -> num continuous, num boolean"""
    var_map = {}
    for i, x in enumerate(stl.walk(phi)):
        vtype = "binary" if isinstance(x, stl.Pred) else "continous"
        var_map[x] = [(vtype, "TODO")] * steps
    return var_map

@singledispatch
def encode(problem):
    """STL -> MILP"""
    env = reduce(stl.And, problem['env'])
    sys = reduce(stl.And, problem['sys'])
    phi = stl.Or(stl.Neg(env), sys)
    steps = int(ceil(problem['params']['time_horizon'] / problem['params']['dt']))

    model = None
    var_map = get_vars(phi, model, steps)
    for psi in stl.walk(phi):
        print(psi)
        encode(psi, model, var_map, 0)

@encode.register(stl.Pred)
def _(psi, model, params, t):
    # mu(x_t) <= M_t z_t^u - eps_t
    # -mu(x_t) <= M_t(1 - z_t^u) - eps_t
    # z_t in {0, 1}
    raise NotImplementedError


@encode.register(stl.Or)
def _(psi, model, var_map, t):
    # psi(t) = or(phi_1(t), ..., phi_m(t))
    # z_psi(t) >= z_phi_i(t) forall i
    # z_psi(t) <= sum(z_phi_i(t), i=1, m)
    # z_psi(t) in [0, 1]
    raise NotImplementedError


@encode.register(stl.And)
def _(psi, model, var_map, t):
    # psi(t) = and(phi_1(t), ..., phi_m(t))
    # z_psi(t) <= z_phi_i(t) forall i
    # z_psi(t) <= 1 - m + sum(z_phi_i(t), i=1, m)
    raise NotImplementedError


@encode.register(stl.F)
def _(psi, model, var_map, t):
    # psi = F[a,b] phi
    # a(t, N) = min(t + a, N)
    # b(t, N) = min(t + b, N)
    # z_psi(t') = or(z_psi(t=i) for i=a(t, N) to b(t, N))
    raise NotImplementedError


@encode.register(stl.G)
def _(psi, model, var_map, t):
    # psi = F[a,b] phi
    # a(t, N) = min(t + a, N)
    # b(t, N) = min(t + b, N)
    # z_psi(t') = and(z_psi(t=i) for i=a(t, N) to b(t, N))
    raise NotImplementedError


@encode.register(stl.FG)
def _(psi, model, var_map, t):
    raise NotImplementedError


@encode.register(stl.GF)
def _(psi, model, var_map, t):
    raise NotImplementedError


@encode.register(stl.Neg)
def _(psi, model, var_map, t):
    raise NotImplementedError
