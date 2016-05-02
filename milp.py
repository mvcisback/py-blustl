from singledispatch import singledispatch

import stl
# import gurobipy as gpy

# http://www.gurobi.com/documentation/6.5/examples/mip1_py.html
# https://pythonhosted.org/PuLP/


@singledispatch
def encode(problem):
    """STL -> MILP"""
    m = gpy.Model("mip1")

    # TODO: do pass to get which variables
    raise NotImplementedError


@encode.register(stl.Pred)
def _(psi, model, params):
    # mu(x_t) <= M_t z_t^u - eps_t
    # -mu(x_t) <= M_t(1 - z_t^u) - eps_t
    # z_t in {0, 1}


@encode.register(stl.Or)
def _(psi, model, params):
    # psi(t) = or(phi_1(t), ..., phi_m(t))
    # z_psi(t) >= z_phi_i(t) forall i
    # z_psi(t) <= sum(z_phi_i(t), i=1, m)
    # z_psi(t) in [0, 1]
    raise NotImplementedError


@encode.register(stl.And)
def _(psi, model, params):
    # psi(t) = and(phi_1(t), ..., phi_m(t))
    # z_psi(t) <= z_phi_i(t) forall i
    # z_psi(t) <= 1 - m + sum(z_phi_i(t), i=1, m)
    raise NotImplementedError


@encode.register(stl.F)
def _(psi, model, params):
    # psi = F[a,b] phi
    # a(t, N) = min(t + a, N)
    # b(t, N) = min(t + b, N)
    # z_psi(t') = or(z_psi(t=i) for i=a(t, N) to b(t, N))
    raise NotImplementedError


@encode.register(stl.G)
def _(psi, model, params):
    # psi = F[a,b] phi
    # a(t, N) = min(t + a, N)
    # b(t, N) = min(t + b, N)
    # z_psi(t') = and(z_psi(t=i) for i=a(t, N) to b(t, N))
    raise NotImplementedError


@encode.register(stl.FG)
def _(psi, params):
    raise NotImplementedError


@encode.register(stl.GF)
def _(psi, params):
    raise NotImplementedError
