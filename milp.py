from __future__ import division

from math import ceil
from itertools import product, chain, starmap
from operator import mul
import operator
from collections import defaultdict
from functools import partial

from singledispatch import singledispatch
from funcy import pairwise, mapcat
import gurobipy as gpy

import stl

M = 10000 # TODO
eps = 0.01 # TODO

# TODO make class for tracking current variables
# - implement get_var as part of accessor
# - automatically encode unseen variables

class Store(object):
    def __init__(self, problem):
        "docstring"
        self.model = gpy.Model(name=problem.get('name', 'problem'))
        self._z = defaultdict(dict)
        self.u = defaultdict(dict)
        self.w = defaultdict(dict)
        self.x = defaultdict(dict)
        self.problem = problem

        n = problem['params']['num_vars']
        n_sys = problem['params']['num_sys_inputs']
        n_env = problem['params']['num_env_inputs']

        # Add state, input, and env vars
        elems = [('x', n, self.x), ('u', n_sys, self.u), ('w', n_env, self.w)]
        for pre, num, d in elems:
            for i, t in product(range(num), range(self.steps)):
                name ="{}{}_{}".format(pre, i, t)
                d[i][t] = self.model.addVar(vtype=gpy.GRB.CONTINUOUS, name=name)

        self.model.update()

    def z(self, x, t):
        if x in self._z and t in self._z[x]:
            return self._z[x][t]

        vtype = gpy.GRB.BINARY if isinstance(x, stl.Pred) else gpy.GRB.CONTINUOUS
        prefix = "z" if isinstance(x, stl.Pred) else "Z"
        i = len(self._z)
        name = "{}{}_{}".format(prefix, i, t)
        self._z[x][t] = self.model.addVar(vtype=vtype, name=name)
        self.model.update()
        encode(x, t, self)
        return self._z[x][t]

    @property
    def H(self):
        return ceil(self.problem['params']['time_horizon'])

    @property
    def dt(self):
        return self.problem['params']['dt']

    @property
    def steps(self):
        return int(ceil(self.H / self.dt))


@singledispatch
def encode(problem):
    """STL -> MILP"""

    sys = reduce(stl.And, problem['sys'])
    env = reduce(stl.And, problem['env'], [])
    phi = stl.Or(stl.Neg(env), sys) if env else sys
    store = Store(problem)

    # encode STL constraints
    encode(phi, 0, store)

    # add input bounds u in [0, 1]
    for u in mapcat(dict.values, chain(store.u.values(), store.w.values())):
        store.model.addConstr(u <= 1)
        store.model.addConstr(u >= 0)

    # TODO: Clean up system evolution
    n = problem['params']['num_vars']
    n_sys = problem['params']['num_sys_inputs']
    n_env = problem['params']['num_env_inputs']
    inputs = lambda t: [store.u[i][t] for i in range(n_sys)] + [store.w[i][t] for i in range(n_env)]
    state = lambda t: [store.x[i][t] for i in range(n)]
    dot = lambda x, y: sum(starmap(mul, zip(x, y)))
    A, B = problem['state_space']['A'], problem['state_space']['B']
    for t in range(store.steps-1):
        for i, (A_i, B_i) in enumerate(zip(A, B)):
            store.model.addConstr(store.x[i][t+1] == dot(A_i, state(t)) + dot(B_i, inputs(t)))

    for psi in problem['init']:
        x = store.x[psi.lit][0]
        const = psi.const
        store.model.addConstr(x == const)

    # Assert top level true
    store.model.addConstr(store.z(phi, 0) == 1)

    stl_vars =  mapcat(lambda phi: store.x[phi].values(), stl.walk(phi))
    store.model.setObjective(sum(stl_vars), gpy.GRB.MAXIMIZE)
    store.model.update()
    store.model.optimize()

    if store.model.status == gpy.GRB.Status.INF_OR_UNBD:
        # Turn presolve off to determine whether store.model is infeasible
        # or unbounded
        store.model.setParam(gpy.GRB.Param.Presolve, 0)
        store.model.optimize()

    if store.model.status == gpy.GRB.Status.OPTIMAL:
        # TODO: return controller
        raise NotImplementedError
        return

    elif store.model.status != gpy.GRB.Status.INFEASIBLE:
        # TODO: return IIS
        raise NotImplementedError
        return


@encode.register(stl.Pred)
def _(psi, t, store):
    const = psi.const
    x = store.x[psi.lit][t]
    z_t = store.z(psi, t)

    if psi.op in ("<", "<=", "="):
        store.model.addConstr(const - x <= M*z_t - eps)
        store.model.addConstr(x - const <= M*(1 - z_t) - eps)

    if psi.op in (">", ">=", "="):
        store.model.addConstr(x - const <= M*z_t - eps)
        store.model.addConstr(const - x <= M*(1 - z_t) - eps)


@encode.register(stl.Or)
def _(psi, t, store):
    encode_bool_op(psi, t, store, False)


@encode.register(stl.And)
def _(psi, t, store):
    encode_bool_op(psi, t, store, True)


def encode_bool_op(psi, t, store, or_flag):
    z_psi = store.z(psi, t)
    elems = [store.z(psi.left, t), store.z(psi.right, t)]
    encode_op(z_psi, elems, model, or_flag=or_flag)


@encode.register(stl.F)
def _(psi, t, store):
    encode_temp_op(psi, t, store, or_flag=True)


@encode.register(stl.G)
def _(psi, t, store):
    encode_temp_op(psi, t, store, or_flag=False)


def encode_temp_op(psi, t, store, or_flag=False):
    z_psi = store.z(psi, t)
    a, b = psi.interval.lower, psi.interval.upper
    f = lambda x: int(ceil(x / store.dt))
    H = store.H
    a, b = f(min(t + a, H)), f(min(t + b, H))

    elems = [store.z(psi.arg, t + i) for i in range(a, b + 1)]
    encode_op(z_psi, elems, store.model, or_flag=or_flag)


def encode_op(z_psi, elems, model, or_flag=False):
    z_phi_total = sum(elems)

    if or_flag:
        rel = operator.ge
        lhs = z_phi_total
    else: # AND
        rel = operator.le
        lhs = 1 - len(elems) + z_phi_total

    for e in elems:
        model.addConstr(rel(z_psi, e))
    model.addConstr(rel(lhs, z_psi))


@encode.register(stl.Neg)
def _(psi, t, store):
    z_psi = store.z(psi, t)
    z_phi = store.z(psi.arg, t)
    model.addConstr(z_psi == 1 - z_phi)

    encode(psi.arg, t, store)


# TODO
@encode.register(stl.FG)
def _(psi, model, var_map, t, H, dt):
    raise NotImplementedError


# TODO
@encode.register(stl.GF)
def _(psi, model, var_map, t, H, dt):
    raise NotImplementedError
