# TODO: factor out encode recursive structure into a generator
# TODO: add useful constraint names
# TODO: add tests where variables are preapplied to constraints
# TODO: implement adversarial w
# TODO: move H, steps, dt into problem

from __future__ import division

from math import ceil
from itertools import product, chain, starmap
import operator
from collections import defaultdict, Counter

from singledispatch import singledispatch
from funcy import mapcat, pluck, group_by
import gurobipy as gpy

import stl
from constraint_kinds import Kind as K

M = 10000  # TODO
eps = 0.01  # TODO

class Store(object):
    def __init__(self, problem):
        "docstring"
        self.model = gpy.Model(name=problem.get('name', 'problem'))
        self._z = defaultdict(dict)
        self.u = defaultdict(dict)
        self.w = defaultdict(dict)
        self.x = defaultdict(dict)
        self.constr_lookup = defaultdict(dict)
        self._constr_counter = Counter()
        self.problem = problem

        n = problem['params']['num_vars']
        n_sys = problem['params']['num_sys_inputs']
        n_env = problem['params']['num_env_inputs']

        # Add state, input, and env vars
        elems = [('x', n, self.x), ('u', n_sys, self.u), ('w', n_env, self.w)]
        for pre, num, d in elems:
            for i, t in product(range(num), range(self.steps)):
                name = "{}{}_{}".format(pre, i, t)
                d[i][t] = self.model.addVar(vtype=gpy.GRB.CONTINUOUS,
                                            name=name)

        self.model.update()

    def z(self, x, t):
        if x in self._z and t in self._z[x]:
            return self._z[x][t]

        vtype = gpy.GRB.BINARY if isinstance(x,
                                             stl.Pred) else gpy.GRB.CONTINUOUS
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

    def add_constr(self, constr, phi=None, kind=None):
        self._constr_counter.update([kind])
        name = "{}{}".format(kind.name, self._constr_counter[kind])
        r = self.model.addConstr(constr, name=name)
        self.model.update()
        self.constr_lookup[r.ConstrName] = (phi, kind)


def encode_state_evolution(store, problem):
    inputs = lambda t: chain(pluck(t, store.u.values()), pluck(t, store.w.values()))
    state = lambda t: pluck(t, store.x.values())
    dot = lambda x, y: sum(starmap(operator.mul, zip(x, y)))
    A, B = problem['state_space']['A'], problem['state_space']['B']
    for t in range(store.steps - 1):
        for i, (A_i, B_i) in enumerate(zip(A, B)):
            constr = store.x[i][t + 1] == dot(A_i, state(t)) + store.dt*dot(B_i, inputs(t))
            store.add_constr(constr, kind=K.DYNAMICS)

def encode_input_constr(store, env=False):
    # add input bounds u in [0, 1]
    inputs = store.w if env else store.u
    k1 = K.ENV_INPUT_UPPER if env else K.SYS_INPUT_UPPER
    k2 = K.ENV_INPUT_LOWER if env else K.SYS_INPUT_LOWER
    for u in mapcat(dict.values, inputs.values()):
        store.add_constr(u <= 1, kind=k1)
        store.add_constr(u >= 0, kind=k2)


@singledispatch
def encode(problem):
    """STL -> MILP"""

    sys = reduce(stl.And, problem['sys'])
    env = reduce(stl.And, problem['env'], [])
    phi = stl.Or(stl.Neg(env), sys) if env else sys
    store = Store(problem)

    # encode STL constraints
    encode(phi, 0, store)

    encode_input_constr(store)
    encode_input_constr(store, env=True)

    encode_state_evolution(store, problem)

    for psi in problem['init']:
        x = store.x[psi.lit][0]
        const = psi.const
        store.add_constr(x == const, kind=K.INIT)

    # Assert top level true
    store.add_constr(store.z(phi, 0) == 1, kind=K.ASSERT_FEASIBLE)

    # Create Objective
    stl_vars = mapcat(dict.values, store._z.values())
    # TODO: support alternative objective functions
    
    store.model.setObjective(sum(stl_vars), gpy.GRB.MAXIMIZE)

    store.model.update()
    return store.model, store


@encode.register(stl.Pred)
def _(psi, t, store):
    const = psi.const
    x = store.x[psi.lit][t]
    z_t = store.z(psi, t)

    # TODO combine
    if psi.op in ("<", "<=", "="):
        store.add_constr(const - x <= M * z_t - eps, phi=psi, kind=K.PRED_UPPER)
        store.add_constr(x - const <= M * (1 - z_t) - eps, phi=psi, kind=K.PRED_LOWER)

    if psi.op in (">", ">=", "="):
        store.add_constr(x - const <= M * z_t - eps, phi=psi, kind=K.PRED_UPPER)
        store.add_constr(const - x <= M * (1 - z_t) - eps, phi=psi, kind=K.PRED_LOWER)


@encode.register(stl.Or)
def _(psi, t, store):
    encode_bool_op(psi, t, store, (K.OR, K.OR_TOTAL), False)


@encode.register(stl.And)
def _(psi, t, store):
    encode_bool_op(psi, t, store, (K.AND, K.AND_TOTAL), True)


def encode_bool_op(psi, t, store, kind, or_flag):
    z_psi = store.z(psi, t)
    elems = [store.z(psi.left, t), store.z(psi.right, t)]
    encode_op(z_psi, elems, model, psi, kind, or_flag=or_flag)


@encode.register(stl.F)
def _(psi, t, store):
    encode_temp_op(psi, t, store, (K.F, K.F_TOTAL), or_flag=True)


@encode.register(stl.G)
def _(psi, t, store):
    encode_temp_op(psi, t, store, (K.G, K.G_TOTAL), or_flag=False)


def encode_temp_op(psi, t, store, kind, or_flag=False):
    z_psi = store.z(psi, t)
    a, b = psi.interval.lower, psi.interval.upper
    f = lambda x: int(ceil(x / store.dt))
    H = store.H
    a, b = f(min(t + a, H)), f(min(t + b, H))

    elems = [store.z(psi.arg, t + i) for i in range(a, b + 1)]
    encode_op(z_psi, elems, store, kind, psi, or_flag=or_flag)


def encode_op(z_psi, elems, store, (k1, k2), phi, or_flag=False):
    z_phi_total = sum(elems)

    if or_flag:
        rel = operator.ge
        lhs = z_phi_total
    else:  # AND
        rel = operator.le
        lhs = 1 - len(elems) + z_phi_total

    for e in elems:
        store.add_constr(rel(z_psi, e), phi, kind=k1)
    store.add_constr(rel(lhs, z_psi), phi, kind=k2)


@encode.register(stl.Neg)
def _(psi, t, store):
    z_psi, z_phi = store.z(psi, t), store.z(psi.arg, t)
    store.add_constr(z_psi == 1 - z_phi, psi, kind=K.Neg)


def encode_and_run(problem):
    model, store = encode(problem)
    model.optimize()

    if model.status == gpy.GRB.Status.INF_OR_UNBD:
        # Turn presolve off to determine whether model is infeasible
        # or unbounded
        model.setParam(gpy.GRB.Param.Presolve, 0)
        model.optimize()

    if model.status == gpy.GRB.Status.INFEASIBLE:
        model.computeIIS()
        IIS = [store.constr_lookup[x.ConstrName] for x in model.getConstrs() if x.IISConstr]
        return (False, IIS)

    elif model.status == gpy.GRB.Status.OPTIMAL:
        f = lambda x: x[0][0]
        solution = group_by(f, [(x.VarName, x.X) for x in model.getVars()])
        return (True, pluck(1, sorted(solution['u'])))
    else:
        raise NotImplementedError
