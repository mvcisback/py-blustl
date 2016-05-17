# TODO: factor out encode recursive structure into a generator
# TODO: add tests where variables are preapplied to constraints
# TODO: Compute eps and M based on x and A, B, dt
# TODO: encode STL robustness metric
# TODO: make inital conditions part of phi
# TODO: convert to PuLP: http://pythonhosted.org/PuLP/CaseStudies/a_blending_problem.html

from __future__ import division

from math import ceil
from itertools import product, chain, starmap
import operator
from collections import defaultdict, Counter
from functools import partial, singledispatch
from funcy import mapcat, pluck, group_by, drop, walk_values, compose

from blustl import stl
from blustl.constraint_kinds import Kind as K

DEFAULT_NAME = 'controller_synth'


class Store(object):
    def __init__(self, params, x=None, u=None, w=None):
        self.model = gpy.Model(name=params.get('name', DEFAULT_NAME))
        self._z = defaultdict(dict)
        self.u = defaultdict(dict)
        self.w = defaultdict(dict)
        self.x = defaultdict(dict)
        self.constr_lookup = defaultdict(dict)
        self._constr_counter = Counter()
        self.params = params
        self._count = 0

        n = params['num_vars']
        n_sys = params['num_sys_inputs']
        n_env = params['num_env_inputs']

        # Add state, input, and env vars
        elems = [('x', n, self.x, x), ('u', n_sys, self.u, u), 
                 ('w', n_env, self.w, w)]

        for pre, num, d, fixed in elems:
            # TODO: support taking bounds from phi/params
            d2 = dict(fixed) if fixed else {}
            ub, lb = (100, -100) if pre == 'x' else (1, 0)
            for i, t in product(range(num), range(self.steps + 1)):
                if (i, t) in d2:
                    d[i][t] = d2[i, t]
                else:
                    name = "{}{}_{}".format(pre, i, t)
                    d[i][t] = self.model.addVar(vtype=gpy.GRB.CONTINUOUS,
                                                name=name, lb=lb, ub=ub)

        self.model.update()

    def z(self, x, t):
        if x in self._z and t in self._z[x]:
            return self._z[x][t]

        vtype = gpy.GRB.BINARY if isinstance(x,
                                             stl.Pred) else gpy.GRB.CONTINUOUS
        prefix = "z" if isinstance(x, stl.Pred) else "q"
        i = self._count
        self._count += 1
        name = "{}{}_{}".format(prefix, i, t)
        self._z[x][t] = self.model.addVar(vtype=vtype, name=name)
        self.model.update()
        encode(x, t, self)
        return self._z[x][t]

    @property
    def H(self):
        return ceil(self.params['time_horizon'])

    @property
    def dt(self):
        return self.params['dt']

    @property
    def steps(self):
        return int(ceil(self.H / self.dt))

    def add_constr(self, constr, phi=None, kind=None):
        if not isinstance(constr, gpy.TempConstr):
            return  # only add if symbolic

        self._constr_counter.update([kind])
        name = "{}{}".format(kind.name, self._constr_counter[kind])
        r = self.model.addConstr(constr, name=name)
        self.model.update()
        self.constr_lookup[r.ConstrName] = (phi, kind)


def encode_state_evolution(store, params):
    inputs = lambda t: chain(pluck(t, store.u.values()), pluck(t, store.w.values()))
    state = lambda t: pluck(t, store.x.values())
    dot = lambda x, y: sum(starmap(operator.mul, zip(x, y)))
    A, B = params['state_space']['A'], params['state_space']['B']
    for t in range(store.steps):
        for i, (A_i, B_i) in enumerate(zip(A, B)):
            dx = store.dt*(dot(A_i, state(t)) + dot(B_i, inputs(t)))
            constr = store.x[i][t + 1] == store.x[i][t] + dx
            store.add_constr(constr, kind=K.DYNAMICS)


@singledispatch
def encode(params, x=None, u=None, w=None, p1=True):
    """STL -> MILP"""

    if p1:
        phi = stl.Or(tuple(params['sys'] + stl.Neg(params.get('env', []))))
    else:
        phi = stl.And(tuple(stl.Neg(params['sys']) + params.get('env', [])))
        
    store = Store(params, x=x, u=u, w=w)

    # encode STL constraints
    encode(phi, 0, store)
    encode_state_evolution(store, params)

    for psi in params['init']:
        x = store.x[psi.lit][0]
        const = psi.const
        store.add_constr(x == const, kind=K.INIT)

    # Assert top level true
    store.add_constr(store.z(phi, 0) == 1, kind=K.ASSERT_FEASIBLE)

    # Create Objective
    # TODO: support alternative objective functions
    store.model.setObjective(0, gpy.GRB.MAXIMIZE)

    store.model.update()
    return store.model, store


@encode.register(stl.Pred)
def _(psi, t, store):
    const = psi.const

    x = store.x[psi.lit][t]
    z_t = store.z(psi, t)

    M = 1000  # TODO
    # TODO: come up w. better value for eps
    eps = 0.01 if psi.op == "=" else 0

    # TODO combine
    if psi.op in ("<", "<=", "="):
        store.add_constr(const - x <= M * z_t - eps,
                         phi=psi,
                         kind=K.PRED_UPPER)
        store.add_constr(x - const <= M * (1 - z_t) - eps,
                         phi=psi,
                         kind=K.PRED_LOWER)

    elif psi.op in (">", ">=", "="):
        store.add_constr(x - const <= M * z_t - eps,
                         phi=psi,
                         kind=K.PRED_UPPER)
        store.add_constr(const - x <= M * (1 - z_t) - eps,
                         phi=psi,
                         kind=K.PRED_LOWER)
        


@encode.register(stl.Or)
def _(psi, t, store):
    encode_bool_op(psi, t, store, (K.OR, K.OR_TOTAL), or_flag=True)


@encode.register(stl.And)
def _(psi, t, store):
    encode_bool_op(psi, t, store, (K.AND, K.AND_TOTAL), or_flag=False)


def encode_bool_op(psi, t, store, kind, or_flag):
    z_psi = store.z(psi, t)
    elems = [store.z(psi2, t) for psi2 in psi.args]
    encode_op(z_psi, elems, store, kind, psi, or_flag=or_flag)


@encode.register(stl.F)
def _(psi, t, store):
    encode_temp_op(psi, t, store, (K.F, K.F_TOTAL), or_flag=True)


@encode.register(stl.G)
def _(psi, t, store):
    encode_temp_op(psi, t, store, (K.G, K.G_TOTAL), or_flag=False)


def encode_temp_op(psi, t, store, kind, or_flag=False):
    z_psi = store.z(psi, t)
    f = lambda x: int(ceil(x / store.dt))
    a, b = f(psi.interval.lower), f(psi.interval.upper)
    elems = [store.z(psi.arg, t + i) for i in range(a, b + 1)
             if t + i <= store.steps]

    encode_op(z_psi, elems, store, kind, psi, or_flag=or_flag)


def encode_op(z_psi, elems, store, kind, phi, or_flag=False):
    k1, k2 = kind
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
    store.add_constr(z_psi == 1 - z_phi, psi, kind=K.NEG)


def encode_and_run(params, x=None, u=None, w=None):
    model, store = encode(params, x=x, u=u, w=w)
    model.optimize()

    if model.status == gpy.GRB.Status.INF_OR_UNBD:
        # Turn presolve off to determine whether model is infeasible
        # or unbounded
        model.setParam(gpy.GRB.Param.Presolve, 0)
        model.optimize()

    if model.status == gpy.GRB.Status.INFEASIBLE:
        model.computeIIS()
        IIS = [store.constr_lookup[x.ConstrName]
               for x in model.getConstrs() if x.IISConstr]
        return (False, IIS)

    elif model.status == gpy.GRB.Status.OPTIMAL:
        f = lambda x: x[0][0]
        f2 = lambda x: (tuple(map(int, x[0][1:].split('_'))), x[1])
        f3 = compose(tuple, sorted, partial(map, f2))
        solution = group_by(f, [(x.VarName, x.X) for x in model.getVars()])
        solution = walk_values(f3, solution)
        cost = 0 # TODO
        return (True, (cost, solution))
    else:
        raise NotImplementedError
