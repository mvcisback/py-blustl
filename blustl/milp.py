# TODO: factor out encode recursive structure into a generator
# TODO: add tests where variables are preapplied to constraints
# TODO: add tests for feasible and infeasible constraints
# TODO: Compute eps and M based on x and A, B, dt
# TODO: encode STL robustness metric
# TODO: make inital conditions part of phi
# TODO: implement IIS via slacks
# TODO: weight IIS slacks based priority
# TODO: move model out of store


from __future__ import division

from itertools import product, chain, starmap
import operator as op
from math import ceil
from collections import defaultdict, Counter, namedtuple
from functools import partial, singledispatch

import pulp as lp
from funcy import mapcat, pluck, group_by, drop, walk_values, compose

from blustl import stl
from blustl.game import Game, Phi
from blustl.constraint_kinds import Kind as K, Kind
from blustl.constraint_kinds import Category as C

DEFAULT_NAME = 'controller_synth'


class Store(object):
    def __init__(self, g:Game, x=None, u=None, w=None):
        self._z = defaultdict(dict)
        self.u = defaultdict(dict)
        self.w = defaultdict(dict)
        self.x = defaultdict(dict)
        self._g = g
        self.constr_lookup = defaultdict(dict)
        self._constr_counter = Counter()
        self._count = 0
        self.model = lp.LpProblem("TODO", lp.LpMaximize)

        # Add state, input, and env vars
        elems = [
            ('x', g.dyn.n_vars, self.x, x), 
            ('u', g.dyn.n_sys, self.u, u), 
            ('w', g.dyn.n_env, self.w, w)
        ]

        for pre, num, d, fixed in elems:
            # TODO: support taking bounds from phi/params
            d2 = dict(fixed) if fixed else {}
            ub, lb = (100, -100) if pre == 'x' else (1, 0)
            for i, t in product(range(num), range(g.N + 1)):
                if (i, t) in d2:
                    d[i][t] = d2[i, t]
                else:
                    name = "{}{}_{}".format(pre, i, t)
                    d[i][t] = lp.LpVariable(cat=C.Real.value, name=name, 
                                            lowBound=lb,  upBound=ub)


    def z(self, x:"STL", t:int):
        if x in self._z and t in self._z[x]:
            return self._z[x][t]

        cat = C.Bool if isinstance(x, stl.Pred) else C.Real
        prefix = "z" if isinstance(x, stl.Pred) else "q"
        i = self._count
        self._count += 1
        name = "{}{}_{}".format(prefix, i, t)
        self._z[x][t] = lp.LpVariable(cat=cat.value, name=name)
        encode(x, t, self, self._g)
        return self._z[x][t]

    def add_constr(self, constr, phi:"STL"=None, kind:K=None):
        if not isinstance(constr, lp.LpConstraint):
            return  # only add if symbolic

        self._constr_counter.update([kind])
        name = "{}{}".format(kind.name, self._constr_counter[kind])
        self.model.addConstraint(constr, name=name)
        self.constr_lookup[name] = (phi, kind)


def encode_state_evolution(s:Store, g:Game):
    inputs = lambda t: chain(pluck(t, s.u.values()), pluck(t, s.w.values()))
    state = lambda t: pluck(t, s.x.values())
    dot = lambda x, y: sum(starmap(op.mul, zip(x, y)))
    A, B = g.dyn.ss
    for t in range(g.N):
        for i, (A_i, B_i) in enumerate(zip(A, B)):
            dx = g.dt*(dot(A_i, state(t)) + dot(B_i, inputs(t)))
            constr = s.x[i][t + 1] == s.x[i][t] + dx
            s.add_constr(constr, kind=K.DYNAMICS)


@singledispatch
def encode(g:Game, x=None, u=None, w=None, p1=True):
    """STL -> MILP"""

    sys, env = stl.And(g.phi.sys), stl.And(g.phi.env)
    phi = stl.Or((sys, stl.Neg(env))) if g.phi.env else sys
    if not p1:
        phi = stl.Neg(phi)

    store = Store(g, x=x, u=u, w=w)

    # encode STL constraints
    encode(phi, 0, store, g)
    encode_state_evolution(store, g)

    for psi in g.phi.init:
        x = store.x[psi.lit][0]
        store.add_constr(x == psi.const, kind=K.INIT)

    # Assert top level true
    store.add_constr(store.z(phi, 0) == 1, kind=K.ASSERT_FEASIBLE)

    # Create Objective
    # TODO: support alternative objective functions
    store.model.setObjective(store.z(phi, 0))

    return store.model, store.constr_lookup


@encode.register(stl.Pred)
def _(psi, t:int, s:Store, _):
    x = s.x[psi.lit][t]
    z_t = s.z(psi, t)

    M = 1000  # TODO
    # TODO: come up w. better value for eps
    eps = 0.01 if psi.op == "=" else 0

    mu = x - psi.const if psi.op in ("<", "<=", "=") else psi.const -x
    s.add_constr(mu <= M * z_t - eps, phi=psi, kind=K.PRED_UPPER)
    s.add_constr(-mu <= M * (1 - z_t) - eps, phi=psi, kind=K.PRED_LOWER)


@encode.register(stl.Neg)
def _(phi, t:int, s:Store, _):
    s.add_constr(s.z(phi, t) == 1 - s.z(phi.arg, t), phi, kind=K.NEG)


def encode_bool_op(psi, t:int, s:Store, g:Game, *, k:Kind, isor:bool):
    elems = [s.z(psi2, t) for psi2 in psi.args]
    encode_op(s.z(psi, t), elems, s, psi, k=k, isor=isor)


def encode_temp_op(psi, t:int, s:Store, g:Game, *, k:Kind, isor:bool):
    f = lambda x: int(ceil(x / g.dt))
    a, b = map(f, psi.interval)
    elems = [s.z(psi.arg, t + i) for i in range(a, b + 1) if t + i <= g.N]

    encode_op(s.z(psi, t), elems, s, psi, k=k, isor=isor)


def encode_op(z_psi, elems, s:Store, phi, *, k:Kind, isor:bool):
    rel, const = (op.ge, 0) if isor else (op.le, 1 - len(elems))

    for e in elems:
        s.add_constr(rel(z_psi, e), phi, kind=k[0])
    s.add_constr(rel(const + sum(elems), z_psi), phi, kind=k[1])


encode.register(stl.Or)(partial(encode_bool_op, k=(K.OR, K.OR_TOTAL), isor=True))
encode.register(stl.And)(partial(encode_bool_op, k=(K.AND, K.AND_TOTAL), isor=False))
encode.register(stl.F)(partial(encode_temp_op, k=(K.F, K.F_TOTAL), isor=True))
encode.register(stl.G)(partial(encode_temp_op, k=(K.G, K.G_TOTAL), isor=False))


Result = namedtuple("Result", ["feasible", "model", "cost", "solution"])

def encode_and_run(params, *, x=None, u=None, w=None):
    model, constr_map = encode(params, x=x, u=u, w=w)
    status = lp.LpStatus[model.solve(lp.solvers.COIN())]
    
    if status in ('Infeasible', 'Unbounded'):
        return Result(False, model, None, None)

    elif status == "Optimal":
        f = lambda x: x[0][0]
        f2 = lambda x: (tuple(map(int, x[0][1:].split('_'))), x[1])
        f3 = compose(tuple, sorted, partial(map, f2))
        solution = group_by(f, [(x.name, x.value()) for x in model.variables()])
        solution = walk_values(f3, solution)
        cost = model.objective.value()
        return Result(True, model, cost, solution)
    else:
        raise NotImplementedError
