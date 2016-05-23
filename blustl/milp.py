# TODO: add tests where variables are preapplied to constraints
# TODO: add tests for feasible and infeasible constraints
# TODO: Compute eps and M based on x and A, B, dt
# TODO: encode STL robustness metric
# TODO: make inital conditions part of phi
# TODO: implement IIS via slacks
# TODO: weight IIS slacks based priority
# TODO: move model out of store
# TODO: make store simply a namedtuple


from __future__ import division

from itertools import product, chain, starmap
import operator as op
from math import ceil
from collections import defaultdict, Counter, namedtuple
from functools import partial, singledispatch

import pulp as lp
from funcy import cat, mapcat, pluck, group_by, drop, walk_values, compose

from blustl import stl
from blustl.game import Game, Phi
from blustl.constraint_kinds import Kind as K, Kind
from blustl.constraint_kinds import Category as C

DEFAULT_NAME = 'controller_synth'

Result = namedtuple("Result", ["feasible", "model", "cost", "solution"])


class Store(object):
    def __init__(self, g:Game, x=None, u=None, w=None):
        self.z = {}
        self.u = defaultdict(dict)
        self.w = defaultdict(dict)
        self.x = defaultdict(dict)
        self._g = g
        self.constr_lookup = defaultdict(dict)
        self._constr_counter = Counter()
        self._count = 0
        self.model = lp.LpProblem(DEFAULT_NAME, lp.LpMaximize)
        self.add_state_vars(g, x=x, u=u, w=w)


    def add_state_vars(self, g:Game, *, x, u, w):
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
            yield constr, K.DYNAMICS


def z(x:"STL", t:int, i:int):
    cat = C.Bool if isinstance(x, stl.Pred) else C.Real
    prefix = "z" if isinstance(x, stl.Pred) else "q"
    name = "{}{}_{}".format(prefix, i, t)
    return lp.LpVariable(cat=cat.value, name=name)


def timed_vars(phi, g:Game):
    times = dict(active_times(phi, dt=g.dt, N=g.N))
    for i, x in enumerate(stl.walk(phi)):
        for t in times[x]:
            yield (x, t), z(phi, t, i)


@singledispatch
def encode(g:Game, x=None, u=None, w=None, p1=True):
    """STL -> MILP"""
    
    sys, env = stl.And(g.phi.sys), stl.And(g.phi.env)

    phi = stl.Or((sys, stl.Neg(env))) if g.phi.env else sys
    if not p1:
        phi = stl.Neg(phi)

    store = Store(g, x=x, u=u, w=w)
    store.z.update(dict(timed_vars(phi, g)))

    stl_constr = cat(encode(x, t, store, g) for x, t in store.z.keys())
    init_constr = ((store.x[x.lit][0] == x.const, K.INIT) for x in g.phi.init)
    constraints = chain(
        stl_constr,
        init_constr,
        encode(phi, 0, store, g),
        encode_state_evolution(store, g),
        [(store.z[phi, 0] == 1, K.ASSERT_FEASIBLE)], # Assert Feasible
    )
    
    # Add Constraints
    for constr, kind in constraints:
        store.add_constr(constr, phi, kind=kind)

    # Create Objective
    # TODO: support alternative objective functions
    store.model.setObjective(store.z[phi, 0])


    return store.model, store.constr_lookup


@encode.register(stl.Pred)
def _(psi, t:int, s:Store, _):
    x = s.x[psi.lit][t]
    z_t = s.z[psi, t]

    M = 1000  # TODO
    # TODO: come up w. better value for eps
    eps = 0.01 if psi.op == "=" else 0

    mu = x - psi.const if psi.op in ("<", "<=", "=") else psi.const -x
    yield mu <= M * z_t - eps, K.PRED_UPPER
    yield -mu <= M * (1 - z_t) - eps, K.PRED_LOWER


@encode.register(stl.Neg)
def _(phi, t:int, s:Store, _):
    yield s.z[phi, t] == 1 - s.z[phi.arg, t], K.NEG


def encode_bool_op(psi, t:int, s:Store, g:Game, *, k:Kind, isor:bool):
    elems = [s.z[psi2, t] for psi2 in psi.args]
    yield from encode_op(s.z[psi, t], elems, s, psi, k=k, isor=isor)


def step(t:float, dt:float):
    return int(ceil(t / dt))


def encode_temp_op(psi, t:int, s:Store, g:Game, *, k:Kind, isor:bool):
    a, b = map(partial(step, dt=g.dt), psi.interval)
    try:
        elems = [s.z[psi.arg, t + i] for i in range(a, b + 1) if t + i <= g.N]
    except:
        import ipdb; ipdb.set_trace()


    yield from encode_op(s.z[psi, t], elems, s, psi, k=k, isor=isor)


def encode_op(z_psi, elems, s:Store, phi, *, k:Kind, isor:bool):
    rel, const = (op.ge, 0) if isor else (op.le, 1 - len(elems))

    for e in elems:
        yield rel(z_psi, e), k[0]
    yield rel(const + sum(elems), z_psi), k[1]


encode.register(stl.Or)(partial(encode_bool_op, k=(K.OR, K.OR_TOTAL), isor=True))
encode.register(stl.And)(partial(encode_bool_op, k=(K.AND, K.AND_TOTAL), isor=False))
encode.register(stl.F)(partial(encode_temp_op, k=(K.F, K.F_TOTAL), isor=True))
encode.register(stl.G)(partial(encode_temp_op, k=(K.G, K.G_TOTAL), isor=False))


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


def active_times(phi, *, dt, N, t_0=0, t_f=0):
    yield phi, range(t_0, t_f + 1)

    if not isinstance(phi, stl.Pred):
        lo, hi = phi.interval if isinstance(phi, stl.ModalOp) else (0, 0)
        f = lambda x: min(step(x, dt=dt), N)
        lo2, hi2 = map(f, (t_0 + lo, t_f + hi))
        for child in phi.children():
            yield from active_times(child, dt=dt, N=N, t_0=lo2, t_f=hi2) 
