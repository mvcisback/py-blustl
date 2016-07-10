# TODO: add tests where variables are preapplied to constraints
# TODO: add tests for feasible and infeasible constraints
# TODO: Compute eps and M based on x and A, B, dt
# TODO: encode STL robustness metric
# TODO: make inital conditions part of phi
# TODO: implement IIS via slacks
# TODO: weight IIS slacks based priority
# TODO: move model out of store
# TODO: make store simply a namedtuple
# TODO: Look into using SMT
# TODO: move dynamics out of encoding
# TODO: Remove dynamics/time info
# TODO: Add constraint that x < M


from __future__ import division

from itertools import product, chain, starmap
import operator as op
from math import ceil
from collections import defaultdict, Counter, namedtuple
from functools import partial, singledispatch

import pulp as lp
import funcy as fn
from funcy import cat, mapcat, pluck, group_by, drop, walk_values, compose

import stl
from blustl import game
from blustl.game import Game
from blustl.constraint_kinds import Kind as K, Kind
from blustl.constraint_kinds import Category as C

DEFAULT_NAME = 'controller_synth'

Result = namedtuple("Result", ["feasible", "model", "cost", "solution"])

def z(x:"STL", i:int):
    cat = C.Bool if isinstance(x, stl.LinEq) else C.Real
    if isinstance(x, stl.LinEq):
        prefix = "q"
    elif isinstance(x, stl.Var):
        prefix = "var"
    else:
        prefix = "z"
    name = "{}{}".format(prefix, i)
    return lp.LpVariable(cat=cat.value, name=name)


def sl_to_milp(phi:"SL", assigned=None, p1=True):
    """STL -> MILP"""
    # TODO: port to new Signal Logic based API
    if assigned is None:
        assigned = {}

    nodes = set(fn.cat(game.vars_in_phi(phi), stl.walk(phi)))
    store = {x: z(x, i) for i, x in nodes}

    return store

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

@singledispatch
def encode(_):
    raise NotImplementedError


@encode.register(stl.LinEq)
def _(psi, t:int, s, _):
    x = s.x[psi.lit][t]
    z_t = s.z[psi, t]

    M = 1000  # TODO
    # TODO: come up w. better value for eps
    eps = 0.01 if psi.op == "=" else 0

    mu = x - psi.const if psi.op in ("<", "<=", "=") else psi.const -x
    yield -mu <= M * z_t - eps, K.PRED_UPPER
    yield mu <= M * (1 - z_t) - eps, K.PRED_LOWER


@encode.register(stl.Neg)
def _(phi, t:int, s, _):
    yield s.z[phi, t] == 1 - s.z[phi.arg, t], K.NEG


def encode_bool_op(psi, t:int, s, g:Game, *, k:Kind, isor:bool):
    elems = [s.z[psi2, t] for psi2 in psi.args]
    yield from encode_op(s.z[psi, t], elems, s, psi, k=k, isor=isor)


def encode_temp_op(psi, t:int, s, g:Game, *, k:Kind, isor:bool):
    a, b = map(partial(step, dt=g.dt), psi.interval)
    elems = [s.z[psi.arg, t + i] for i in range(a, b + 1) if t + i <= g.N]

    yield from encode_op(s.z[psi, t], elems, s, psi, k=k, isor=isor)


def encode_op(z_psi, elems, s, phi, *, k:Kind, isor:bool):
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
