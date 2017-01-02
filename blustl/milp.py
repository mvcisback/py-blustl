# TODO: add tests where variables are preapplied to constraints
# TODO: add tests for feasible and infeasible constraints
# TODO: Compute eps and M based on x and A, B, dt
# TODO: encode STL robustness metric
# TODO: make inital conditions part of phi
# TODO: implement IIS via slacks
# TODO: weight IIS slacks based priority
# TODO: Look into using SMT
# TODO: Add constraint that x < M
# TODO: incorporate bounds

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

def z(x:"SL", i:int):
    # TODO: come up with better function name
    cat = C.Bool if isinstance(x, stl.LinEq) else C.Real
    if isinstance(x, stl.LinEq):
        prefix = "q"
    elif isinstance(x, stl.Var):
        prefix = "var"
    else:
        prefix = "z"
    name = "{}{}".format(prefix, i)
    return lp.LpVariable(cat=cat.value, name=name)


def add_constr(model, constr, kind:K, i:int):
    name = "{}{}".format(kind.name, i)
    model.addConstraint(constr, name=name)


def sl_to_milp(phi:"SL", assigned=None, p1=True):
    """STL -> MILP"""
    # TODO: port to new Signal Logic based API
    # TODO: constraint_map
    # TODO: optimize away top level Ands
    if assigned is None:
        assigned = {}

    constraint_map = {}
    model = lp.LpProblem(DEFAULT_NAME, lp.LpMaximize)
    lp_vars = set(game.vars_in_phi(phi))
    nodes = set(stl.walk(phi))
    store = {x: z(x, i) for i, x in enumerate(nodes | lp_vars)}

    stl_constr = cat(encode(phi, store) for phi in nodes)
    constraints = chain(
        stl_constr,
        [(store[phi] == 1, K.ASSERT_FEASIBLE)] # Assert Feasibility
    )
    
    for i, (constr, kind) in enumerate(constraints):
        add_constr(model, constr, kind, i)

    # TODO: support alternative objective functions
    model.setObjective(store[phi])
    
    return model, constraint_map


@singledispatch
def encode(psi, s):
    raise NotImplementedError


@encode.register(stl.LinEq)
def _(psi, s:dict):
    z_t = s[psi]
    x = sum(float(term.coeff)*s[(term.id, term.time)] for term in psi.terms)
    M = 1000  # TODO
    # TODO: come up w. better value for eps
    eps = 0.01 if psi.op == "=" else 0

    mu = x - psi.const if psi.op in ("<", "<=", "=") else psi.const -x
    yield -mu <= M * z_t - eps, K.PRED_UPPER
    yield mu <= M * (1 - z_t) - eps, K.PRED_LOWER


@encode.register(stl.Neg)
def _(phi, s:dict):
    yield s[phi] == 1 - s[phi.arg], K.NEG


def encode_op(phi:"SL", s:dict, *, k:Kind, isor:bool):
    z_phi = s[phi]
    elems = [s[psi] for psi in phi.args]
    rel, const = (op.ge, 0) if isor else (op.le, 1 - len(elems))

    for e in elems:
        yield rel(z_phi, e), k[0]
    yield rel(const + sum(elems), z_phi), k[1]


encode.register(stl.Or)(partial(encode_op, k=(K.OR, K.OR_TOTAL), isor=True))
encode.register(stl.And)(partial(encode_op, k=(K.AND, K.AND_TOTAL), isor=False))


def encode_and_run(phi, *, assigned=None):
    if assigned is None:
        assigned = {}
    model, _ = sl_to_milp(phi, assigned=assigned)
    status = lp.LpStatus[model.solve(lp.solvers.COIN())]

    if status in ('Infeasible', 'Unbounded'):
        return Result(False, model, None, None)

    elif status == "Optimal":
        f = lambda x: x[0][0]
        f2 = lambda x: (tuple(map(int, x[0][1:].split('_'))), x[1])
        f3 = compose(tuple, sorted, partial(map, f2))
        # TODO:
        # - Change to list of 2*Horizon state/input/env variable sets
        solution = group_by(f, [(x.name, x.value()) for x in model.variables()])
        solution = walk_values(f3, solution)
        cost = model.objective.value()
        return Result(True, model, cost, solution)
    else:
        raise NotImplementedError
