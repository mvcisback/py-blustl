# TODO: add tests where variables are preapplied to constraints
# TODO: add tests for feasible and infeasible constraints
# TODO: Compute eps and M based on x and A, B, dt
# TODO: encode STL robustness metric
# TODO: make inital conditions part of phi
# TODO: implement IIS via slacks
# TODO: weight IIS slacks based priority
# TODO: Look into using SMT
# TODO: Add constraint that x < M

from __future__ import division

from itertools import product, chain, starmap
import operator as op
from math import ceil
from collections import defaultdict
from functools import partial, singledispatch

import pulp as lp
import funcy as fn
from funcy import cat, mapcat, pluck, group_by, drop, walk_values, compose

import stl
from magnum import game
from magnum.game import Game
from magnum.constraint_kinds import Kind as K, Kind
from magnum.constraint_kinds import Category as C
from magnum.utils import Result

DEFAULT_NAME = 'controller_synth'


def z(x: "SL", i: int, g: Game):
    # TODO: come up with better function name
    cat = C.Bool if isinstance(x, stl.LinEq) else C.Real
    if isinstance(x, stl.LinEq):
        prefix = "q"
    else:
        prefix = "z"
    kwargs = {"name": "{}{}".format(prefix, i)}
    if str(x[0]) in g.model.bounds:
        lo, hi = g.model.bounds.get(str(x[0]))
        kwargs = {"lowBound": lo, "upBound": hi, "name": f"{x[0]}_{x[1]}"}
    return lp.LpVariable(cat=cat.value, **kwargs)


def add_constr(model, constr, kind: K, i: int):
    name = "{}{}".format(kind.name, i)
    model.addConstraint(constr, name=name)


def game_to_milp(g: Game):
    # TODO: port to new Signal Logic based API
    # TODO: optimize away top level Ands
    phi = game_to_stl(g)
    model = lp.LpProblem(DEFAULT_NAME, lp.LpMaximize)
    lp_vars = set(stl.utils.vars_in_phi(phi))

    nodes = set(stl.walk(phi))
    store = {x: z(x, i, g) for i, x in enumerate(nodes | lp_vars)}

    stl_constr = cat(encode(phi, store) for phi in nodes)
    constraints = chain(
        stl_constr,
        [(store[phi] == 1, K.ASSERT_FEASIBLE)]  # Assert Feasibility
    )

    for i, (constr, kind) in enumerate(constraints):
        add_constr(model, constr, kind, i)

    # TODO: support alternative objective functions
    model.setObjective(store[phi])
    return model, store


@singledispatch
def encode(psi, s):
    raise NotImplementedError(psi)


@encode.register(stl.LinEq)
def _(psi, s: dict):
    x = sum(float(term.coeff) * s[(term.id, term.time)] for term in psi.terms)
    if psi.op == "=":
        yield x == psi.const, K.PRED_EQ
    else:
        z_t = s[psi]

        M = 1000  # TODO
        # TODO: come up w. better value for eps
        eps = 1e-5 if psi.op in (">=", "<=") else 0
        mu = x - psi.const + \
            eps if psi.op in ("<", "<=") else psi.const - x - eps
        yield -mu <= M * z_t, K.PRED_UPPER
        yield mu <= M * (1 - z_t), K.PRED_LOWER


@encode.register(stl.Neg)
def _(phi, s: dict):
    yield s[phi] == 1 - s[phi.arg], K.NEG


def encode_op(phi: "SL", s: dict, *, k: Kind, isor: bool):
    z_phi = s[phi]
    elems = [s[psi] for psi in phi.args]
    rel, const = (op.ge, 0) if isor else (op.le, 1 - len(elems))

    for e in elems:
        yield rel(z_phi, e), k[0]
    yield rel(const + sum(elems), z_phi), k[1]


encode.register(stl.Or)(partial(encode_op, k=(K.OR, K.OR_TOTAL), isor=True))
encode.register(stl.And)(
    partial(encode_op, k=(K.AND, K.AND_TOTAL), isor=False))


def encode_and_run(g: Game):

    model, store = game_to_milp(g)
    status = lp.LpStatus[model.solve(lp.solvers.COIN())]

    if status in ('Infeasible', 'Unbounded'):
        return Result(False, model, None, None)

    elif status == "Optimal":
        f = lambda x: x[0][0]
        f2 = lambda x: (tuple(map(int, x[0][1:].split('_'))), x[1])
        f3 = compose(tuple, sorted, partial(map, f2))
        variables = {v: (k[1], k[0], v) for k, v in store.items()
                     if not isinstance(k[0], tuple)}

        sol = filter(None, map(variables.get, model.variables()))
        sol = fn.group_by(op.itemgetter(0), sol)
        sol = {t: {y[1]: y[2].value() for y in x} for t, x in sol.items()}
        cost = model.objective.value()
        return Result(True, model, cost, sol)
    else:
        raise NotImplementedError((model, status))
