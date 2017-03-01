# TODO: add tests where variables are preapplied to constraints
# TODO: add tests for feasible and infeasible constraints
# TODO: Compute eps and M based on x and A, B, dt
# TODO: encode STL robustness metric
# TODO: make inital conditions part of phi
# TODO: implement IIS via slacks
# TODO: weight IIS slacks based priority
# TODO: Look into using SMT
# TODO: Add constraint that x < M

from itertools import chain
import operator as op
from functools import partial

import pulp as lp
import funcy as fn
from funcy import cat, group_by, compose

import stl
from magnum import game
from magnum.game import Game
from magnum.constraint_kinds import Kind as K, Kind
from magnum.utils import Result
from magnum.solvers.milp import boolean_encoding as bool_encode

DEFAULT_NAME = 'controller_synth'


def add_constr(model, constr, kind: K, i: int):
    name = "{}{}".format(kind.name, i)
    model.addConstraint(constr, name=name)


def game_to_milp(g: Game):
    # TODO: port to new Signal Logic based API
    # TODO: optimize away top level Ands
    phi = game.game_to_stl(g)
    model = lp.LpProblem(DEFAULT_NAME, lp.LpMaximize)
    lp_vars = set(stl.utils.vars_in_phi(phi))

    nodes = set(stl.walk(phi))
    store = {x: bool_encode.z(x, i, g) for i, x in enumerate(nodes | lp_vars)}

    stl_constr = cat(bool_encode.encode(phi, store) for phi in nodes)
    constraints = chain(
        stl_constr,
        [(store[phi] == 1, K.ASSERT_FEASIBLE)]  # Assert Feasibility
    )

    for i, (constr, kind) in enumerate(constraints):
        add_constr(model, constr, kind, i)

    # TODO: support alternative objective functions
    model.setObjective(store[phi])
    return model, store

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
