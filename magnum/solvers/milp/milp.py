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
from funcy import cat, compose

import stl
from magnum.game import Game
from magnum.constraint_kinds import Kind as K
from magnum.utils import Result
from magnum.solvers.milp import boolean_encoding as bool_encode
from magnum.solvers.milp import robustness_encoding as rob_encode

DEFAULT_NAME = 'controller_synth'


def add_constr(model, constr, kind: K, i: int):
    name = "{}{}".format(kind.name, i)
    model.addConstraint(constr, name=name)


def spec_nodes(phi):
    if isinstance(phi, stl.And):
        return cat(stl.walk(psi) for psi in phi.args)
    return stl.walk(phi)


def game_to_milp(g: Game, robust=True):
    # TODO: optimize away top level Ands
    z = rob_encode.z if robust else bool_encode.z
    encode = rob_encode.encode if robust else bool_encode.encode

    model = lp.LpProblem(DEFAULT_NAME, lp.LpMaximize)

    spec = ~g.spec.env | g.spec.sys
    phis = [spec] + list(g.spec[2:])
    phis = [x for x in phis if x not in (stl.TOP, stl.BOT)]

    lp_vars = set.union(*(set(stl.utils.vars_in_phi(phi)) for phi in phis))
    nodes = set.union(*(set(spec_nodes(phi)) for phi in phis)) | {spec}
    store = {x: z(x, i, g) for i, x in enumerate(nodes | lp_vars)}

    stl_constr = cat(encode(phi, store) for phi in nodes)
    constraints = chain(
        stl_constr,
        [(store[spec] == 1, K.ASSERT_FEASIBLE)]  # Assert Feasibility
    ) if not robust else stl_constr

    for i, (constr, kind) in enumerate(constraints):
        add_constr(model, constr, kind, i)

    # TODO: support alternative objective functions
    J = store[spec][0] if isinstance(store[spec], tuple) else store[spec]
    model.setObjective(J)
    return model, store


def encode_and_run(g: Game, robust=True):
    model, store = game_to_milp(g, robust)
    status = lp.LpStatus[model.solve(lp.solvers.COIN())]
    if status in ('Infeasible', 'Unbounded'):
        return Result(False, None, None)

    elif status == "Optimal":
        key = op.itemgetter(0) if robust else lambda x: x
        symbols = {key(v): (k[1], k[0], v[0]) for k, v in store.items()
                   if not isinstance(k[0], tuple)}

        sol = (symbols[sym] for sym in model.variables() if sym in symbols)
        sol = fn.group_by(op.itemgetter(0), sol)
        sol = {t: {y[1]: y[2].value() for y in x} for t, x in sol.items()}
        cost = model.objective.value()
        return Result(cost > 0, cost, sol)
    else:
        raise NotImplementedError((model, status))
