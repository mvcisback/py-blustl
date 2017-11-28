# TODO: add tests where variables are preapplied to constraints
# TODO: add tests for feasible and infeasible constraints
# TODO: Compute eps and M based on x and A, B, dt
# TODO: Add constraint that x < M

from itertools import chain, product
import operator as op
from functools import partial

import pulp as lp
import funcy as fn
import stl
import traces
from bidict import bidict
from lenses import bind
from funcy import cat, compose

from magnum.game import Game
from magnum.constraint_kinds import Kind as K
from magnum.utils import Result
from magnum.solvers.milp import robustness_encoding as rob_encode
from magnum.solvers.milp import boolean_encoding as bool_encode


DEFAULT_NAME = 'controller_synth'


def add_constr(model, constr, kind: K, i: int):
    name = "{}{}".format(kind.name, i)
    model.addConstraint(constr, name=name)


def create_store(obj, non_obj, times):
    z = rob_encode.z
    
    var_names = fn.mapcat(lambda x: x.var_names, ({obj} | non_obj))
    lp_vars = product(var_names, times)
    nodes = set(obj.walk())
    return bidict({x: z(x, i) for i, x 
                   in enumerate(fn.chain(nodes, lp_vars))})
    

def game_to_milp(g: Game, robust=True):
    model = lp.LpProblem(DEFAULT_NAME, lp.LpMaximize)

    specs = {
        'obj': g.specs.obj,
        'init': g.specs.init,
        'dyn': g.specs.dyn,
        'learned': g.specs.learned,
        'bounds': g.specs.bounds
    }

    obj = g.specs.obj
    non_obj = set(g.specs) - {obj, stl.TOP}

    obj = stl.utils.discretize(obj, dt=g.model.dt, distribute=True)
    non_obj = {stl.utils.discretize(phi, dt=g.model.dt, distribute=True)
               for phi in non_obj}
    
    store = create_store(obj, non_obj, g.times)

    # Constraints
    robustness = rob_encode.encode(obj, store, 0)
    dynamics = rob_encode.encode_dynamics(g, store)
    other = cat(bool_encode.encode(psi, store, 0) for psi in non_obj)

    constraints = fn.chain(robustness, dynamics, other)

    for i, (constr, kind) in enumerate(constraints):
        add_constr(model, constr, kind, i)

    # TODO: support alternative objective functions
    J = store[obj][0] if isinstance(store[obj], tuple) else store[obj]
    model.setObjective(J)
    return model, store


# Encoding the dynamics

def extract_ts(name, model, g, store):
    dt = g.model.dt
    model = {x: x.value() for x in model.variables()}
    ts = traces.TimeSeries(((dt*t, model[store[name, t][0]])
                             for t in g.times), domain=(0, g.model.H))
    ts.compact()
    return ts


def encode_and_run(g: Game, robust=True):
    model, store = game_to_milp(g, robust)
    status = lp.LpStatus[model.solve(lp.solvers.COIN())]
    if status in ('Infeasible', 'Unbounded'):
        return Result(False, None, None)

    elif status == "Optimal":
        cost = model.objective.value()
        sol = {v: extract_ts(v, model, g, store) for v in fn.cat(g.model.vars)}
        return Result(cost > 0, cost, sol)
    else:
        raise NotImplementedError((model, status))

