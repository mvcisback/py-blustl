# TODO: add tests where variables are preapplied to constraints
# TODO: add tests for feasible and infeasible constraints
# TODO: Compute eps and M based on x and A, B, dt
# TODO: Add constraint that x < M

from collections import defaultdict
from itertools import chain, product
import operator as op
from functools import partial

import funcy as fn
import stl
import traces
from lenses import bind
from funcy import cat, compose
from optlang import Model, Variable, Constraint, Objective

from magnum.game import Game, Specs, Vars
from magnum.constraint_kinds import Kind as K
from magnum.utils import Result
from magnum.solvers.milp import robustness_encoding as rob_encode
from magnum.solvers.milp import boolean_encoding as bool_encode


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def add_constr(model, constr, kind: K, i: int):
    model.add(constr)


def counter_example_store(g, ce, i):
    def relabel(x):
        return x if i == 0 else f"{x}#{i}"

    dt = g.model.dt
    return {(relabel(name), t): (trace[dt*t],)
            for (name, trace), t in product(ce.items(), g.times)}


def encode_game(g, store, ce=None):
    # obj, *non_obj = relabel_specs(g, counter_examples)
    obj, *non_obj = g.specs

    obj = stl.utils.discretize(obj, dt=g.model.dt, distribute=True)
    non_obj = {stl.utils.discretize(phi, dt=g.model.dt, distribute=True)
               for phi in non_obj if phi != stl.TOP}

    # Constraints
    robustness = rob_encode.encode(obj, store, 0)
    # TODO
    dynamics = rob_encode.encode_dynamics(g, store)
    other = cat(bool_encode.encode(psi, store, 0) for psi in non_obj)
    return fn.chain(robustness, dynamics, other), obj


def create_scenario(g, i):
    def relabel(x):
        return x if i == 0 or x in g.model.vars.input else f"{x}#{i}"

    relabel_phi = stl.ast.lineq_lens.terms.Each().id.modify(relabel)

    g = bind(g).specs.Each().modify(relabel_phi)
    g = bind(g).model.vars.Each().Each().modify(relabel)
    return g


def game_to_milp(g: Game, robust=True, counter_examples=None):
    # TODO: implement counter_example encoding
    if not counter_examples:
        counter_examples = [{}]

    model = Model()
    store = keydefaultdict(lambda x: rob_encode.z(x, g))
    # Add counter examples to store
    for i, ce in enumerate(counter_examples):
        store.update(counter_example_store(g, ce, i))

    # Encode each scenario.
    scenarios = [create_scenario(g, i) for i, ce in enumerate(counter_examples)]
    constraints, objs = zip(*(encode_game(g2, store) for g2 in scenarios))

    # Objective is to maximize the minimum robustness of the scenarios.
    obj = stl.andf(*objs)
    constraints = chain(rob_encode.encode(obj, store, 0), fn.cat(constraints))

    for i, (constr, kind) in enumerate(constraints):
        if constr is True:
            continue
        add_constr(model, constr, kind, i)

    # TODO: support alternative objective functions
    J = store[obj][0] if isinstance(store[obj], tuple) else store[obj]
    model.objective = Objective(J, direction='max')
    return model, store


# Encoding the dynamics

def extract_ts(name, model, g, store):
    dt = g.model.dt
    model = {k: v.primal for k, v in model.variables.items()}
    ts = traces.TimeSeries(((dt*t, model[store[name, t][0].name])
                            for t in g.times 
                            if not isinstance(store[name, t][0], (float, int)) 
                            and store[name, t][0].name in model)
                           , domain=(0, g.model.H))

    ts.compact()
    return ts


def encode_and_run(g: Game, robust=True, counter_examples=None):
    model, store = game_to_milp(g, robust, counter_examples)
    status = model.optimize()

    if status in ('infeasible', 'unbounded'):
        return Result(False, None, None)

    elif status == "optimal":
        cost = model.objective.value
        sol = {v: extract_ts(v, model, g, store) for v in fn.cat(g.model.vars)}
        return Result(cost > 0, cost, sol)
    else:
        raise NotImplementedError((model, status))

