"""
TODO: Game to pdf
TODO: Create script to automatically generate game spec
TODO: Include meta information in Game
- Annotate stl with priority
- Annotate stl with name
- Annotate stl with changeability
TODO: create map from SL expr to matching Temporal Logic term after conversion
TODO: refactor discretization
"""

from typing import NamedTuple, Tuple, TypeVar, Mapping

import numpy as np
from lenses import bind

import stl
from stl import STL

Matrix = TypeVar('M')


class Dynamics(NamedTuple):
    A: Matrix
    B: Matrix
    C: Matrix


class Specs(NamedTuple):
    obj: STL
    init: STL
    dyn: STL
    learned: STL
    bounds: STL


class Vars(NamedTuple):
    state: Tuple[STL]
    input: Tuple[STL]
    env: Tuple[STL]


class Model(NamedTuple):
    dt: float
    H: float
    vars: Vars
    dyn: Dynamics


class Meta(NamedTuple):
    names: Mapping[STL, str]
    pri: Mapping[STL, int]
    drdu: float
    drdw: float


class DiscreteGame(NamedTuple):
    specs: Specs
    vars: Vars
    meta: Meta


# TODO: Make a more convenient constructor
class Game(NamedTuple):
    specs: Specs
    model: Model
    meta: Meta

    def as_stl(self):
        g = self
        dyn = matrix_to_dyn_stl(g)
        obj = g.specs.obj
        learned = g.specs.learned
        init = g.specs.init
        bounds = g.specs.bounds
        return dyn & obj & learned & init & bounds

    def invert(self):
        # Swap Spec
        g = bind(self).specs.obj.set(~self.specs.obj)
        # Swap Inputs
        g = bind(g).model.vars.env.set(self.model.vars.input)
        g = bind(g).model.vars.input.set(self.model.vars.env)

        # Swap Dynamics Bounds
        g = bind(g).meta.drdu.set(self.meta.drdw)
        g = bind(g).meta.drdw.set(self.meta.drdu)
        return g


def discretize(g):
    specs = Specs(*(discretize_stl(spec, m=g.model) for spec in g.spec))
    return bind(g).specs.set(specs)
