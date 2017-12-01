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
from math import ceil
from typing import NamedTuple, Tuple, TypeVar, Mapping

import stl
from lenses import bind
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


# TODO: Make a more convenient constructor
class Game(NamedTuple):
    specs: Specs
    model: Model
    meta: Meta

    def spec_as_stl(self, discretize=True):
        spec = self.specs.obj
        spec &= self.specs.learned
        spec &= self.specs.init
        spec &= self.specs.bounds

        if discretize:
            spec = stl.utils.discretize(spec, self.model.dt)

        return spec

    def invert(self):
        # Swap Spec
        g = bind(self).specs.obj.set(~self.specs.obj)
        # Swap Inputs
        g = bind(g).model.vars.env.set(self.model.vars.input)
        g = bind(g).model.vars.input.set(self.model.vars.env)

        # Swap Dynamics
        g = bind(g).model.dyn.B.set(self.model.dyn.C)
        g = bind(g).model.dyn.C.set(self.model.dyn.B)

        # Swap Dynamics Bounds
        g = bind(g).meta.drdu.set(self.meta.drdw)
        g = bind(g).meta.drdw.set(self.meta.drdu)
        return g

    @property
    def times(self):
        return range(ceil(self.model.H/self.model.dt) + 1)
