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
from typing import NamedTuple, Tuple, TypeVar

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
    learned: STL


class Vars(NamedTuple):
    state: Tuple[STL]
    input: Tuple[STL]
    env: Tuple[STL]


class Model(NamedTuple):
    dt: float
    vars: Vars
    dyn: Dynamics


# TODO: Make a more convenient constructor
class Game(NamedTuple):
    specs: Specs
    model: Model

    def spec_as_stl(self, discretize=True):
        spec = stl.andf(*self.specs)

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
        return g

    @property
    def times(self):
        return range(1 + self.scope)

    @property
    def scope(self):
        return stl.utils.scope(self.spec_as_stl(), 1)

    @property
    def scaled_scope(self):
        return stl.utils.scope(self.spec_as_stl(), self.model.dt)

    @property
    def scaled_times(self):
        return [self.model.dt * t for t in self.times]

    def new_horizon(self, H):
        g = bind(self).specs.obj.modify(lambda x: stl.alw(x, lo=0, hi=H))
        return g

    def learn(self, phi):
        return bind(self).specs.learned.modify(lambda x: x & phi)

    def __rshift__(self, t):
        return bind(self).specs.Each().call('__rshift__', t=t)

    def __and__(self, g):
        if self.model != g.model:
            # TODO:
            # 1. Check that the dynamics are compatible.
            # 2. If so, merge states etc.
            raise NotImplementedError

        g2 = bind(self).specs.obj.modify(lambda x: x & g.specs.obj)
        g2 = bind(self).specs.init.modify(lambda x: x & g.specs.init)
        g2 = bind(self).specs.learned.modify(lambda x: x & g.specs.learned)
        return g2

    def reinit(self, phi):
        return bind(self).specs.init.set(phi)

    def new_obj(self, phi):
        return bind(self).specs.obj.set(phi)
