from typing import NamedTuple, TypeVar, Mapping, List
from itertools import product

import funcy as fn

import stl

Trace = TypeVar('Trace')


class Result(NamedTuple):
    feasible: bool
    cost: float
    solution: Mapping[str, Trace]
    counter_examples: List[Trace]

    def input(self, g):
        return fn.project(self.solution, g.model.vars.input)

    def output(self, g):
        return fn.project(self.solution, g.model.vars.state)

    def env_input(self, g):
        return fn.project(self.solution, g.model.vars.env)


def encode_refuted_rec(refuted_input, radius, times, dt=1):
    def _encode_refuted(name_time):
        u, t = name_time
        val = refuted_input[u][dt * t]
        lo, hi = val - radius, val + radius
        phi = stl.BOT
        if lo > -1:
            phi |= stl.utils.next(stl.parse(f'{u} < {lo:f}'), i=t)

        if hi < 1:
            phi |= stl.utils.next(stl.parse(f'{u} > {hi:f}'), i=t)

        if phi == stl.BOT:
            return stl.TOP
        else:
            return phi

    return stl.orf(*map(_encode_refuted, product(refuted_input.keys(), times)))
