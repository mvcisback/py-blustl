"""Module for taking a 1 off game and turning into a MPC games."""
# TODO: incorporate external measurements
import stl
import funcy as fn
from lenses import bind, lens
from traces import TimeSeries
from typing import NamedTuple, Mapping

from magnum.game import Game
from magnum.solvers.cegis import solve


class PlannerMessage(NamedTuple):
    measurement: Mapping[str, float]
    obj: stl.STL


def decision_times(g):
    N = g.scope
    dt = g.model.dt
    i = 0
    while True:
        yield i * dt
        if i < N - 1:
            i += 1


@fn.curry
def _time_shift(t, trace):
    return TimeSeries([(tau - t, v) for tau, v in trace.items() if tau >= 0])


@fn.curry
def time_shift(t, xs):
    return fn.walk_values(_time_shift(t), xs)


def get_state(t, xs):
    return fn.walk_values(lens[0].get(), xs)


def echo_env_mpc(orig: Game, *, use_smt=False):
    controller = mpc(orig, use_smt=use_smt)
    res = next(controller)
    while True:
        msg = PlannerMessage(get_state(0, res.solution), obj=orig.specs.obj)
        res = controller.send(msg)
        yield res


def measurement_to_stl(meas):
    def _lineq(var_val):
        var, val = var_val
        return stl.parse(f"{var} = {val:f}")

    return stl.andf(*map(_lineq, meas.items()))


def mpc(orig: Game, *, use_smt=False,):
    prev_games = [(0, orig.scope, orig)]
    while True:
        max_age = max(map(lens[0].get(), prev_games))

        age, _, g = prev_games[0]
        g = g >> max_age - age
        for age, _, g2 in prev_games[1:]:
            g &= g2 >> max_age - age

        res = solve(g, use_smt=use_smt)
        msg = yield bind(res).solution.modify(time_shift(max_age))

        new_init = measurement_to_stl(msg.measurement)
        next_game = orig.reinit(new_init)
        next_game = next_game.new_obj(msg.obj)

        # Stale old games + Remove out of scope games + add new game.
        prev_games = [(t+1, s, g) for t, s, g in prev_games if t < s]
        prev_games.append((0, next_game.scope, next_game))
