from pathlib import Path

import capnp

from magnum import game

GameCapnp = capnp.load(str(Path(__file__).parent / "game.capnp"))
Game = GameCapnp.Game
Specs = GameCapnp.Game.Specs
Spec = GameCapnp.Game.Specs.Spec
Model = GameCapnp.Game.Model
Var = GameCapnp.Game.Model.Var
MetaData = GameCapnp.Game.MetaData


########## Load ##########

def load_specs():
    pass


def load_model():
    pass


def load_metadata():
    pass


def load(f):
    pass


########## Write ##########

def to_capnp_spec(g, phi, name=None):
    return Spec.new_message(
        name=g.meta.names.get(phi, ""),
        priority=g.meta.pri.get(phi, 100),
        text=str(phi)
    )


def to_capnp_specs(g):
    return Specs.new_message(
        objective=to_capnp_spec(g, g.spec.obj),
        dynamics=[to_capnp_spec(g, x) for x in g.spec.dyn],
        learned=[to_capnp_spec(g, x) for x in g.spec.learned],
    )


def to_capnp_var(g, var):
    lo, hi = g.model.bounds[var]
    return Var.new_message(
        name=str(var),
        lowerBound=lo,
        upperBound=hi
    )


def to_capnp_model(g):
    return Model.new_message(
        dt=g.model.dt,
        horizon=g.model.N,
        state=[to_capnp_var(g, x) for x in g.model.vars.state],
        inputs=[to_capnp_var(g, x) for x in g.model.vars.input],
        environmentInput=[to_capnp_var(g, x) for x in g.model.vars.env],
        currentTimeStep=g.model.t,
    )


def to_capnp_metadata(g):
    md = MetaData.new_message()
    if g.meta.dxdu is not None:
        md.dxdu = g.meta.dxdu
    if g.meta.dxdw is not None:
        md.dxdw = g.meta.dxdw
    if g.meta.drdx is not None:
        md.drdx = g.meta.drdx

    return md


def to_capnp_game(g):
    return Game.new_message(
        specs=to_capnp_specs(g),
        model=to_capnp_model(g),
        meta=to_capnp_metadata(g)
    )


def write(g, f):
    pass
