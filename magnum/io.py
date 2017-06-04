from itertools import chain
from pathlib import Path

import capnp
import stl
import numpy as np
from sympy import Symbol


from magnum import game

GameCapnp = capnp.load(str(Path(__file__).parent / "game.capnp"))
Game = GameCapnp.Game
Specs = GameCapnp.Game.Specs
Spec = GameCapnp.Game.Specs.Spec
Model = GameCapnp.Game.Model
Var = GameCapnp.Game.Model.Var
Dynamics = GameCapnp.Game.Model.Dynamics
MetaData = GameCapnp.Game.MetaData


oo = float('inf')


########## Load ##########

def load_specs(g):
    return game.Specs(
        obj=stl.parse(g.specs.objective.text),
        learned=[stl.parse(x.text) for x in g.specs.learned],
    )


def load_model(g):
    bounds = {Symbol(x.name): (x.lowerBound, x.upperBound) for x in 
              chain(g.model.state, g.model.inputs, g.model.environmentInput)}
    return game.Model(
        dt=g.model.dt,
        N=g.model.horizon,
        bounds=bounds,
        t=g.model.currentTimeStep,
        vars=game.Vars(
            state=tuple(Symbol(x.name) for x in g.model.state),
            input=tuple(Symbol(x.name) for x in g.model.inputs),
            env=tuple(Symbol(x.name) for x in g.model.environmentInput)
        ),
        dyn=game.Dynamics(
            A=np.array(g.model.dynamics.aMatrix, dtype=np.float64),
            B=np.array(g.model.dynamics.bMatrix, dtype=np.float64),
            C=np.array(g.model.dynamics.cMatrix, dtype=np.float64)
        )
    )


def load_metadata(g):
    return game.Meta(
        dxdu=g.meta.dxdu,
        dxdw=g.meta.dxdw,
        drdx=g.meta.drdx,
        pri={stl.parse(x.text): x.priority for x in
             chain([g.specs.objective], g.specs.learned) if x.priority != 0},
        names={stl.parse(x.text): x.name for x in
               chain([g.specs.objective], g.specs.learned) if x.name != ""},
    )


def load_game(g):
    return game.Game(
        spec=load_specs(g),
        model=load_model(g),
        meta=load_metadata(g),
    )


def load(f):
    return load_game(Game.read(f))


########## Write ##########

def to_capnp_spec(g, phi, name=None):
    return Spec.new_message(
        name=g.meta.names.get(phi, ""),
        priority=g.meta.pri.get(phi, 0),
        text=str(phi)
    )


def to_capnp_specs(g):
    return Specs.new_message(
        objective=to_capnp_spec(g, g.spec.obj),
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
        dynamics=Dynamics.new_message(
            aMatrix=g.model.dyn.A,
            bMatrix=g.model.dyn.B,
            cMatrix=g.model.dyn.C,
        )
    )


def to_capnp_metadata(g):
    return MetaData.new_message(
        dxdu=oo if g.meta.dxdu is None else g.meta.dxdu,
        dxdw=oo if g.meta.dxdw is None else g.meta.dxdw,
        drdx=oo if g.meta.drdx is None else g.meta.drdx
    )

def to_capnp_game(g):
    return Game.new_message(
        specs=to_capnp_specs(g),
        model=to_capnp_model(g),
        meta=to_capnp_metadata(g)
    )


def write(g, f):
    to_capnp_game(g).write(f)
