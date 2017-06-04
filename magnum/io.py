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


def write(g, f):
    pass
