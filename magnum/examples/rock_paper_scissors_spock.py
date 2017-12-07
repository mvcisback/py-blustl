import stl

from magnum import game as G

import numpy as np

H = 1

model = G.Model(
    dt=1,
    vars=G.Vars(state=("x", "y"), input=("u", ), env=("w", )),
    dyn=G.Dynamics(
        A=np.array([[0, 0], [0, 0]]),
        B=np.array([60, 0]).reshape((2, 1)),
        C=np.array([0, 60]).reshape((2, 1)),
    ))


def parse(x):
    return stl.parse(x, H=1)


context = {
    # Init
    parse("Init"):
    stl.parse('(x = 0) & (y = 0)'),

    # Rock
    parse("xRock"):
    parse("(x < 9) | (x >= 49)"),
    parse("yRock"):
    parse("(y < 10) | (y >= 50)"),

    # Paper
    parse("xPaper"):
    parse("(x >= 10) & (x < 29)"),
    parse("yPaper"):
    parse("(y >= 10) & (y < 30)"),

    # Scissors
    parse("xScissors"):
    parse("(x >= 30) & (x < 50)"),
    parse("yScissors"):
    parse("(y >= 30) & (y < 50)"),

    # Rules
    parse("PaperBeatsRock"):
    parse("(yPaper) -> (~(xRock))"),
    parse("ScissorsBeatsPaper"):
    parse("(yScissors) -> (~(xPaper))"),
    parse("RockBeatsScissors"):
    parse("(yRock) -> (~(xScissors))"),
    parse("Rules"):
    parse("(PaperBeatsRock) & (ScissorsBeatsPaper) & (RockBeatsScissors)"),
}

spec = G.Specs(
    obj=stl.parse('G(Rules)', H=H).inline_context(context),
    learned=stl.TOP,
    init=stl.parse("Init").inline_context(context),
)

rps = G.Game(specs=spec, model=model)
