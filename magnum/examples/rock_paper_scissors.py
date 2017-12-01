import stl

from sympy import Symbol

import magnum
from magnum import game as G
from magnum.solvers.cegis import cegis_loop

import numpy as np

model = G.Model(
    dt=1,
    H=1,
    vars=G.Vars(
        state=("x", "y"),
        input=("u", ),
        env=("w", )),
    dyn=G.Dynamics(
        A=np.array([[0, 0], [0, 0]]),
        B=np.array([60, 0]).reshape((2, 1)),
        C=np.array([0, 60]).reshape((2, 1)),
    ))

parse = lambda x: stl.parse(x, H=model.H)

context = {
    # Init
    parse("Init"): stl.parse('(x = 0) & (y = 0)'),


    # Rock
    parse("xRock"):
    parse("((x >= -1) & (x < 10)) | ((x >= 50) & (x <= 70))"),
    parse("yRock"):
    parse("((y >= -1) & (y < 10)) | ((y >= 50) & (y <= 70))"),

    # Paper
    parse("xPaper"):
    parse("(x >= 10) & (x < 30)"),
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

    # Bounds
    parse("uBounds"):
    parse("(u >= 0) & (u <= 1)"),
    parse("wBounds"):
    parse("(w >= 0) & (w <= 1)"),
}

spec = G.Specs(
    obj=stl.parse('G(Rules)', H=model.H).inline_context(context),
    learned=stl.TOP,
    init=stl.parse("Init").inline_context(context),
    bounds=stl.parse("G((uBounds) & (wBounds))", 
                     H=model.H).inline_context(context),
    dyn=stl.TOP
)

A, B, C = model.dyn
A = np.eye(A.shape[0]) + A
L_u = magnum.utils.dynamics_lipschitz(A, B, model.H)
L_w = magnum.utils.dynamics_lipschitz(A, C, model.H)
L_x = stl.utils.linear_stl_lipschitz(spec.obj)

meta = G.Meta(
    pri={},
    names={},
    drdu=L_u * L_x,
    drdw=L_w * L_x,
)

rps = G.Game(specs=spec, model=model, meta=meta)
