import stl

from sympy import Symbol

import magnum
from magnum import game as G
from magnum import io
from magnum.solvers.cegis import cegis_loop

import numpy as np

model = G.Model(
    dt=1,
    H=2,
    vars=G.Vars(
        state=(
            Symbol("x"),
            Symbol("y"),
        ), 
       input=(Symbol("u"),),
        env=(Symbol("w"),)
    ),
    t=0,
    dyn=G.Dynamics(
        A=np.array([
            [-1, 0],
            [0, -1]
        ]),
        B=np.array([3, 0]).reshape((2,1)),
        C=np.array([0, 3]).reshape((2,1)),
    )
)

parse = lambda x: stl.parse(x, H=model.H)

context = {
    # Init
    parse("xInit"): parse("x = 0"),
    parse("yInit"): parse("y = 0"),
    parse("Init"): parse("(xInit) & (yInit)"),

    # Rock
    parse("xRock"): parse("(x >= 0) & (x < 1)"),
    parse("yRock"): parse("(y >= 0) & (y < 1)"),

    # Paper
    parse("xPaper"): parse("(x >= 1) & (x < 2)"),
    parse("yPaper"): parse("(y >= 1) & (y < 2)"),

    # Scissors
    parse("xScissors"): parse("(x >= 2) & (x < 3)"),
    parse("yScissors"): parse("(y >= 2) & (y < 3)"),

    # Rules
    parse("PaperBeatsRock"): parse("(xRock) -> (~yPaper)"),
    parse("ScissorsBeatsPaper"): parse("(xPaper) -> (~yScissors)"),
    parse("RockBeatsScissors"): parse("(xScissors) -> (~yRock)"),
    parse("Rules"): parse(
        "(PaperBeatsRock) & (ScissorsBeatsPaper) & (RockBeatsScissors)"),

    # Bounds
    parse("uBounds"): parse("(u >= 0) & (u <= 1)"),
    parse("wBounds"): parse("(w >= 0) & (w <= 1)"),
}

spec = G.Specs(
    obj=stl.utils.inline_context(parse("~yPaper"), context),
    learned=stl.TOP,
    init=stl.utils.inline_context(parse("Init"), context),
    #bounds=stl.utils.inline_context(
    #    parse("G((uBounds) & (wBounds))"), context),
    bounds=stl.TOP,
)


A, B, C = model.dyn
A = np.eye(A.shape[0]) + A
L_u = magnum.utils.dynamics_lipschitz(A, B, model.H)
L_w = magnum.utils.dynamics_lipschitz(A, C, model.H)
L_x = stl.utils.linear_stl_lipschitz(spec.obj)

meta = G.Meta(
    pri={},
    names={},
    drdu=L_u*L_x,
    drdw=L_w*L_x,
)


rps = G.Game(spec=spec, model=model, meta=meta)
