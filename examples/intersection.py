import stl

from sympy import Symbol

from magnum import game as G
from magnum import io
from magnum.solvers.cegis import cegis_loop

import numpy as np

model = G.Model(
    dt=1,
    N=4,
    vars=G.Vars(
        state=(
            Symbol("x"),
            Symbol("vx"),
            Symbol("y"),
            Symbol("vy")
        ),
        input=(Symbol("u"),),
        env=(Symbol("w"),)
    ),
    bounds={
        Symbol("u"): (0, 1),
        Symbol("w"): (0, 1),
    },
    t=0,
    dyn=G.Dynamics(
        A=np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ]),
        B=np.array([0, 10, 0, 0]).reshape((4,1)),
        C=np.array([0, 0, 0, 10]).reshape((4,1)),
    )
)

parse = lambda x: stl.parse(x, H=model.N)

context = {
    # Init
    parse("xInit"): parse("x = -5"),
    parse("vxInit"): parse("vx = 0"),
    parse("yInit"): parse("y = -5"),
    parse("vyInit"): parse("vy = 0"),
    parse("Init"): parse(
        "(xInit) & (vxInit) & (yInit) & (vyInit)"),

    # General
    parse("xInIntersection"): parse("(x > 0) & (x < 5)"),
    parse("yNotInIntersection"): parse("(y <= 0) | (y >= 5)"),

    # Env Assumptions
    parse("obeySpeedLimitY"): parse("G(vy < 10.1)"),
    #"noReversing": "G(vy >= 10)",
    parse("noParking"): parse("F(y > 10)"),
    parse("A"): parse("(obeySpeedLimitY) & (noParking)"),

    # Guarantees
    parse("goal"): parse("F(x > 5)"),
    parse("dontCrash"): parse("G((xInIntersection) -> (yNotInIntersection))"),
    parse("obeySpeedLimitX"): parse("G(vx <= 10.1)"),
    parse("noReversing"): parse("G(vx >= 0)"),
    parse("G"): parse(
        "(goal) & (dontCrash) & (obeySpeedLimitX) & (noReversing)"),

    # Spec
    #parse("spec"): parse("(A) -> (G)"),
    parse("spec"): parse("A"),
}

spec = G.Specs(
    obj=stl.utils.inline_context(parse("(Init) & (spec)"), context),
    learned=stl.TOP
)


meta = G.Meta(
    pri={},
    names={},
    dxdu=float('inf'),
    dxdw=float('inf'),
    drdx=float('inf')
)


intersection = G.Game(spec=spec, model=model, meta=meta)

if __name__ == '__main__':
    with open('intersection.bin', 'wb') as f:
        io.write(intersection, f)


    _intersection = G.discretize_game(intersection)
    #for _, (r, s) in zip(range(3), cegis_loop(_intersection)):
    #    print('-------------')
    #    print(r)
    #    print(s)
