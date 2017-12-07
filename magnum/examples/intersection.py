import stl

import magnum
from magnum import game as G

import numpy as np

H = 4

model = G.Model(
    dt=1,
    vars=G.Vars(state=("x", "vx", "y", "vy"), input=("u", ), env=("w", )),
    t=0,
    dyn=G.Dynamics(
        A=np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]),
        B=np.array([0, 10, 0, 0]).reshape((4, 1)),
        C=np.array([0, 0, 0, 10]).reshape((4, 1)),
    ))


def parse(x):
    return stl.parse(x, H=model.H)


context = {
    # Init
    parse("xInit"): parse("x = -5"),
    parse("vxInit"): parse("vx = 0"),
    parse("yInit"): parse("y = -5"),
    parse("vyInit"): parse("vy = 0"),
    parse("Init"): parse("(xInit) & (vxInit) & (yInit) & (vyInit)"),

    # Intersection
    parse("xInIntersection"): parse("(x > 0) & (x < 5)"),
    parse("yInIntersection"): parse("(y > 0) & (y < 5)"),
    parse("crash"): parse("(xInIntersection) & (yInIntersection)"),

    # Goals
    parse("reachDest"): parse("F(x > 5)"),
    parse("dontCrash"): parse("G(~(crash))"),

    # Env Assumptions
    parse("obeySpeedLimitY"): parse("G(vy <= 10.1)"),
    parse("A"): parse("(obeySpeedLimitY)"),

    # Guarantees
    parse("G"): parse("(reachDest) & (dontCrash)"),

    # Bounds
    parse("uBounds"): parse("(u >= 0) & (u <= 1)"),
    parse("wBounds"): parse("(w >= 0) & (w <= 1)"),

    # Spec
    parse("spec"): parse("(A) -> (G)"),
}

spec = G.Specs(
    obj=stl.utils.inline_context(parse("(A) -> (G)"), context),
    learned=stl.TOP,
    init=stl.utils.inline_context(parse("(Init)"), context),
    bounds=stl.utils.inline_context(
        parse("G((uBounds) & (wBounds))"), context),
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

intersection = G.Game(spec=spec, model=model, meta=meta)
