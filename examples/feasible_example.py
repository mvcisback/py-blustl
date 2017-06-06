import stl

from sympy import Symbol

from magnum import game as G
from magnum import io

import numpy as np

## Setup the Model

model = G.Model(
    dt=0.01,
    H=2,
    vars=G.Vars(
        state=(Symbol("x"),),
        input=(Symbol("u"),),
        env=()
    ),
    t=0,
    dyn=G.Dynamics(
        A=np.array([[0]]),
        B=np.array([[10]]),
        C=np.array([[]])
    )
)

# Setup the specificatoion

context = {
    stl.parse("Init"): stl.parse("x = 0"),
    stl.parse("ReachFive"): stl.parse("F(x > 5)", H=model.H),
}

spec = G.Specs(
    obj=stl.utils.inline_context(stl.parse("ReachFive"), context),
    init=stl.utils.inline_context(stl.parse("Init"), context),
    bounds=stl.parse("G((u <= 1) & (u >= 0))", H=model.H),
    learned=stl.TOP
)


meta = G.Meta(
    pri={},
    names={},
    dxdu=None,
    dxdw=None,
    drdx=None
)


feasible_example = G.Game(spec=spec, model=model, meta=meta)

