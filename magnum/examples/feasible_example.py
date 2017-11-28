import stl

from magnum import game as G

import numpy as np

## Setup the Model

model = G.Model(
    #dt=0.01,
    dt=0.5,
    H=1,
    vars=G.Vars(state=("x", ), input=("u", ), env=()),
    dyn=G.Dynamics(A=np.array([[0]]), B=np.array([[10]]), C=np.array([[]])))

# Setup the specification

context = {
    stl.parse("Init"): stl.parse("x = 0"),
    stl.parse("ReachFive"): stl.parse("F(x > 5)", H=model.H),
}

spec = G.Specs(
    obj=stl.parse("ReachFive").inline_context(context),
    init=stl.parse("Init").inline_context(context),
    dyn=stl.TOP,
    learned=stl.TOP,
    bounds=stl.parse("G((u <= 1) & (u >= 0))", H=model.H),
)

meta = G.Meta(
    pri={},
    names={},
    drdu=None,
    drdw=None,
)

feasible_example = G.Game(specs=spec, model=model, meta=meta)
