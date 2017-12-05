import stl

from magnum import game as G

import numpy as np

## Setup the Model
H=3
model = G.Model(
    dt=1,

    vars=G.Vars(state=("x", ), input=("u", ), env=()),
    dyn=G.Dynamics(A=np.array([[0]]), B=np.array([[10]]), C=np.array([[]])))

# Setup the specification

context = {
    stl.parse("Init"): stl.parse("x = 0"),
    stl.parse("ReachFive"): stl.parse("(F(x > 5)) & (G((x < 5) -> (F[0,3](x < 3))))", H=H),
}

spec = G.Specs(
    obj=stl.parse("ReachFive").inline_context(context),
    init=stl.parse("Init").inline_context(context),
    learned=stl.TOP,
)

feasible_example = G.Game(specs=spec, model=model)
