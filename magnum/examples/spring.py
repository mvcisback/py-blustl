import stl

from magnum import game as G

import numpy as np

## Setup the Model
H=8

model = G.Model(
    dt=0.1,

    vars=G.Vars(state=("x1", "x2"), input=("u", ), env=("w1", "w2")),
    dyn=G.Dynamics(A=np.array([[0, 1],[-1, -2]]), 
                   B=np.array([[20],[0]]), 
                   C=np.array([[1, 0],[0, 1]])))

# Setup the specification

context = {
    stl.parse("Init"): stl.parse("(x1 = 4.2) & (x2 = 0)"),
    stl.parse("Sys"): stl.parse("F[0, 3]((x1 >= 5) & (x1 <= 6))", H=1),
}

spec = G.Specs(
    obj=stl.parse("Sys").inline_context(context),
    init=stl.parse("Init").inline_context(context),
    learned=stl.TOP,
)

g = G.Game(specs=spec, model=model)
