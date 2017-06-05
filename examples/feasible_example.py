import stl

from sympy import Symbol

from magnum import game as G
from magnum import io

import numpy as np

## Setup the Model

model = G.Model(
    dt=1,
    N=2,
    vars=G.Vars(
        state=(Symbol("x"),),
        input=(Symbol("u"),),
        env=()
    ),
    bounds={
        Symbol("x"): (0, 100),
        Symbol("u"): (0, 1),
    },
    t=0,
    dyn=G.Dynamics(
        A=np.array([[0]]),
        B=np.array([[5]]),
        C=np.array([[]])
    )
)

# Setup the specificatoion

context = {
    stl.parse("Init"): stl.parse("x = 0"),
    stl.parse("ReachFive"): stl.parse("F(x > 5)"),
}

spec = G.Specs(
    obj=stl.utils.inline_context(stl.parse("(Init) & (ReachFive)"), context),
    learned=[]
)


meta = G.Meta(
    pri={},
    names={},
    dxdu=None,
    dxdw=None,
    drdx=None
)


feasible_example = G.Game(spec=spec, model=model, meta=meta)

if __name__ == '__main__':
    with open("feasible_example.bin", "wb") as f:
        io.write(feasible_example, f)
